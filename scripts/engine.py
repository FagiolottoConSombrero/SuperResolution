import torch
from tqdm import tqdm
from utils import *


def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def training_step(train_loader, model, optimizer, device, ssim_loss_fn, sam_loss_fn):
    model.train()
    ssim_loss_total = 0
    sam_loss_total = 0
    combined_loss_total = 0
    loop = tqdm(train_loader, desc="Training", leave=True, dynamic_ncols=True)

    for batch, (X, y) in enumerate(loop):
        # Send data to GPU
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Forward Pass
        y_pred = model(X)

        # Calculate losses
        loss_1 = ssim_loss_fn(y_pred, y)
        loss_2 = sam_loss_fn(y_pred, y)
        loss = loss_1 + loss_2

        # Accumulate loss
        ssim_loss_total += loss_1.item()
        sam_loss_total += loss_2.item()
        combined_loss_total += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # **Libera memoria per evitare il CUDA Out Of Memory**
        del X, y, y_pred, loss, loss_1, loss_2  # Cancella i tensori per liberare memoria
        torch.cuda.empty_cache()  # Libera la cache di PyTorch
        torch.cuda.ipc_collect()  # Ottimizza l'allocazione della memoria

    # Calcola le medie delle perdite
    combined_loss_total /= len(train_loader)
    ssim_loss_total /= len(train_loader)
    sam_loss_total /= len(train_loader)

    return combined_loss_total, ssim_loss_total, sam_loss_total


def validation_step(val_loader, model, device, ssim_loss_fn, sam_loss_fn):
    model.eval()
    ssim_loss_total = 0
    sam_loss_total = 0
    combined_loss_total = 0
    loop = tqdm(val_loader, desc="Validation", leave=True)

    with torch.no_grad():
        for batch, (X, y) in enumerate(loop):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(X)

            # Calculate losses
            loss_1 = ssim_loss_fn(y_pred, y)
            loss_2 = sam_loss_fn(y_pred, y)
            loss = loss_1 + loss_2

            # Accumulate loss
            ssim_loss_total += loss_1.item()
            sam_loss_total += loss_2.item()
            combined_loss_total += loss.item()

            # **Libera memoria per evitare il CUDA Out Of Memory**
            del X, y, y_pred, loss, loss_1, loss_2
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Calcola le medie delle perdite
    combined_loss_total /= len(val_loader)
    ssim_loss_total /= len(val_loader)
    sam_loss_total /= len(val_loader)

    return combined_loss_total, ssim_loss_total, sam_loss_total


def train(train_loader,
          val_loader,
          model,
          epochs,
          optimizer,
          device,
          best_model_path,
          ssim_loss_fn,
          sam_loss_fn,
          initial_lr,
          patience=20):
    early_stopping = EarlyStopping(patience=patience, mode='min')

    for epoch in tqdm(range(epochs), desc="All"):
        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, epoch, initial_lr)
        print(f"Epoch: {epoch}, Learning Rate: {current_lr:.6f}\n-----------")
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        train_loss = training_step(train_loader, model, optimizer, device, ssim_loss_fn, sam_loss_fn)
        val_loss = validation_step(val_loader, model, device, ssim_loss_fn, sam_loss_fn)

        print(
            f"Train combined_loss: {train_loss[0]:.4f} | "
            f"Train MRAE_loss: {train_loss[1]:.4f} | "
            f"Train SID_loss: {train_loss[2]:.4f} | "
            f"Val combined_loss: {val_loss[0]:.4f} | "
            f"Val MRAE_loss: {val_loss[1]:.4f} | "
            f"Val SID_loss: {val_loss[2]:.4f} | "
        )
        print("-------------\n")

        # **Libera memoria dopo ogni epoca**
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if check_early_stopping(val_loss[0], model, early_stopping, epoch, best_model_path):
            break

    # Ripristina i pesi migliori
    model.load_state_dict(torch.load(best_model_path))
    print(f"Restored best model weights with val_loss: {early_stopping.best_val:.4f}")
