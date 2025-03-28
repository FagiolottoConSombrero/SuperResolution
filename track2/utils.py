import os
import torch
from torchvision.transforms import v2
from skimage.metrics import structural_similarity as compare_ssim
from torch import nn
from pytorch_msssim import ssim


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    Utilizza pytorch_msssim per calcolare la similarità strutturale tra predizione e target.
    La loss è definita come (1 - SSIM), in modo da essere minimizzata.
    """
    def __init__(self, normalize=True):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        # SSIM restituisce un valore in [0, 1]; 1 = immagini identiche
        ssim_value = ssim(pred, target)
        return 1 - ssim_value  # Più simile sono, più la loss si avvicina a 0


class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, pred, target):
        dot_product = torch.sum(pred * target, dim=1)
        norm_pred = torch.norm(pred, p=2, dim=1)
        norm_target = torch.norm(target, p=2, dim=1)
        angle = torch.acos(dot_product / (norm_pred * norm_target + 1e-8))

        return torch.mean(angle)


class MRAELoss(nn.Module):
    def __init__(self):
        super(MRAELoss, self).__init__()

    def forward(self, predicted, target):
        """
        Calculate the Mean Relative Absolute Error (MRAE) loss.

        Args:
            predicted (torch.Tensor): Predicted values, shape (B, C, H, W) or (B, C).
            target (torch.Tensor): Ground truth values, shape (B, C, H, W) or (B, C).

        Returns:
            torch.Tensor: MRAE loss.
        """
        # Add a small value to avoid division by zero
        #epsilon = 1e-8

        # Compute relative error
        relative_error = torch.abs(predicted - target) / (target + 1.0/65535.0)

        # Return mean relative error
        return torch.mean(relative_error)



class SIDLoss(nn.Module):
    def __init__(self):
        super(SIDLoss, self).__init__()

    def forward(self, predicted, target):
        """
        Calculate the Spectral Information Divergence (SID) loss.

        Args:
            predicted (torch.Tensor): Predicted spectra, shape (B, C, H, W) or (B, C).
            target (torch.Tensor): Ground truth spectra, shape (B, C, H, W) or (B, C).

        Returns:
            torch.Tensor: SID loss.
        """
        epsilon = 1e-3

        predicted = torch.clamp(predicted, min=epsilon)
        target = torch.clamp(target, min=epsilon)

        a1 = predicted * torch.log10((predicted + epsilon) / (target + epsilon))
        a2 = target * torch.log10((target + epsilon) / (predicted + epsilon))

        a1_sum = a1.sum(dim=3).sum(dim=2)
        a2_sum = a2.sum(dim=3).sum(dim=2)

        errors = torch.abs(a1_sum + a2_sum)

        return torch.mean(errors)


def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(model.state_dict(), model_out_path)
    print(f"Saved checkpoint to {model_out_path}")


def check_early_stopping(val_loss, model, early_stopping, epoch, best_model_path):
    if early_stopping.best_val is None or val_loss < early_stopping.best_val:
        early_stopping.best_val = val_loss  # Aggiorna la migliore loss
        early_stopping.counter = 0  # Resetta il contatore
        torch.save(model.state_dict(), best_model_path)  # Salva i pesi migliori
        print(f"Saved best model with val_loss: {val_loss:.4f} at epoch {epoch}")
        return False  # Non fermare il training
    else:
        early_stopping.counter += 1
        if early_stopping.counter >= early_stopping.patience:
            print(f"Early stopping at epoch {epoch} with best val_loss: {early_stopping.best_val:.4f}")
            return True  # Fermare il training
    return False  # Continuare il training


def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    return lr

class EarlyStopping():
    """
    stop the training when the loss does not improve.
    """
    def __init__(self, patience=50, mode='min'):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_val = None

    def __call__(self, val):
        val = float(val)
        if self.best_val is None:
            self.best_val = val
        elif self.mode == 'min' and val < self.best_val:
            self.best_val = val
            self.counter = 0
        elif self.mode == 'max' and val > self.best_val:
            self.best_val = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early Stopping!")
                return True
        return False


def scale_0_1(x):
    x = torch.tensor(x, dtype=torch.float32)
    return x/65535.0


def get_transforms():
    """
    Crea una pipeline di trasformazioni normalizzando i valori con mean e std scalati.
    """
    # Trasformazioni
    basic_transforms = v2.Compose([
        v2.Lambda(scale_0_1),
    ])
    return basic_transforms