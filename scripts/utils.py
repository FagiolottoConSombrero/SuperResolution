import numpy as np
import os
import scipy.io as sio
import torch
from torchvision.transforms import v2
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from torch import nn


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        ssim_values = []
        for i in range(pred.shape[0]):  # Loop over batch
            ssim_value = compare_ssim(pred[i].cpu().numpy().transpose(1, 2, 0),
                                      target[i].cpu().numpy().transpose(1, 2, 0),
                                      multichannel=True)
            ssim_values.append(ssim_value)
        return 1 - torch.tensor(ssim_values, dtype=torch.float32).mean()  # SSIM ranges [0, 1]


class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, pred, target):
        # Flatten the spectral data
        pred = pred.view(pred.shape[0], -1, pred.shape[-1])  # (Batch, Pixels, Channels)
        target = target.view(target.shape[0], -1, target.shape[-1])  # (Batch, Pixels, Channels)

        dot_product = torch.sum(pred * target, dim=-1)
        pred_norm = torch.norm(pred, dim=-1)
        target_norm = torch.norm(target, dim=-1)
        cos_theta = dot_product / (pred_norm * target_norm + 1e-8)
        sam_angle = torch.acos(torch.clamp(cos_theta, -1, 1))  # Ensure values are in valid range

        return sam_angle.mean()


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
        epsilon = 1e-8
        relative_error = torch.abs(predicted - target) / (target + epsilon)
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
        epsilon = 1e-8

        # Normalize along the spectral dimension (assume C is spectral channels)
        predicted = predicted / (torch.sum(predicted, dim=1, keepdim=True) + epsilon)
        target = target / (torch.sum(target, dim=1, keepdim=True) + epsilon)

        # Compute divergence terms
        divergence_1 = predicted * (torch.log(predicted + epsilon) - torch.log(target + epsilon))
        divergence_2 = target * (torch.log(target + epsilon) - torch.log(predicted + epsilon))

        # Sum along spectral channels and mean over the batch
        return torch.mean(torch.sum(divergence_1 + divergence_2, dim=1))



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
    def __init__(self, patience=20, mode='min'):
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


def calculate_mean_std(dataset):
    """
    Calcola la media e la deviazione standard del dataset su tutti i canali.
    """
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count())
    mean = 0.0
    std = 0.0
    total_samples = 0

    for data, _ in loader:  # Supponiamo che il dataset restituisca (input, target)
        batch_samples = data.size(0)  # Numero di immagini nel batch
        data = data.view(batch_samples, data.size(1), -1)  # Ridimensiona a (batch, channel, H*W)
        mean += data.mean(2).sum(0)  # Somma la media su ogni canale
        std += data.std(2).sum(0)  # Somma la deviazione standard su ogni canale
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.numpy(), std.numpy()

def get_transforms():
    # Medie e deviazioni standard scalate per il range [0, 1]
    mean = [53.63522448, 58.30869008, 57.10608313, 108.39563089, 111.95032318,
            105.9614267, 90.19320486, 122.35541551, 116.40579098, 107.22275369,
            71.46360708, 89.26980523, 62.22486445, 55.22971295]
    std = [36.34952606, 42.07507617, 41.42893763, 74.13830411, 76.55226538,
           72.69417164, 62.22261903, 85.1147687, 82.00505138, 74.16726475,
           48.74070934, 61.01281974, 44.47178134, 40.2399229]

    # Scala mean e std al range [0, 1]
    scaled_mean = [m / 255.0 for m in mean]
    scaled_std = [s / 255.0 for s in std]

    # Trasformazioni con ordine corretto
    basic_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),  # Converte numpy.ndarray in torch.Tensor con valori [0, 1]
        v2.Normalize(mean=scaled_mean, std=scaled_std)  # Applica normalizzazione
    ])
    return basic_transforms


def mat2npy(dirs):
    for source_dir in dirs:
        # Iterate over all .mat files in the directory
        for filename in os.listdir(source_dir):
            if filename.startswith("._"):
                continue

            if filename.endswith(".mat"):
                file_path = os.path.join(source_dir, filename)

                try:
                    # Load the .mat file
                    mat_data = sio.loadmat(file_path)

                    # Check if 'replaced' variable exists
                    if 'bands' in mat_data:
                        replaced_data = mat_data['bands']

                        # Define the output .npy file path
                        output_path = os.path.join(source_dir, filename.replace(".mat", ".npy"))

                        # Save the replaced data to a .npy file
                        np.save(output_path, replaced_data)
                        print(f"Converted: {filename} -> {output_path}")

                        # Remove the original .mat file
                        os.remove(file_path)
                        print(f"Deleted original file: {file_path}")
                    else:
                        print(f"'replaced' variable not found in {filename}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")