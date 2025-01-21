import numpy as np
import os
import scipy.io as sio
import torch
from torchvision.transforms import v2
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from torch import nn


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        """
        Calculate the Structural Similarity Index (SSIM) loss.

        Args:
            pred (torch.Tensor): Predicted image, shape (B, C, H, W) or (B, C, H).
            target (torch.Tensor): Target image, shape (B, C, H, W) or (B, C, H).

        Returns:
            torch.Tensor: 1 - mean SSIM loss.
        """
        ssim_values = []
        pred = pred.detach().cpu().numpy()  # Convert to NumPy
        target = target.detach().cpu().numpy()

        for i in range(pred.shape[0]):  # Loop over batch
            ssim_value = compare_ssim(
                pred[i],
                target[i],
                data_range=pred[i].max() - pred[i].min(),  # Explicit data range
                channel_axis=0  # Specify the channel axis
            )
            ssim_values.append(ssim_value)

        # Return 1 - mean SSIM (since SSIM ranges [0, 1])
        return 1 - torch.tensor(ssim_values, dtype=torch.float32).mean()


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
        epsilon = 1e-3

        # Take the absolute value of target to avoid negative values
        target = torch.abs(target)

        # Compute relative error
        relative_error = torch.abs(predicted - target) / (target + epsilon)

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
        epsilon = 1e-8

        # Normalize along the spectral dimension (assume C is spectral channels)
        predicted = predicted / (torch.sum(predicted, dim=1, keepdim=True) + epsilon)
        target = target / (torch.sum(target, dim=1, keepdim=True) + epsilon)

        # Clamp to prevent log(0)
        predicted = torch.clamp(predicted, min=epsilon)
        target = torch.clamp(target, min=epsilon)

        # Compute divergence terms
        divergence_1 = predicted * (torch.log(predicted) - torch.log(target))
        divergence_2 = target * (torch.log(target) - torch.log(predicted))

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


def scale_0_1(x):
    return x/255


def get_transforms(mean, std):
    """
    Crea una pipeline di trasformazioni normalizzando i valori con mean e std scalati.
    """
    scaled_mean = [m / 255.0 for m in mean]
    scaled_std = [s / 255.0 for s in std]

    # Trasformazioni
    basic_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Lambda(scale_0_1),
        v2.Normalize(mean=scaled_mean, std=scaled_std)  # Applica normalizzazione
    ])
    return basic_transforms


def get_input_transforms():
    """
    Restituisce trasformazioni separate per input e target, utilizzando mean e std definiti.
    """
    # Mean e std per input
    input_mean = [53.63522448, 58.30869008, 57.10608313, 108.39563089, 111.95032318,
                  105.9614267, 90.19320486, 122.35541551, 116.40579098, 107.22275369,
                  71.46360708, 89.26980523, 62.22486445, 55.22971295]
    input_std = [36.34952606, 42.07507617, 41.42893763, 74.13830411, 76.55226538,
                 72.69417164, 62.22261903, 85.1147687, 82.00505138, 74.16726475,
                 48.74070934, 61.01281974, 44.47178134, 40.2399229]
    # Creazione delle trasformazioni
    input_transform = get_transforms(input_mean, input_std)

    return input_transform


def get_target_transforms():
    """
    Restituisce trasformazioni separate per input e target, utilizzando mean e std definiti.
    """
    # Mean e std per target
    target_mean = [22.857117, 24.836193, 24.325882, 46.193604, 47.71157,  45.155228, 38.434223,
                   52.14261,  49.610733, 45.696888, 30.455795, 38.04653,  26.50503,  23.525055]
    target_std = [11.985463, 14.316695, 14.093582, 24.880175, 25.645956, 24.483856, 21.006227,
                  28.719099, 27.773434, 24.99851,  16.294231, 20.485502, 15.148599, 13.781046]

    target_transform = get_transforms(target_mean, target_std)

    return target_transform


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