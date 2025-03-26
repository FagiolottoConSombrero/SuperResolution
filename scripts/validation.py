from engine import *
from dataset import *
from models import *
from torch.utils.data import DataLoader
import argparse
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import tifffile as tiff
import os

parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument("--model", default="checkpoint/model_epoch_600.pth", type=str, help="model path")
parser.add_argument("--results", default="results", type=str, help="Result save location")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")
parser.add_argument('--data_path', type=str, default='/home/matteo/Downloads/pirm/hd5', help='Dataset path')
parser.add_argument('--batchSize', type=int, default='32', help='Training batch size')


def MSE(gt, rc):
    return np.mean((gt - rc) ** 2)


def PSNR(gt, rc):
    mse = MSE(gt, rc)
    pmax = 1
    return 10 * np.log10(pmax ** 2 / mse)


def MRAE(gt, rc):
    return np.mean(np.abs(gt - rc) / (gt + 1e-3))


def SID(gt, rc):
    epsilon = 1e-3  # puoi modificare questo valore se necessario
    max_val = 1.2095
    gt_norm = gt / np.clip(np.sum(gt, axis=0, keepdims=True), a_min=epsilon, a_max=None)
    rc_norm = rc / np.clip(np.sum(rc, axis=0, keepdims=True), a_min=epsilon, a_max=None)

    divergence = (gt_norm * np.log(np.clip(gt_norm / rc_norm, a_min=epsilon, a_max=max_val)) +
                  rc_norm * np.log(np.clip(rc_norm / gt_norm, a_min=epsilon, a_max=max_val)))
    return np.mean(np.sum(divergence, axis=0))


def APPSA(gt, rc):
    dot_product = np.sum(gt * rc, axis=0)
    norm_gt = np.sqrt(np.sum(gt ** 2, axis=0))
    norm_pred = np.sqrt(np.sum(rc ** 2, axis=0))

    cos_theta = np.clip(dot_product / (norm_gt * norm_pred + 1e-6), -1, 1)
    spectral_agle = np.arccos(cos_theta)
    return np.mean(spectral_agle)


def SSIM(gt, rc):
    ssim_values = []
    for i in range(gt.shape[0]):
        ssim_value = compare_ssim(
                rc[i],
                gt[i],
                data_range=rc[i].max() - rc[i].min(),  # Explicit data range
                channel_axis=0  # Specify the channel axis
            )
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)


# First element is the ground truth, second is the prediction
measures = {
    'APPSA': APPSA,
    'SID': SID,
    'PSNR': PSNR,
    'SSIM': SSIM,
    'MSE': MSE,
    'MRAE': MRAE
}

# Dizionario per memorizzare il miglior valore e l'indice della migliore immagine per ogni metrica
best_measures = {k: (np.inf if k in ['MSE', 'SID', 'MRAE', 'APPSA'] else -np.inf, -1) for k in measures}

opt = parser.parse_args()
model = SPAN(14, 14)
model.load_state_dict(torch.load(opt.model, weights_only=True))

valid_set = Hdf5Dataset(opt.data_path, training=False, transforms=get_transforms())
valid_loader = DataLoader(dataset=valid_set, batch_size=opt.batchSize, shuffle=True)

model = model.to(opt.device)
summed_measures = None
save_dir = "/home/matteo/PycharmProjects/SuperResolution/data/track_1"

print("===> Validation")
for iteration, (x, gt) in enumerate(valid_loader, 1):
    x = x.to(opt.device)
    output = model(x)
    output = output.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    hr_measures = {k: np.array(func(gt, output)) for (k, func) in measures.items()}

    print(f"\n===== Image {iteration} =====")
    for key, value in hr_measures.items():
        print(f"{key} (original): {np.mean(value):.6f}")  # Stampa media per evitare array multipli

    # Salvataggio delle immagini in formato TIFF
    gt_filename = os.path.join(save_dir, f"GT_{iteration:03d}.tif")
    sr_filename = os.path.join(save_dir, f"SR_{iteration:03d}.tif")

    tiff.imwrite(gt_filename, gt.astype(np.float32), dtype=np.float32)
    tiff.imwrite(sr_filename, output.astype(np.float32), dtype=np.float32)

    print(f"Immagini salvate: {gt_filename}, {sr_filename}")

    if summed_measures is None:
        summed_measures = hr_measures
    else:
        summed_measures = {k: v + hr_measures[k] for (k, v) in summed_measures.items()}

    # Controlliamo il miglior caso per ogni metrica
    for k, v in hr_measures.items():
        best_value, best_index = best_measures[k]
        if (k in ['PSNR', 'SSIM'] and v > best_value) or (k in ['MSE', 'SID', 'MRAE', 'APPSA'] and v < best_value):
            best_measures[k] = (v, iteration)

# Calcolo della media
summed_measures = {k: v / len(valid_loader) for (k, v) in summed_measures.items()}

# Stampiamo le medie
print("\n===== Average Measures =====")
for key, value in summed_measures.items():
    print(f"{key}: {value:.6f}")

# Stampiamo i migliori casi
print("\n===== Best Cases =====")
for key, (value, index) in best_measures.items():
    print(f"{key}: {value:.6f} (Image {index})")