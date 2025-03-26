'''import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Percorsi delle immagini multispettrali
gt_path = "/home/matteo/PycharmProjects/SuperResolution/data/GT_003.tif"
sr_path = "/home/matteo/PycharmProjects/SuperResolution/data/SR_003.tif"

# Carica le immagini multispettrali
gt_image = tiff.imread(gt_path)
sr_image = tiff.imread(sr_path)

# Verifica il numero di bande
num_bands = gt_image.shape[0]

# Imposta la scala fissa della barra dei colori
vmin, vmax = 0, 1  # La SSIM varia sempre tra 0 e 1

# Loop su ogni banda per calcolare e visualizzare solo la SSIM
for i in range(num_bands):
    # Estrai la banda corrispondente e ruotala di 90° verso destra
    gt_band = np.rot90(gt_image[i, :, :], k=-1)
    sr_band = np.rot90(sr_image[i, :, :], k=-1)

    # Calcola SSIM e genera la mappa di similarità
    ssim_value, ssim_map = ssim(gt_band, sr_band, full=True, data_range=gt_band.max() - gt_band.min())

    # Visualizza solo la SSIM Map con colormap invertita e scala fissa
    plt.figure(figsize=(5, 5))
    im = plt.imshow(ssim_map, cmap="cividis_r", vmin=vmin, vmax=vmax)  # Fissa la scala colori tra 0 e 1
    plt.title(f"SSIM Map - Banda {i + 1}\nSSIM: {ssim_value:.4f}")
    plt.axis("off")

    # Barra dei colori con scala fissa
    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.5, 1])  # Imposta i valori sulla barra

    plt.show()


import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Percorsi delle immagini multispettrali
gt_path = "/home/matteo/PycharmProjects/SuperResolution/data/GT_010.tif"
sr_path = "/home/matteo/PycharmProjects/SuperResolution/data/SR_010.tif"

# Carica le immagini multispettrali
gt_image = tiff.imread(gt_path)
sr_image = tiff.imread(sr_path)

# Verifica il numero di bande
num_bands = gt_image.shape[0]

# Loop su ogni banda per calcolare e visualizzare solo MRAE
for i in range(num_bands):
    # Estrai la banda e ruotala di 90° verso destra
    gt_band = np.rot90(gt_image[i, :, :], k=-1)
    sr_band = np.rot90(sr_image[i, :, :], k=-1)

    # Calcola MRAE (Mean Relative Absolute Error)
    epsilon = 1e-6  # Evita divisioni per zero
    mrae_map = np.abs(gt_band - sr_band) / (gt_band + epsilon)

    # Determina i limiti della scala (Min e Max dell'errore)
    vmin = np.min(mrae_map)
    vmax = np.max(mrae_map)

    # Visualizza la mappa MRAE
    plt.figure(figsize=(5, 5))
    im = plt.imshow(mrae_map, cmap="magma", vmin=0, vmax=1)  # Colormap che enfatizza gli errori
    plt.title(f"MRAE Map - Banda {i+1}")
    plt.axis("off")

    # Barra dei colori con scala basata su min e max reali
    cbar = plt.colorbar(im)
    cbar.set_label("MRAE")

    plt.show()


import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Percorsi delle immagini multispettrali
gt_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2/GT_010.tif"
sr_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2/SR_010.tif"

# Carica le immagini multispettrali
gt_image = tiff.imread(gt_path)
sr_image = tiff.imread(sr_path)

# Verifica il numero di bande
num_bands = gt_image.shape[0]

# Loop su ogni banda per calcolare e visualizzare solo RMSE
for i in range(num_bands):
    # Estrai la banda e ruotala di 90° verso destra
    gt_band = np.rot90(gt_image[i, :, :], k=-1)
    sr_band = np.rot90(sr_image[i, :, :], k=-1)

    # Calcola RMSE (Root Mean Square Error)
    rmse_map = np.sqrt((gt_band - sr_band) ** 2)

    # Determina i limiti della scala (Min e Max dell'errore)
    vmin = np.min(rmse_map)
    vmax = np.max(rmse_map)

    # Visualizza la mappa RMSE
    plt.figure(figsize=(5, 5))
    im = plt.imshow(rmse_map, cmap="inferno", vmin=vmin, vmax=vmax)  # Colormap che enfatizza errori alti
    plt.title(f"RMSE Map - Banda {i+1}")
    plt.axis("off")

    # Barra dei colori con scala basata su min e max reali
    cbar = plt.colorbar(im)
    cbar.set_label("RMSE")

    plt.show()

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Percorsi delle immagini multispettrali
gt_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2/GT_009.tif"
sr_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2_rgb/SR_009.tif"

# Carica le immagini multispettrali
gt_image = tiff.imread(gt_path)
sr_image = tiff.imread(sr_path)

# Verifica il numero di bande
num_bands = gt_image.shape[0]

# Loop su ogni banda per calcolare e visualizzare l'Absolute Difference
for i in range(num_bands):
    # Estrai la banda e ruotala di 90° verso destra
    gt_band = np.rot90(gt_image[i, :, :], k=-1)
    sr_band = np.rot90(sr_image[i, :, :], k=-1)

    # Calcola Absolute Difference
    abs_diff = np.abs(gt_band - sr_band)

    # Determina i limiti della scala
    vmin = np.min(abs_diff)
    vmax = np.max(abs_diff)

    # Visualizza la mappa di errore (Absolute Difference)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax[0].imshow(abs_diff, cmap="magma", vmin=vmin, vmax=vmax)
    ax[0].set_title(f"Absolute Difference - Banda {i+1}")
    ax[0].axis("off")
    plt.colorbar(im1, ax=ax[0])

    # Istogramma dei valori di errore assoluto
    ax[1].hist(abs_diff.ravel(), bins=50, color="black", alpha=0.8)
    ax[1].set_title("Histogram of Residuals")
    ax[1].set_xlabel("Absolute Error")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()'''


import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec

# Percorsi delle immagini multispettrali
gt_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2/GT_003.tif"
sr_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2_rgb/SR_003.tif"

# Cartella di salvataggio
save_dir = "/home/matteo/PycharmProjects/SuperResolution/data/comparison/image_03/rgbsr"
os.makedirs(save_dir, exist_ok=True)

# Carica le immagini multispettrali
gt_image = tiff.imread(gt_path)
sr_image = tiff.imread(sr_path)

# Bande da analizzare (1-based indexing)
bands_to_process = [1, 6, 14]

# Loop solo sulle bande desiderate
for i in range(gt_image.shape[0]):
    band_id = i + 1
    if band_id not in bands_to_process:
        continue

    # Estrai e ruota le bande
    gt_band = np.rot90(gt_image[i, :, :], k=-1)
    sr_band = np.rot90(sr_image[i, :, :], k=-1)

    # Calcola la differenza assoluta
    abs_diff = np.abs(gt_band - sr_band)
    vmin = np.min(abs_diff)
    vmax = np.max(abs_diff)

    # === Salva mappa dell'errore assoluto ===
    fig1, ax1 = plt.subplots(figsize=(10, 9))
    im1 = ax1.imshow(abs_diff, cmap="magma", vmin=vmin, vmax=vmax)
    ax1.set_title(f"Absolute Difference - Band {band_id}")
    ax1.axis("off")
    cbar = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, f"abs_diff_band_{band_id}.png"))
    plt.close(fig1)

    # === Salva istogramma dei residui ===
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    ax2.hist(abs_diff.ravel(), bins=50, color="black", alpha=0.8)
    ax2.set_title(f"Histogram of Residuals - Band {band_id}")
    ax2.set_xlabel("Absolute Error")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"residual_hist_band_{band_id}.png"))
    plt.close(fig2)




