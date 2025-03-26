
'''
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os

# Percorsi file
sr_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2/SR_010.tif"
gt_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2/GT_010.tif"
rgb_path = "/home/matteo/PycharmProjects/SuperResolution/data/track_2_rgb/SR_010.tif"

# Cartella di salvataggio
save_dir = "/home/matteo/PycharmProjects/SuperResolution/data/comparison/image_10"
os.makedirs(save_dir, exist_ok=True)

# Carica immagini
sr_image = tiff.imread(sr_path)
gt_image = tiff.imread(gt_path)
rgb_image = tiff.imread(rgb_path)

# Bande da salvare (1-based indexing)
bands_to_save = [1, 6, 14]

# Loop su bande
for i in range(sr_image.shape[0]):
    band_id = i + 1
    if band_id not in bands_to_save:
        continue

    # Ruota entrambe le bande
    sr_band = np.rot90(sr_image[i, :, :], k=-1)
    gt_band = np.rot90(gt_image[i, :, :], k=-1)
    rgb_band = np.rot90(gt_image[i, :, :], k=-1)

    # === Salva SR ===
    fig_sr, ax_sr = plt.subplots(figsize=(6, 5), dpi=200)
    ax_sr.imshow(sr_band, cmap='gray')
    ax_sr.set_title(f"SR - Band {band_id}")
    ax_sr.axis("off")
    fig_sr.tight_layout()
    fig_sr.savefig(os.path.join(save_dir, f"sr_band_{band_id}.png"))
    plt.close(fig_sr)

    # === Salva GT ===
    fig_gt, ax_gt = plt.subplots(figsize=(6, 5), dpi=200)
    ax_gt.imshow(gt_band, cmap='gray')
    ax_gt.set_title(f"GT - Band {band_id}")
    ax_gt.axis("off")
    fig_gt.tight_layout()
    fig_gt.savefig(os.path.join(save_dir, f"gt_band_{band_id}.png"))
    plt.close(fig_gt)

    fig_gt, ax_gt = plt.subplots(figsize=(6, 5), dpi=200)
    ax_gt.imshow(rgb_band, cmap='gray')
    ax_gt.set_title(f"ColorSR - Band {band_id}")
    ax_gt.axis("off")
    fig_gt.tight_layout()
    fig_gt.savefig(os.path.join(save_dir, f"colorSR_band_{band_id}.png"))
    plt.close(fig_gt)

    '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Caricamento dati
df = pd.read_csv("/home/matteo/Downloads/pirm/model_comparison_task_1.csv")  # <-- metti il tuo path qui

# Filtra per i valori significativi (se necessario)
filtered_df = df.copy()

# Binning della colonna "Size" in 4 range numerici
size_bins = [0, 300, 1000, 5000, np.inf]
filtered_df["SizeCategory"] = pd.cut(filtered_df["Size"], bins=size_bins, labels=False)

# Mappa colori (palette viridis con 4 step)
color_palette = plt.get_cmap("viridis", 4)

# Definizione dei punti di rottura per asse X
gflops_min = 0
gflops_split1 = 130
gflops_split2 = 5330
gflops_max = filtered_df["#GFLOPs"].max()

# Dati per il primo e secondo segmento
df_ax1 = filtered_df[filtered_df["#GFLOPs"] <= gflops_split1]
df_ax2 = filtered_df[filtered_df["#GFLOPs"] >= gflops_split2]

# Crea sottoplot con asse spezzato
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13, 6), gridspec_kw={'width_ratios': [2, 1]})

# Definizione dimensioni in modo molto marcato
size_mapping = {
    0: 200,
    1: 300,
    2: 1500,
    3: 5000
}

# Applica la mappa per ogni SizeCategory
df_ax1["BubbleSize"] = df_ax1["SizeCategory"].map(size_mapping)
df_ax2["BubbleSize"] = df_ax2["SizeCategory"].map(size_mapping)

# Primo segmento
ax1.scatter(df_ax1["#GFLOPs"], df_ax1["PSNR"],
            c=df_ax1["SizeCategory"].map(lambda x: color_palette(x)),
            s=df_ax1["BubbleSize"],
            alpha=0.8, edgecolors='k', marker='o')

# Secondo segmento
ax2.scatter(df_ax2["#GFLOPs"], df_ax2["PSNR"],
            c=df_ax2["SizeCategory"].map(lambda x: color_palette(x)),
            s=df_ax2["BubbleSize"],
            alpha=0.8, edgecolors='k', marker='o')

# Etichette con offset verticale per non sovrapporsi
for ax, data in zip([ax1, ax2], [df_ax1, df_ax2]):
    for _, row in data.iterrows():
        ax.text(row["#GFLOPs"], row["PSNR"] + 0.1, row["Team"], fontsize=10,
                ha='center', va='bottom', weight='bold')

# Impostazioni limiti assi X
ax1.set_xlim(gflops_min, gflops_split1 + 10)
ax2.set_xlim(gflops_split2 - 200, gflops_max + 500)

# Etichette e titolo
ax1.set_xlabel("#GFLOPs")
ax2.set_xlabel("#GFLOPs")
ax1.set_ylabel("PSNR (dB)")
ax1.set_title("Flops vs PSNR vs Size")

# Aggiunta rottura asse X
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()
d = .015  # dimensione della "rottura"
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Legenda con range numerici
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label=f'{int(size_bins[i])}-{int(size_bins[i+1]) if size_bins[i+1] != np.inf else "+"}',
           markerfacecolor=color_palette(i), markersize=10, markeredgecolor='k')
    for i in range(4)
]
fig.legend(handles=legend_elements, title='Model Size (MB)', loc='upper right')

plt.tight_layout()
fig.savefig("/home/matteo/PycharmProjects/SuperResolution/data/comparison/flops_vs_psnr_vs_size.png", dpi=300, bbox_inches='tight')
plt.show()

