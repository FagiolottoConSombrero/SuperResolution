import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, training=True, input_transform=None, target_transform=None):
        self.training = training
        self.input_transform = input_transform
        self.target_transform = target_transform

        # Configura i percorsi in base al flag di training
        base_path = '/Volumes/Lexar/PIRM/StereoMSI/data/MultispectralSR/SR/patches'
        if training:
            self.input_dir = os.path.join(base_path, 'train_preproced_patch')
            self.target_dir = os.path.join(base_path, 'train_demo_patch')
        else:
            self.input_dir = os.path.join(base_path, 'test_preproced_patch')
            self.target_dir = os.path.join(base_path, 'test_demo_patch')

        # Lista dei file comuni a entrambe le directory, escludendo quelli che iniziano con ._
        self.file_names = [
            f for f in os.listdir(self.input_dir)
            if f.endswith('.npy') and not f.startswith('._') and os.path.exists(os.path.join(self.target_dir, f))
        ]

        if not self.file_names:
            raise ValueError("Nessun file corrispondente trovato tra input e target directories.")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # Percorsi dei file
        input_path = os.path.join(self.input_dir, file_name)
        target_path = os.path.join(self.target_dir, file_name)

        # Caricamento delle immagini
        input_image = np.load(input_path)
        target_image = np.load(target_path)

        # Applica trasformazioni se definite
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target_image = self.target_transform(target_image)

        return input_image, target_image


