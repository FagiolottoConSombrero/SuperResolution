from torch.utils.data import Dataset
import h5py
import os
import numpy as np

class Hdf5Dataset(Dataset):
    def __init__(self, base_path='', training=True, transforms=None):
        super(Hdf5Dataset, self).__init__()

        self.training = training
        self.base = base_path
        self.transforms = transforms

        val = ''
        if not training:
            val = '_v'

        # File paths
        self.hr_dataset = h5py.File(f'{self.base}/hr{val}.h5', 'r')['/data']
        self.out_dataset = h5py.File(f'{self.base}/out{val}.h5', 'r')['/data']
        self.tif_dataset = h5py.File(f'{self.base}/tif{val}.h5', 'r')['/data']

    def __getitem__(self, index):
        x, tif, y = self.out_dataset[index], self.tif_dataset[index], self.hr_dataset[index]
        # Applica trasformazioni se definite
        if self.transforms:
            x = self.transforms(x)
            tif = self.transforms(tif)
            y = self.transforms(y)
        return x, tif, y

    def __len__(self):
        return self.hr_dataset.shape[0]


class H5Dataset(Dataset):
    def __init__(self, base_path='', transforms=None):
        super(H5Dataset, self).__init__()
        self.base = base_path
        self.transforms = transforms
        self.data_key = '/data'
        self.target_key = '/hrdata'
        self.tif_key = '/datatif'
        self.file_list = [os.path.join(self.base, f) for f in os.listdir(self.base) if f.endswith('.h5')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        with h5py.File(file_path, 'r') as f:
            data = np.array(f[self.data_key], dtype=np.float32)   # Hyperspettrale
            target = np.array(f[self.target_key], dtype=np.float32)  # Ground truth
            tif = np.array(f[self.tif_key], dtype=np.float32)  # Immagine TIFF RGB

        if data.shape[0] == 1:
            data = data.squeeze(0)
        if target.shape[0] == 1:
            target = target.squeeze(0)
        if tif.shape[0] == 1:
            tif = tif.squeeze(0)

        if self.transforms:
            data = self.transforms(data)
            target = self.transforms(target)
            tif = self.transforms(tif)
        return data, tif, target