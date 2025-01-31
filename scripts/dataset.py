from torch.utils.data import Dataset
import h5py


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
        #self.out_dataset = h5py.File(f'{self.base}/out{val}.h5', 'r')['/data']
        self.replaced_dataset = h5py.File(f'{self.base}/replaced{val}.h5', 'r')['/data']
        #self.hr_g_dataset = h5py.File(f'{self.base}/hr_g{val}.h5', 'r')['/data']

    def __getitem__(self, index):
        x, y = self.replaced_dataset[index], self.hr_dataset[index]
        # Applica trasformazioni se definite
        if self.transforms:
            x = self.transforms(x)
        if self.transforms:
            y = self.transforms(y)
        return x, y

    def __len__(self):
        return self.hr_dataset.shape[0]

