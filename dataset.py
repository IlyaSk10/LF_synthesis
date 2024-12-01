from torch.utils.data import Dataset
import h5py
import numpy as np

class MicroseismDataset(Dataset):
    def __init__(self, path_to_full_batch, path_to_LF_batch):
        self.full_batch = h5py.File(path_to_full_batch)
        self.LF_batch = h5py.File(path_to_LF_batch)

    def __len__(self):
        init_sens = list(self.full_batch['Channels'].keys())[0]
        num_power_points = self.full_batch['Channels'][init_sens]['data'].shape[0]
        return len(self.full_batch['Channels'].keys()) * num_power_points * 6

    def create_dataset(self):
        self.batch_indx = []
        for sen in self.full_batch['Channels'].keys():
            for i in range(1):
                for j in range(6):
                    self.batch_indx.append([sen, i, j])
        return self.batch_indx

    def __getitem__(self, idx):
        sen, pp, comp = self.create_dataset()[idx]
        input = np.expand_dims(self.LF_batch['Channels'][sen]['data'][pp, comp, :], axis=0)
        output = np.expand_dims(self.full_batch['Channels'][sen]['data'][pp, comp, :], axis=0)
        return input, output