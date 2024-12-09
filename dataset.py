from torch.utils.data import Dataset
import h5py
import numpy as np


class MicroseismDataset(Dataset):
    def __init__(self, path_to_full_batch, path_to_LF_batch, channels, sens_names, distance, sensors_names):
        self.full_batch = h5py.File(path_to_full_batch)
        self.LF_batch = h5py.File(path_to_LF_batch)
        self.channels = channels
        self.sens_names = sens_names
        self.distance = distance
        self.sensors_names = sensors_names

    def __len__(self):
        return len(self.create_dataset())

    def create_dataset(self):
        self.sens = []
        for ch in self.channels:
            self.sens.extend([s + f'_{ch}' for s in self.sens_names])

        self.num_power_points = self.full_batch['Channels'][self.sens[0]]['data'].shape[0]

        self.batch_indx = []
        for sen in self.sens:
            for i in range(self.num_power_points):
                for j in range(6):
                    sen.split('_')[0]
                    self.batch_indx.append([sen, i, j])
        return self.batch_indx

    def __getitem__(self, idx):
        sen, pp, comp = self.create_dataset()[idx]
        input = np.expand_dims(self.LF_batch['Channels'][sen]['data'][pp, comp, :], axis=0)
        output = np.expand_dims(self.full_batch['Channels'][sen]['data'][pp, comp, :], axis=0)
        return input, output
