from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import shutil

from net import *
from dataset import *

import pickle

delete = True
folder_name = 'results_MSE'
model_name = "checkpoint_MSE_Z.pth"
val_ind_name_file = 'val_ind.pkl'
channels = ['Z']

if delete:
    try:
        shutil.rmtree(f'{folder_name}')
    except FileNotFoundError:
        print(f'dir /{folder_name}/ already deleted')

with open(val_ind_name_file, 'rb') as f:
    val_ind = pickle.load(f)


def Z_score(data, mean_std=False, inverse=False, mean=None, std=None):
    batch_mean = data.mean(dim=2, keepdim=True)
    batch_std = data.std(dim=2, keepdim=True)

    data = (data - batch_mean) / batch_std

    if mean_std:
        mean = batch_mean[:, 0, 0].tolist()
        std = batch_std[:, 0, 0].tolist()
        return mean, std, data

    if inverse:
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] + mean[i]) * std[i]
        return data

    return data


model = Unet()

model.load_state_dict(torch.load(model_name, weights_only=True))

val_data = MicroseismDataset(path_to_full_batch='./batch_obj.hdf5',
                             path_to_LF_batch='./batch_Lf.hdf5', channels=channels, sens_names=val_ind)

val_dataloader = DataLoader(val_data, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    os.mkdir(f'{folder_name}')
except FileExistsError:
    print('dir already exists')

k = 0
model.eval()
with torch.no_grad():
    for i, (input, output) in enumerate(val_dataloader):

        mean, std, input = Z_score(input.float().to(device), mean_std=True)
        output = output.float().to(device)
        # output = Z_score(output.float().to(device))
        outputs = Z_score(model(input), inverse=True, mean=mean, std=std)
        # outputs = model(input)

        print(f'batch number {i}')

        for j in range(input.shape[0]):
            plt.figure(figsize=(15, 7))
            plt.plot(outputs[j, 0, :], label='predicted')
            plt.plot(output[j, 0, :], label='full wave (label)')
            # plt.plot(input[j, 0, :], label='input', linestyle=':')
            plt.legend()
            plt.savefig(f'./{folder_name}/{k}.png')
            plt.close()
            k += 1

pass
