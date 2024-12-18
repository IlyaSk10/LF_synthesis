from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import shutil

from net_using_dist import *
from dataset import *
from funcs import *

import pickle

delete = True
folder_name = 'results_MSE_dist'
model_name = "checkpoint_MSE_dist.pth"
val_ind_name_file = 'val_ind.pkl'
channels = ['Z']
batch_size = 10

if delete:
    try:
        shutil.rmtree(f'{folder_name}')
    except FileNotFoundError:
        print(f'dir /{folder_name}/ already deleted')

with open(val_ind_name_file, 'rb') as f:
    val_ind = pickle.load(f)

with open('./data/points_tdsh434.txt') as f:
    points = f.readlines()

source_points = []
for pp in points:
    res = pp.split(' ')
    source_points.append([int(res[3]), int(res[4]), int(res[6])])

with open('./data/sensors.txt') as f:
    sensors = f.readlines()

sensors_names = []
sensors_coords = []
for ss in sensors:
    res = ss.split(' ')
    sensors_names.append(res[0])
    sensors_coords.append([int(res[3]), int(res[4]), int(res[6])])

distance = dist(source_points, sensors_coords)

model = Unet()

model.load_state_dict(torch.load(model_name, weights_only=True))

val_data = MicroseismDataset(path_to_full_batch='./data/batch_obj.hdf5',
                             path_to_LF_batch='./data/batch_LF.hdf5', channels=channels, sens_names=val_ind,
                             distance=distance, sensors_names=sensors_names)

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    os.mkdir(f'{folder_name}')
except FileExistsError:
    print('dir already exists')

k = 0
sk = 0
s = 0
num_sens = val_ind[sk]
comp_tenz = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']

model.eval()
with torch.no_grad():
    for i, (add_input, input, output) in enumerate(val_dataloader):
        mean, std, input = Z_score(input.float().to(device), mean_std=True)
        add_input = add_input.float().to(device)
        # output = output.float().to(device)
        output = Z_score(output.float().to(device))
        outputs = model(add_input, input)
        # outputs = model(input)
        # outputs = Z_score(model(input), inverse=True, mean=mean, std=std)

        print(f'batch number {i}')

        for j in range(input.shape[0]):
            plt.figure(figsize=(15, 7))
            plt.plot(outputs[j, 0, :], label='predicted')
            plt.plot(output[j, 0, :], label='full wave (label)')
            plt.plot(input[j, 0, :], label='input', linestyle=':')
            if k % 6 == 0 and k != 0:
                sk += 1
                num_sens = val_ind[sk]
                s = 0
            plt.title(f'sensor name {num_sens}, {comp_tenz[s]}')
            plt.legend()
            plt.savefig(f'./{folder_name}/sensor_name_{num_sens}_{comp_tenz[s]}_{k}.png')
            plt.close()
            k += 1
            s += 1

pass
