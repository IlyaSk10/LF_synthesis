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
model_name = "checkpoint_MSE.pth"
val_ind_name_file = 'val_ind.pkl'

if delete:
    try:
        shutil.rmtree(f'{folder_name}')
    except FileNotFoundError:
        print(f'dir /{folder_name}/ already deleted')

model = Unet()

model.load_state_dict(torch.load(model_name, weights_only=True))

data = MicroseismDataset(path_to_full_batch='C:/Users/admin/Downloads/batch_obj.hdf5',
                         path_to_LF_batch='C:/Users/admin/Downloads/batch_Lf.hdf5')

train_data, val_data = torch.utils.data.random_split(data, [0.0, 1.0])

with open(val_ind_name_file, 'rb') as f:
    val_ind = pickle.load(f)

val_data.indices = val_ind

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
        input = input.float().to(device)
        output = output.float().to(device)
        outputs = model(input)

        print(f'batch number {i}')

        for j in range(input.shape[0]):
            plt.figure(figsize=(15, 7))
            plt.plot(outputs[j, 0, :], label='predicted')
            plt.plot(output[j, 0, :], label='full wave (label)')
            plt.plot(input[j, 0, :], label='input', linestyle=':')
            plt.legend()
            plt.savefig(f'./{folder_name}/{k}.png')
            plt.close()
            k += 1

pass
