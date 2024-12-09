from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py

from net import *
from dataset import *
from early_stop import *

import pickle

save_model_name = f'checkpoint_MSE_Z.pth'
val_ind_name_file = 'val_ind.pkl'
channels = ['Z']
train_fraction = 0.8

path_to_full_batch = './data/batch_obj.hdf5'
path_to_LF_batch = './data/batch_LF.hdf5'

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


def dist(source_points, sensors_coords):
    distance = []
    for l in range(len(sensors_coords)):
        distance.append(np.sqrt((sensors_coords[l][0] - source_points[0][0]) ** 2 + (
                sensors_coords[l][1] - source_points[0][1]) ** 2 + (
                                        sensors_coords[l][2] - source_points[0][2]) ** 2))
    return distance


distance = dist(source_points, sensors_coords)

# get all sens names
full_batch = h5py.File(path_to_full_batch)
sens_names = list(set([s.split('_')[0] for s in full_batch['Channels'].keys()]))
train_sens_names = sens_names[:int(train_fraction * len(sens_names))]
val_sens_names = sens_names[int(train_fraction * len(sens_names)):]

train_data = MicroseismDataset(path_to_full_batch=path_to_full_batch,
                               path_to_LF_batch=path_to_LF_batch, channels=channels, sens_names=train_sens_names,
                               distance=distance, sensors_names=sensors_names)

len(train_data)
val_data = MicroseismDataset(path_to_full_batch=path_to_full_batch,
                             path_to_LF_batch=path_to_LF_batch, channels=channels, sens_names=val_sens_names)

print(len(train_data), len(val_data), channels)

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=4, shuffle=True)

model = Unet()

total_params = sum(p.numel() for p in model.parameters())
print('model params', total_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

early_stopping = EarlyStopping(patience=5, min_delta=0)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss = []
val_loss = []
for epoch in range(150):
    acc_train_loss = []
    acc_val_loss = []

    model.train()
    for i, (input, output) in enumerate(train_dataloader):
        # Forward pass
        input = input.float().to(device)
        output = output.float().to(device)
        outputs = model(input)
        loss = criterion(outputs, output)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train_loss.append(loss.item())

    train_loss.append(np.mean(acc_train_loss))

    model.eval()
    with torch.no_grad():
        for i, (input, output) in enumerate(val_dataloader):
            input = input.float().to(device)
            output = output.float().to(device)
            outputs = model(input)
            loss = criterion(outputs, output)

            acc_val_loss.append(loss.item())

    val_loss.append(np.mean(acc_val_loss))

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': np.mean(acc_val_loss)
    }

    early_stopping(np.mean(acc_val_loss), checkpoint, save_model_name)
    if early_stopping.early_stop:
        print(f"Ранний останов на эпохе {epoch}")
        break

    print(f'epoch {epoch}, train_loss {np.mean(acc_train_loss)}, val_loss {np.mean(acc_val_loss)}')

with open(val_ind_name_file, 'wb') as file:
    pickle.dump(val_sens_names, file)

plt.plot(train_loss, marker='*', color='k', label='train')
plt.plot(val_loss, marker='*', color='r', label='validation')
plt.grid()
plt.legend()
plt.show()

pass
