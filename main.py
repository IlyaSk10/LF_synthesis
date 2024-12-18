from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from net_using_dist import *
from dataset import *
from early_stop import *
from funcs import *

import pickle

save_model_name = f'checkpoint_MSE_dist.pth'
val_ind_name_file = 'val_ind.pkl'
train_ind_name_file = 'train_ind.pkl'
channels = ['Z']
train_fraction = 0.8
batch_size = 10
path_to_full_batch = './data/batch_obj.hdf5'
path_to_LF_batch = './data/batch_LF.hdf5'
num_sensors = 60
random_choice = True
epochs = 150
use_selection_sens = True

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

if use_selection_sens:

    with open(train_ind_name_file, 'rb') as f:
        train_sens_names = pickle.load(f)

    with open(val_ind_name_file, 'rb') as f:
        val_sens_names = pickle.load(f)
else:

    if random_choice:
        sensors_names = list(map(lambda x: int(x), sensors_names))
        selected_sensors_names = np.random.choice(range(min(sensors_names), max(sensors_names) + 1), size=num_sensors,
                                                  replace=False)
        selected_sensors_names = list(map(lambda x: str(x), selected_sensors_names))
        sensors_names = list(map(lambda x: str(x), sensors_names))
    else:
        selected_sensors_names = sensors_names[:num_sensors]

    train_sens_names = selected_sensors_names[:int(train_fraction * len(selected_sensors_names))]
    val_sens_names = selected_sensors_names[int(train_fraction * len(selected_sensors_names)):]

train_data = MicroseismDataset(path_to_full_batch=path_to_full_batch,
                               path_to_LF_batch=path_to_LF_batch, channels=channels, sens_names=train_sens_names,
                               distance=distance, sensors_names=sensors_names)

val_data = MicroseismDataset(path_to_full_batch=path_to_full_batch,
                             path_to_LF_batch=path_to_LF_batch, channels=channels, sens_names=val_sens_names,
                             distance=distance, sensors_names=sensors_names)

print(len(train_data), len(val_data), channels)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

model = Unet()

total_params = sum(p.numel() for p in model.parameters())
print('model params', total_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

early_stopping = EarlyStopping(patience=20, min_delta=0)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # , weight_decay=0.005)

train_loss = []
val_loss = []
for epoch in range(epochs):
    acc_train_loss = []
    acc_val_loss = []

    model.train()
    for i, (add_input, input, output) in enumerate(train_dataloader):
        # Forward pass
        mean, std, input = Z_score(input.float().to(device), mean_std=True)
        add_input = add_input.float().to(device)
        # output = output.float().to(device)
        output = Z_score(output.float().to(device))
        outputs = model(add_input, input)
        # outputs = model(input)
        # outputs = Z_score(model(input), inverse=True, mean=mean, std=std)
        loss = criterion(outputs, output)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train_loss.append(loss.item())

    train_loss.append(np.mean(acc_train_loss))

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

if not use_selection_sens:
    with open(train_ind_name_file, 'wb') as file:
        pickle.dump(train_sens_names, file)

    with open(val_ind_name_file, 'wb') as file:
        pickle.dump(val_sens_names, file)

plt.plot(train_loss, marker='*', color='k', label='train')
plt.plot(val_loss, marker='*', color='r', label='validation')
plt.grid()
plt.legend()
plt.show()

# k = 0
# plt.plot(input[k, 0, :], label='input')
# plt.plot(outputs[k, 0, :].detach().numpy(), label='predicted')
# plt.plot(output[k, 0, :].detach().numpy(), label='full wave')
# plt.grid()
# plt.legend()
#
# # check initial input output
# k = 1
# add_input, input, output = next(iter(val_dataloader))
# plt.plot(input[k, 0, :], label='input')
# plt.plot(output[k, 0, :].detach().numpy(), label='full wave')
# plt.grid()
# plt.legend()

pass
