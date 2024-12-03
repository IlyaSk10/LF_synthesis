from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt

from net import *
from dataset import *
from early_stop import *

import pickle

save_model_name = f'checkpoint_MSE.pth'
val_ind_name_file = 'val_ind.pkl'

data = MicroseismDataset(path_to_full_batch='C:/Users/admin/Downloads/batch_obj.hdf5',
                         path_to_LF_batch='C:/Users/admin/Downloads/batch_Lf.hdf5')

train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=4, shuffle=True)

model = Unet()

total_params = sum(p.numel() for p in model.parameters())
print('model params', total_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

early_stopping = EarlyStopping(patience=5, min_delta=0)

criterion = nn.MSELoss()
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
    pickle.dump(val_data.indices, file)

pass
