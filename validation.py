from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from net import *
from dataset import *

import pickle

model = Unet()

model.load_state_dict(torch.load("checkpoint.pth", weights_only=True))
model.eval()

data = MicroseismDataset(path_to_full_batch='batch_obj.hdf5', path_to_LF_batch='batch_Lf.hdf5')

train_data, val_data = torch.utils.data.random_split(data, [0.0, 1.0])

with open('val_ind.pkl', 'rb') as f:
    val_ind = pickle.load(f)

val_data.indices = val_ind

val_dataloader = DataLoader(val_data, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
with torch.no_grad():
    for i, (input, output) in enumerate(val_dataloader):
        input = input.float().to(device)
        output = output.float().to(device)
        outputs = model(input)

pass
