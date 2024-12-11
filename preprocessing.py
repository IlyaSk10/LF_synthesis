import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import h5py
import time

cutoff = 20  # Частота среза, Гц
order = 5  # Порядок фильтра
fs = 500
batch_filename = './data/batch_LF.hdf5'

ff = h5py.File('./data/batch_obj.hdf5')

initial_sens = list(ff['Channels'].keys())[2]
signal = ff['Channels'][initial_sens]['data'][0, 2, :]

num_source_points = ff['Channels'][initial_sens]['data'].shape[0]
num_tensor_comp = 6
len_response = ff['Channels'][initial_sens]['data'].shape[2]

b, a = butter(order, cutoff / (0.5 * fs), btype='low')
filtered_signal = filtfilt(b, a, signal)

plt.plot(signal, label='original')
plt.plot(filtered_signal, label='filtered_signal')
plt.legend()
plt.grid()
plt.show()

st = time.time()

with h5py.File(batch_filename, 'w') as f:
    g = f.create_group('Channels')
    for s in ff['Channels'].keys():
        g1 = g.create_group(s)
        initial_data = np.zeros((num_source_points, num_tensor_comp, len_response))
        for p in range(num_source_points):
            for j in range(num_tensor_comp):
                data = ff['Channels'][s]['data'][p, j, :]
                initial_data[p, j, :] = filtfilt(b, a, data)
        g1.create_dataset('data', data=initial_data)

print(time.time() - st)
