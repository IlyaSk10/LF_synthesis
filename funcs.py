import numpy as np

def dist(source_points, sensors_coords):
    distance = []
    for l in range(len(sensors_coords)):
        distance.append(np.sqrt((sensors_coords[l][0] - source_points[0][0]) ** 2 + (
                sensors_coords[l][1] - source_points[0][1]) ** 2 + (
                                        sensors_coords[l][2] - source_points[0][2]) ** 2))
    return distance


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