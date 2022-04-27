import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
import torch
from torch import nn
import resnet


def read_data(input_id=0, train_size=0.75):
    path = "Input/Input_{}.csv".format(input_id)

    full_data = pd.read_csv(path, header=None).values

    L, W = full_data.shape
    ts_len = int(full_data[:, 1].max())
    ts_num = int(L / ts_len)

    unique_id = np.arange(ts_num)
    target = full_data[::ts_len, -1]

    train_unique_index, test_unique_index, _, _ = train_test_split(
        unique_id, target, stratify=target, random_state=12, train_size=train_size
    )

    train_unique_index.sort()
    test_unique_index.sort()

    L_train = train_unique_index.shape[0]
    train_index = np.zeros(L_train * ts_len, dtype=np.int32)
    for ii in range(L_train):
        train_index[ii * ts_len : (ii + 1) * ts_len] = list(
            range(
                train_unique_index[ii] * ts_len, (train_unique_index[ii] + 1) * ts_len,
            )
        )

    L_test = test_unique_index.shape[0]
    test_index = np.zeros(L_test * ts_len, dtype=np.int32)
    for ii in range(L_test):
        test_index[ii * ts_len : (ii + 1) * ts_len] = list(
            range(test_unique_index[ii] * ts_len, (test_unique_index[ii] + 1) * ts_len,)
        )

    X_train = full_data[train_index, :-1]
    y_train = full_data[train_index, -1]
    y_train = y_train[::ts_len]

    X_test = full_data[test_index, :-1]
    y_test = full_data[test_index, -1]
    y_test = y_test[::ts_len]

    scaler = StandardScaler()
    scaler.fit(X_train[:, 2:])
    scaled_data = scaler.transform(X_train[:, 2:])
    X_train[:, 2:] = scaled_data.copy()
    scaled_data = scaler.transform(X_test[:, 2:])
    X_test[:, 2:] = scaled_data.copy()

    return X_train, y_train, X_test, y_test


def sliding_window(full_data, size=0.5):
    L, W = full_data.shape
    ts_len = int(full_data[:, 1].max())
    ts_num = int(L / ts_len)
    if type(size) == float:
        new_ts_len = int(ts_len * size)
        time_id = np.arange(1, new_ts_len + 1)
        new_full_data = np.zeros((ts_num * new_ts_len, W))
        for ii in range(ts_num):
            data = full_data[ii * ts_len : (ii + 1) * ts_len, :].copy()
            init_id = np.random.randint(0, int(ts_len - (ts_len * size)))
            data = data[init_id : init_id + new_ts_len, :]
            data[:, 1] = time_id
            new_full_data[ii * new_ts_len : (ii + 1) * new_ts_len, :] = data.copy()

        return new_full_data

    if type(size) == int:
        new_full_data = np.zeros((ts_num * size, W))
        time_id = np.arange(1, size + 1)
        for ii in range(ts_num):
            data = full_data[ii * ts_len : (ii + 1) * ts_len, :].copy()
            init_id = np.random.randint(0, int(ts_len - size))
            data = data[init_id : init_id + size, :]
            data[:, 1] = time_id
            new_full_data[ii * size : (ii + 1) * size, :] = data.copy()

        return new_full_data


def reshape_dataset(data):
    L, W = data.shape
    ts_len = int(data[:, 1].max())
    ts_num = int(L / ts_len)
    data = data[:, 2:]
    data = data.reshape(ts_num, 1, ts_len, W - 2)

    return torch.from_numpy(data).float()


def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


# TODO
class DualEncoder(Model):
    def __init__(
        self, encoder_0=None, encoder_1=None, decoder_0=None, decoder_1=None, **kwargs
    ):
        super(DualEncoder, self).__init__(**kwargs)

        mlp = "8192-8192-8192"
        if encoder_0 is None:
            self.encoder_0, embedding = resnet.resnet50(num_channels=1)
            self.decoder_0 = Projector(mlp, self.embedding)
            self.model_0 = nn.Sequential(self.encoder_0, self.decoder_0)
        else:
            self.encoder_0 = encoder_0
            self.decoder_0 = decoder_0

        if encoder_1 is None:
            self.encoder_1, embedding = resnet.resnet50(num_channels=1)
            self.decoder_1 = Projector(mlp, self.embedding)
            self.model_1 = nn.Sequential(self.encoder_1, self.decoder_1)
        else:
            self.encoder_1 = encoder_1
            self.decoder_1 = decoder_1
