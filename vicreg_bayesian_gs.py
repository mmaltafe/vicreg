import glob
import time
import math
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA

import resnet

from ax.service.managed_loop import optimize


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
            range(train_unique_index[ii] * ts_len, (train_unique_index[ii] + 1) * ts_len,)
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


class sliding_window:
    def __init__(self, size=0.5):
        self.size = size

    def fit(self, X, y):
        pass

    def transform(self, X):
        def reshape_dataset(data):
            L, W = data.shape
            ts_len = int(data[:, 1].max())
            ts_num = int(L / ts_len)
            data = data[:, 2:]
            data = data.reshape(ts_num, 1, ts_len, W - 2)

            return torch.from_numpy(data).float()

        L, W = X.shape
        ts_len = int(X[:, 1].max())
        ts_num = int(L / ts_len)
        if type(self.size) == float:
            new_ts_len = int(ts_len * self.size)
            time_id = np.arange(1, new_ts_len + 1)
            new_full_data_0 = np.zeros((ts_num * new_ts_len, W))
            new_full_data_1 = np.zeros((ts_num * new_ts_len, W))
            for ii in range(ts_num):
                data = X[ii * ts_len : (ii + 1) * ts_len, :].copy()

                init_id_0 = np.random.randint(0, int(ts_len - (ts_len * self.size)))
                data_0 = data[init_id_0 : init_id_0 + new_ts_len, :]
                data_0[:, 1] = time_id
                new_full_data_0[ii * new_ts_len : (ii + 1) * new_ts_len, :] = data_0.copy()

                init_id_1 = np.random.randint(0, int(ts_len - (ts_len * self.size)))
                data_1 = data[init_id_1 : init_id_1 + new_ts_len, :]
                data_1[:, 1] = time_id
                new_full_data_1[ii * new_ts_len : (ii + 1) * new_ts_len, :] = data_1.copy()

            return reshape_dataset(new_full_data_0), reshape_dataset(new_full_data_1)

        if type(self.size) == int:
            new_full_data_0 = np.zeros((ts_num * self.size, W))
            new_full_data_1 = np.zeros((ts_num * self.size, W))
            time_id = np.arange(1, self.size + 1)
            for ii in range(ts_num):
                data = X[ii * ts_len : (ii + 1) * ts_len, :].copy()

                init_id_0 = np.random.randint(0, int(ts_len - self.size))
                data_0 = data[init_id_0 : init_id_0 + self.size, :]
                data_0[:, 1] = time_id
                new_full_data_0[ii * self.size : (ii + 1) * self.size, :] = data.copy()

                init_id_1 = np.random.randint(0, int(ts_len - self.size))
                data_1 = data[init_id_1 : init_id_1 + self.size, :]
                data_1[:, 1] = time_id
                new_full_data_1[ii * self.size : (ii + 1) * self.size, :] = data.copy()

            return reshape_dataset(new_full_data_0), reshape_dataset(new_full_data_1)


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def adjust_learning_rate(args, optimizer, un_steps, step):
    max_steps = args.epochs * un_steps
    warmup_steps = 10 * un_steps
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * float(step / warmup_steps)
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](zero_init_residual=True)
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def exclude_bias_and_norm(p):
    return p.ndim == 1


def train_evaluate(params):
    input_id = 0
    train_size = 0.75
    reduction = 0.1
    ts_num = 25
    N_PCs = 3
    FOLDER = "bayesian_gs"

    try:
        results_df = pd.read_csv(f"{FOLDER}/results.csv")
    except:
        results_df = pd.DataFrame()

    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma="scale"),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1, max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    # Read Data
    X_train, y_train, X_test, y_test = read_data(input_id, train_size)

    # Sliding Window
    crop = sliding_window(reduction)
    X_train_0, X_train_1 = crop.transform(X_train)
    X_test_0, _ = crop.transform(X_test)

    del X_train, X_test, _

    print(datetime.now())
    print(json.dumps(params))

    tik = datetime.now()

    args = pd.Series(params)
    args["device"] = "cuda"
    args["wd"] = 1e-6
    args["base_lr"] = 0.2
    args["epochs"] = 500
    args["mlp"] = f"{args.mlp_n}-{args.mlp_n}-{args.mlp_n}"

    gpu = torch.device(args.device)

    model = VICReg(args).cuda(gpu)
    optimizer = LARS(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    start_epoch = 0

    unique_id = np.array([i for i in range(len(X_train_0))])
    np.random.shuffle(unique_id)
    (L,) = unique_id.shape
    un_steps = int(np.ceil(L // ts_num))

    unique_id = [unique_id[ts_num * i : ts_num * (i + 1)] for i in range(un_steps + 1)]

    flag = True
    loss_list = []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        for step, index in enumerate(unique_id, start=epoch * len(unique_id)):
            x = X_train_0[index, :, :, :].cuda(gpu, non_blocking=True)
            y = X_train_1[index, :, :, :].cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, un_steps, step)

            optimizer.zero_grad()
            loss = model.forward(x, y)
            scaler.scale(loss.cuda()).backward()
            scaler.step(optimizer)
            scaler.update()

            stats = dict(epoch=epoch, step=step, loss=loss.item(), lr=lr,)
            print(json.dumps(stats))

        if loss.item() < 1e9:
            loss_list.append(loss.item())
            std = np.std(loss_list[-10:])
            mean = np.mean(loss_list[-10:])
            if mean < 1e-12 or (std / mean < 1e-3 and len(loss_list) > 10):
                break
        else:
            flag = False
            break

    args["epochs"] = epoch + 1

    if flag:
        for step, index in enumerate(unique_id):
            x = X_train_0[index, :, :, :].cuda(gpu, non_blocking=True)

            xx = model.backbone(x).cpu().detach().numpy()

            if step == 0:
                train_data = xx.copy()
            else:
                train_data = np.vstack((train_data, xx))

        col = np.all(train_data == train_data[0, :], axis=0)
        equal_columns = np.sum(col)

        unique_id = np.array([i for i in range(len(X_test_0))])
        np.random.shuffle(unique_id)
        (L,) = unique_id.shape

        unique_id = [
            unique_id[ts_num * i : ts_num * (i + 1)] for i in range(int(np.ceil(L // ts_num)) + 1)
        ]

        for step, index in enumerate(unique_id):
            x = X_test_0[index, :, :, :].cuda(gpu, non_blocking=True)

            xx = model.backbone(x).cpu().detach().numpy()

            if step == 0:
                test_data = xx.copy()
            else:
                test_data = np.vstack((test_data, xx))

        pca = PCA(n_components=N_PCs)
        pca.fit(train_data)

        X_train_projected = pca.transform(train_data)
        X_test_projected = pca.transform(test_data)
        del train_data, test_data

        tok = datetime.now()

        final_test_acc = 0

        for name, clf in zip(names, classifiers):
            clf.fit(X_train_projected, y_train)
            y_pred = clf.predict(X_train_projected)
            train_acc = balanced_accuracy_score(y_train, y_pred)

            y_pred = clf.predict(X_test_projected)
            test_acc = balanced_accuracy_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            args["clf"] = name
            args["train_acc"] = train_acc
            args["test_acc"] = test_acc
            args["cm"] = list(cm)
            args["PCs"] = N_PCs
            args["time"] = tok - tik
            args["equal_columns"] = equal_columns
            args["final_loss"] = mean

            results_df = results_df.append(args, ignore_index=True)
            results_df.to_csv(f"{FOLDER}/results.csv", index=False)
            print(
                f"{reduction},{ts_num},{args.epochs},{args.batch_size},{args.arch},{args.mlp},{args.base_lr},{args.wd},{args.sim_coeff},{args.std_coeff},{args.cov_coeff},{name},{train_acc},{test_acc}",
                file=open(f"{FOLDER}/Log.csv", "a+"),
            )
            if test_acc > final_test_acc:
                final_test_acc = test_acc
            print(test_acc * 100)
        del X_train_projected, X_test_projected, model
    else:
        final_test_acc = 0
        print(
            f"{reduction},{ts_num},{args.epochs},{args.batch_size},{args.arch},{args.mlp},{args.base_lr},{args.wd},{args.sim_coeff},{args.std_coeff},{args.cov_coeff},0,0,0",
            file=open(f"{FOLDER}/Log.csv", "a+"),
        )

    return final_test_acc


if __name__ == "__main__":
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "arch",
                "type": "choice",
                "values": ["resnet34", "resnet50"],
                "value_type": "str",
            },
            {"name": "mlp_n", "type": "range", "bounds": [128, 4096], "value_type": "int",},
            {"name": "batch_size", "type": "range", "bounds": [2, 4096], "value_type": "int",},
            {"name": "sim_coeff", "type": "range", "bounds": [0, 50], "value_type": "int",},
            {"name": "std_coeff", "type": "range", "bounds": [0, 50], "value_type": "int",},
            {"name": "cov_coeff", "type": "range", "bounds": [0, 50], "value_type": "int",},
        ],
        evaluation_function=train_evaluate,
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)

