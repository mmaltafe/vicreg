import time
import glob

import numpy as np
import pandas as pd
import torch
from datetime import datetime

from main_vicreg import *
from resnet import (
    resnet34,
    resnet50,
    resnet101,
    resnet50x2,
    resnet50x4,
    resnet50x5,
    resnet200x2,
)
from vicreg_timeseries import *

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

folder = "grid_search_sem_cuda_1"

reduction_list = [0.1, 0.2, 0.25, 0.5, 0.75]
ts_num_list = [80, 50, 40, 30, 20]
arch_list = [
    "resnet50",
    "resnet101",
    "resnet34",
]
mlp_list = [
    "8192-8192-8192",
    "4096-4096-4096",
    "2048-2048-2048",
    "1024-1024-1024",
]
batch_size_list = [2048, 512, 128]
sim_coeff_list = [50, 25, 10, 5, 1, 0]
std_coeff_list = [50, 25, 10, 5, 1, 0]
cov_coeff_list = [50, 25, 10, 5, 1, 0]


it = 0
list_files = glob.glob(f"{folder}/model_*.pth")
list_files = [int(f.split("_")[-1].split(".")[0]) for f in list_files]

N_PCs = 3

try:
    results_df = pd.read_csv(f"{folder}/results.csv")
except:
    results_df = pd.DataFrame()

for (reduction, ts_num) in zip(reduction_list, ts_num_list):
    X_train, y_train, X_test, y_test = read_data(0, 0.75)
    np.savetxt(f"{folder}/y_train.csv", y_train, delimiter=",")
    np.savetxt(f"{folder}/y_test.csv", y_test, delimiter=",")

    X_train_0 = sliding_window(X_train, reduction)
    X_train_0 = reshape_dataset(X_train_0)

    X_train_1 = sliding_window(X_train, reduction)
    X_train_1 = reshape_dataset(X_train_1)

    X_test_0 = sliding_window(X_test, reduction)
    X_test_0 = reshape_dataset(X_test_0)

    del X_train, X_test
    for arch in arch_list:
        for mlp in mlp_list:
            for batch_size in batch_size_list:
                for sim_coeff in sim_coeff_list:
                    for std_coeff in std_coeff_list:
                        for cov_coeff in cov_coeff_list:
                            args = {
                                "arch": arch,
                                "mlp": mlp,
                                "epochs": 500,
                                "batch_size": batch_size,
                                "base_lr": 0.2,
                                "wd": 1e-6,
                                "sim_coeff": sim_coeff,
                                "std_coeff": std_coeff,
                                "cov_coeff": cov_coeff,
                                "device": "cpu",
                            }

                            print("---> ", it)
                            print(datetime.now())
                            print(json.dumps(args))
                            if it not in list_files:
                                tik = datetime.now()
                                args = pd.Series(args)

                                model = VICReg(args)

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

                                unique_id = [
                                    unique_id[ts_num * i : ts_num * (i + 1)]
                                    for i in range(int(np.ceil(L // ts_num)) + 1)
                                ]

                                start_time = time.time()
                                flag = False
                                stop = False
                                loss_list = []
                                for epoch in range(start_epoch, args.epochs):
                                    for step, index in enumerate(
                                        unique_id, start=epoch * len(unique_id)
                                    ):
                                        x = X_train_0[index, :, :, :]
                                        y = X_train_1[index, :, :, :]

                                        lr = adjust_learning_rate(args, optimizer, unique_id, step)

                                        optimizer.zero_grad()
                                        loss = model.forward(x, y)
                                        loss.backward()
                                        optimizer.step()

                                        current_time = time.time()
                                        stats = dict(
                                            epoch=epoch,
                                            step=step,
                                            loss=loss.item(),
                                            time=int(current_time - start_time),
                                            lr=lr,
                                        )
                                        print(json.dumps(stats))

                                    if loss.item() < 1e9:
                                        loss_list.append(loss.item())
                                        std = np.std(loss_list[-10:])
                                        mean = np.mean(loss_list[-10:])
                                        if std / mean < 1e-3 and len(loss_list) > 10:
                                            stop = True
                                            break
                                    else:
                                        flag = True
                                        break

                                state = dict(
                                    epoch=epoch + 1,
                                    model=model.state_dict(),
                                    optimizer=optimizer.state_dict(),
                                )
                                torch.save(state, f"{folder}/model_{it}.pth")
                                args["epochs"] = epoch + 1

                                if not flag:
                                    for step, index in enumerate(unique_id):
                                        x = X_train_0[index, :, :, :]

                                        xx = model.backbone(x).detach().numpy()

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
                                        unique_id[ts_num * i : ts_num * (i + 1)]
                                        for i in range(int(np.ceil(L // ts_num)) + 1)
                                    ]

                                    for step, index in enumerate(unique_id):
                                        x = X_test_0[index, :, :, :]

                                        xx = model.backbone(x).detach().numpy()

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

                                    for clf, name in zip(classifiers, names):
                                        clf.fit(X_train_projected, y_train)
                                        y_pred = clf.predict(X_train_projected)
                                        train_acc = balanced_accuracy_score(y_train, y_pred)

                                        y_pred = clf.predict(X_test_projected)
                                        test_acc = balanced_accuracy_score(y_test, y_pred)

                                        cm = confusion_matrix(y_test, y_pred)
                                        args["clf"] = name
                                        args["train_acc"] = train_acc
                                        args["test_acc"] = test_acc
                                        args["cm"] = cm
                                        args["PCs"] = N_PCs
                                        args["time"] = tok - tik
                                        args["equal_columns"] = equal_columns
                                        args["final_loss"] = mean

                                        results_df = results_df.append(args, ignore_index=True)
                                        results_df.to_csv(f"{folder}/results.csv", index=False)
                                        print(
                                            f"{it},{reduction},{ts_num},{args.epochs},{args.batch_size},{args.arch},{args.mlp},{args.base_lr},{args.wd},{args.sim_coeff},{args.std_coeff},{args.cov_coeff},{name},{train_acc},{test_acc}",
                                            file=open(f"{folder}/Log.csv", "a+"),
                                        )
                                    del X_train_projected, X_test_projected
                                else:
                                    print(
                                        f"{it},{reduction},{ts_num},{args.epochs},{args.batch_size},{args.arch},{args.mlp},{args.base_lr},{args.wd},{args.sim_coeff},{args.std_coeff},{args.cov_coeff},0,0,0",
                                        file=open(f"{folder}/Log.csv", "a+"),
                                    )

                            it += 1

