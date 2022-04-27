import time
import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import balanced_accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from main_vicreg import *
from resnet import resnet50
from vicreg_timeseries import *


for (reduction, ts_num) in zip([0.1, 0.5, 0.75], [30, 10, 5]):

    X_train, y_train, X_test, y_test = read_data(0)

    X_train_0 = sliding_window(X_train, reduction)

    X_train_0 = reshape_dataset(X_train_0)

    X_train_1 = sliding_window(X_train, reduction)

    X_train_1 = reshape_dataset(X_train_1)

    del X_test, y_test

    args = {
        "data-dir": "Input/",
        "exp_dir": "Results/",
        "log-freq-time": 60,
        "arch": "resnet50",
        "mlp": "1024-1024-1024",  # "8192-8192-8192",
        "epochs": 100,
        "batch_size": 2048,
        "base_lr": 0.5,
        "wd": 1e-6,
        "sim_coeff": 25.0,
        "std_coeff": 25.0,
        "cov_coeff": 1.0,
        "num-workers": 1,
        "device": "cuda",
        "world-size": 1,
        "dist-url": "env://",
    }
    args = pd.Series(args)

    gpu = torch.device(args.device)
    model = VICReg(args)  # .cuda(gpu)

    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    stats_file = open(args.exp_dir + "stats.txt", "a", buffering=1)

    start_epoch = 0

    unique_id = np.array([i for i in range(len(X_train_0))])
    np.random.shuffle(unique_id)
    (L,) = unique_id.shape

    unique_id = [
        unique_id[ts_num * i : ts_num * (i + 1)]
        for i in range(int(np.ceil(L // ts_num)) + 1)
    ]

    start_time = last_logging = time.time()
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        for step, index in enumerate(unique_id, start=epoch * len(unique_id)):
            x = X_train_0[index, :, :, :]  # .cuda(gpu, non_blocking=True)
            y = X_train_1[index, :, :, :]  # .cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, unique_id, step)

            optimizer.zero_grad()
            loss = model.forward(x, y)
            # scaler.scale(loss.cuda()).backward()
            # scaler.step(optimizer)
            # scaler.update()
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
            print(json.dumps(stats), file=stats_file)
            last_logging = current_time
        state = dict(
            epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
        )
        torch.save(state, args.exp_dir + "model.pth")

    stats_file.close()

    for step, index in enumerate(unique_id):
        x = X_train_0[index, :, :, :].cuda(gpu, non_blocking=True)
        y = X_train_1[index, :, :, :].cuda(gpu, non_blocking=True)

        target = y_train[index]

        xx = model.backbone(x).cpu().detach().numpy()
        yy = model.backbone(y).cpu().detach().numpy()

        if step == 0:
            data = np.hstack((xx, yy, target.reshape(-1, 1)))
        else:
            temp = np.hstack((xx, yy, target.reshape(-1, 1)))
            data = np.vstack((data, temp))

    X = data[:, :-1]
    y = data[:, -1]
    del data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=12
    )

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
    # for N_PCs in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100]:
    if True:
        # pca = PCA(n_components=N_PCs)
        # pca.fit(X_train)

        # X_train_projected = pca.transform(X_train)
        # X_test_projected = pca.transform(X_test)
        N_PCs = 2048
        X_train_projected = X_train
        X_test_projected = X_test
        for name, clf in zip(names, classifiers):
            clf.fit(X_train_projected, y_train)

            y_pred = clf.predict(X_train_projected)
            acc = balanced_accuracy_score(y_train, y_pred)
            print("{:10d} - {:.2f} - {:30s} - {:.2f}".format(N_PCs, 1, name, acc * 100))

            y_pred = clf.predict(X_test_projected)

            acc = balanced_accuracy_score(y_test, y_pred)
            print("{:10d} - {:.2f} - {:30s} - {:.2f}".format(N_PCs, 1, name, acc * 100))
            print(
                f"{reduction}, \
                {ts_num}, \
                {args.epochs}, \
                {args.batch_size}, \
                {args.arch}, \
                {args.mlp}, \
                {args.base_lr}, \
                {args.wd}, \
                {args.sim_coeff}, \
                {args.std_coeff}, \
                {args.cov_coeff}, \
                {N_PCs}, \
                {name}, \
                {acc}",
                file=open("Results/results.csv", "a+"),
            )

