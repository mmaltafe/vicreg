import time
import os
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

folder = "Extracted_Input_1"

X_train, y_train, X_test, y_test = read_data(0, 0.5)

reduction_list = [0.1, 0.2, 0.25, 0.5, 0.75]
ts_num_list = [60, 50, 40, 30, 20]
arch_list = [
    "resnet34",
    "resnet50",
    "resnet101",
]
mlp_list = ["1024-1024-1024", "2048-2048-2048", "4096-4096-4096", "8192-8192-8192"]


np.savetxt(f"{folder}/Target.csv", y_train, delimiter=",")

it = 0
list_files = glob.glob(f"{folder}/Extracted_Input_*")
list_files = [int(f.split("_")[-1].split(".")[0]) for f in list_files]


del X_test, y_test
for (reduction, ts_num) in zip(reduction_list, ts_num_list):
    X_train_0 = sliding_window(X_train, reduction)
    X_train_0 = reshape_dataset(X_train_0)

    X_train_1 = sliding_window(X_train, reduction)
    X_train_1 = reshape_dataset(X_train_1)
    for arch in arch_list:
        for mlp in mlp_list:
            args = {
                "data-dir": "Input/",
                "exp_dir": "Results/",
                "log-freq-time": 60,
                "arch": arch,
                "mlp": mlp,
                "epochs": 500,
                "batch_size": 2048,
                "base_lr": 0.2,
                "wd": 1e-6,
                "sim_coeff": 25.0,
                "std_coeff": 25.0,
                "cov_coeff": 1.0,
                "num-workers": 1,
                "device": "cuda",
                "world-size": 1,
                "dist-url": "env://",
            }
            print("---> ", it)
            print(datetime.now())
            if it not in list_files:
                args = pd.Series(args)

                model = VICReg(args)

                optimizer = LARS(
                    model.parameters(),
                    lr=args.base_lr,
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

                start_time = time.time()
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
                        print(json.dumps(stats), file=stats_file)

                stats_file.close()

                for step, index in enumerate(unique_id):
                    x = X_train_0[index, :, :, :]
                    y = X_train_1[index, :, :, :]

                    target = y_train[index]

                    xx = model.backbone(x).detach().numpy()
                    yy = model.backbone(y).detach().numpy()

                    if step == 0:
                        data = np.hstack((xx, yy, target.reshape(-1, 1)))
                    else:
                        temp = np.hstack((xx, yy, target.reshape(-1, 1)))
                        data = np.vstack((data, temp))

                np.savetxt(
                    "{}/Extracted_Input_{}.csv".format(folder, it), data, delimiter=",",
                )
                del data, temp

                print(
                    f"{it},{reduction},{ts_num},{args.epochs},{args.batch_size},{args.arch},{args.mlp},{args.base_lr},{args.wd},{args.sim_coeff},{args.std_coeff},{args.cov_coeff}",
                    file=open(f"{folder}/Log.csv", "a+"),
                )
            it += 1
