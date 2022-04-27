import pandas as pd

reduction_list = [0.1, 0.2, 0.25, 0.5, 0.75]
ts_num_list = [30, 30, 30, 10, 5]
arch_list = [
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet50x2",
    "resnet50x4",
    "resnet50x5",
    "resnet200x2",
]
mlp_list = ["1024-1024-1024", "2048-2048-2048", "4096-4096-4096"]

it = 0
for (reduction, ts_num) in zip(reduction_list, ts_num_list):
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
            args = pd.Series(args)

            print(
                f"{it},{reduction},{ts_num},{args.epochs},{args.batch_size},{args.arch},{args.mlp},{args.base_lr},{args.wd},{args.sim_coeff},{args.std_coeff},{args.cov_coeff}",
                file=open("Log.csv", "a+"),
            )
            it += 1
