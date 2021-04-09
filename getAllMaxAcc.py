import pandas as pd
import os
import argparse
import numpy as np
import torch


args = argparse.ArgumentParser()

args.add_argument("--keyword", type=str, default=None, required=False)

args = args.parse_args()


result_dir_list = os.listdir("ckpt")

for result_dir in result_dir_list[::-1]:
    k = False
    if args.keyword is not None:
        for keyword in args.keyword.split(","):
            if not (keyword in result_dir):
                k = True

    if k:
        continue

    if "ckpt/" in result_dir:
        result_dir = result_dir[5:]

    setting_path = os.path.join("ckpt", result_dir, "checkpoint-best")
    setting = torch.load(os.path.join(setting_path, "training_args.bin"))

    if os.path.exists(os.path.join("ckpt", result_dir, "dev")):
        result_path = os.path.join("ckpt", result_dir, "dev")
    else:
        result_path = os.path.join("ckpt", result_dir, "test")
    epoch_list = os.listdir(result_path)

    acc_dict = dict()
    for i in epoch_list:
        with open(os.path.join(result_path,i),"r") as fp:
            acc_dict[int(i.split("-")[-1].split(".")[0])] = float(fp.readline().split()[-1])

    reversed_dict = { y:x for x,y in acc_dict.items()}
    acc_dict = sorted(acc_dict.items())


    acc_dict = list(map(list,sorted(acc_dict)))
    print("\n",result_dir)
    print("max = ",max(np.array(acc_dict)[:,-1]))
    print("max step = ",reversed_dict[max(np.array(acc_dict)[:,-1])])
    print("\n----------------")