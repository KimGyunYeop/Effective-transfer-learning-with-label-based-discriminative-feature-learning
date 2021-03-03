#!/bin/bash

datas=("aclImdb" "SST-2" "SST-5")
model_mode="Star_Label_AM"

echo datas
for data in ${datas[@]}; do
    python3 -u train.py --result_dir ${model_mode}_${data} --model_mode ${model_mode} --gpu 0 --dataset ${data}
done