#!/bin/bash

datas=("SST-5" "aclImdb" "SST-2" "MR")
model_mode="Star_Label_ANN"

echo datas
for data in ${datas[@]}; do
    python3 -u train.py --result_dir ${model_mode}_${data} --model_mode ${model_mode} --gpu 0 --dataset ${data}
done