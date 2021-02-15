#!/bin/bash

datas=("SST-5" "aclImdb" "SST-2" "MR")
model_mode="Star_Label_AM"
tf_mode="ELECTRA"

echo datas
for data in ${datas[@]}; do
  python3 -u train.py --result_dir ${model_mode}_${data}_${tf_mode} --model_mode ${model_mode} --dataset ${data} --transformer_mode ${tf_mode} --gpu 0
done