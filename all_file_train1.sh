#!/bin/bash

datas=("SST-5" "aclImdb" "SST-2" "MR" )
model_mode="Star_Label_ANN_w_linear"
tf_mode = "ELECTRA"

echo datas
for data in ${datas[@]}; do
  python3 -u train.py --result_dir ${model_mode}_${data}_${tf_mode} --model_mode ${model_mode} --gpu 1 --dataset ${data} --transformer_mode ${tf_mode}
done