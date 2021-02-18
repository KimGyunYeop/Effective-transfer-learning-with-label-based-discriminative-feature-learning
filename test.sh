#!/bin/bash


data="R8"
model_mode="Star_Label_ANN_w_linear"
tf_mode="ELECTRA"

python3 -u test.py --result_dir ${model_mode}_${data}_${tf_mode} --model_mode ${model_mode} --dataset ${data} --transformer_mode ${tf_mode} --gpu 1