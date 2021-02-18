 #-*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import re
import random

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        super(BaseDataset,self).__init__()
        self.tokenizer = tokenizer
        self.maxlen = args.max_seq_len
        if "train" in mode:
            data_path = os.path.join(args.data_dir, args.train_file)
        elif "dev" in mode:
            data_path = os.path.join(args.data_dir,  args.dev_file)
            if not os.path.isfile(data_path):
                data_path = os.path.join(args.data_dir, args.test_file)
        elif "test" in mode:
            data_path = os.path.join(args.data_dir, args.test_file)
        self.dataset = pd.read_csv(data_path, encoding="utf8", sep="\t")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        txt = str(self.dataset.at[idx,"data"])
        data = self.tokenizer(txt, pad_to_max_length=True, max_length=self.maxlen, truncation=True)
        input_ids = torch.LongTensor(data["input_ids"])
        try:
            token_type_ids = torch.LongTensor(data["token_type_ids"])
        except :
            token_type_ids = None
        attention_mask = torch.LongTensor(data["attention_mask"])
        label = self.dataset.at[idx,"label"]
        if token_type_ids == None:
            return (input_ids, attention_mask, label),txt
        else:
            return (input_ids, token_type_ids, attention_mask, label),txt

    def getLabelNumber(self):
        return len(set(self.dataset["label"]))

DATASET_LIST = {
    "Star_Label_AM": BaseDataset,
    "Star_Label_AM_w_linear": BaseDataset,
    "Star_Label_ANN" : BaseDataset,
    "Star_Label_ANN_w_linear" : BaseDataset,
    "Star_Label_AM_att": BaseDataset,
    "BaseModel": BaseDataset
}