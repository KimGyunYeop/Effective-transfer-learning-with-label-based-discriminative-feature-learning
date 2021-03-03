import torch
import torch.nn.functional as F
from torch import nn

from src import (
    MODEL_ORIGINER
)
from transformers import AutoModel

class BaseModel(nn.Module):
    def __init__(self, transformers_mode, model_type, model_name_or_path, config, labelNumber, margin=-0.5):
        super(BaseModel, self).__init__()
        self.emb = AutoModel.from_pretrained(transformers_mode)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, labelNumber)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.labelNumber = labelNumber
        self.margin = margin

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        if token_type_ids is None:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, self.labelNumber), labels.view(-1))

        result = (loss1, outputs, embs[:, 0, :])

        return result


class Star_Label_AM(nn.Module):
    def __init__(self, transformers_mode, model_type, model_name_or_path, config, labelNumber, margin=-0.5):
        super(Star_Label_AM, self).__init__()
        self.emb = AutoModel.from_pretrained(transformers_mode)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, labelNumber)
        self.star_emb = nn.Embedding(labelNumber, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.labelNumber = labelNumber
        self.margin = margin

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        if token_type_ids is None:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        #print(outputs)
        #print(torch.argmax(outputs, axis=1))
        #print(labels)
        #print((torch.argmax(outputs, axis=1) == labels).float().mean())

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, self.labelNumber), labels.view(-1))
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=self.margin)
        loss2s = []
        for i in range(batch_size):
            diff_indexs = labels == labels[i].repeat(batch_size)
            diff_label_datas = embs[diff_indexs,0,:].squeeze()
            stretch_ori_datas = embs[i, 0, :].repeat(sum(diff_indexs),1)
            loss2s.append(loss_fn(diff_label_datas.view(-1, w2v_dim),
                        stretch_ori_datas.view(-1, w2v_dim),
                        torch.ones(sum(diff_indexs)).to(self.config.device)))

        loss2 = sum(loss2s)/len(loss2s)

        #calculate loss with same label's represntation vector
        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs, embs[:, 0, :])

        return result


class Star_Label_AM_w_linear(nn.Module):
    def __init__(self, transformers_mode, model_type, model_name_or_path, config, labelNumber, margin=-0.5):
        super(Star_Label_AM_w_linear, self).__init__()
        self.emb = AutoModel.from_pretrained(transformers_mode)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, labelNumber)
        self.star_emb = nn.Linear(labelNumber, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.labelNumber = labelNumber
        self.margin = margin

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        if token_type_ids is None:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        #print(outputs)
        #print(torch.argmax(outputs, axis=1))
        #print(labels)
        #print((torch.argmax(outputs, axis=1) == labels).float().mean())

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, self.labelNumber), labels.view(-1))
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=self.margin)
        loss2s = []
        for i in range(batch_size):
            diff_indexs = labels == labels[i].repeat(batch_size)
            diff_label_datas = embs[diff_indexs,0,:].squeeze()
            stretch_ori_datas = embs[i, 0, :].repeat(sum(diff_indexs),1)
            loss2s.append(loss_fn(diff_label_datas.view(-1, w2v_dim),
                        stretch_ori_datas.view(-1, w2v_dim),
                        torch.ones(sum(diff_indexs)).to(self.config.device)))

        loss2 = sum(loss2s)/len(loss2s)

        #calculate loss with same label's represntation vector
        ont_hot_labels = torch.zeros((batch_size,self.labelNumber)).to(self.config.device)
        ont_hot_labels[range(batch_size),labels] = 1
        star = self.star_emb(ont_hot_labels)
        #spread = torch.zeros((self.labelNumber,self.labelNumber)).to(self.config.device)
        #spread[range(self.labelNumber), range(self.labelNumber)] = 1
        #print(self.star_emb(spread))

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs, embs[:, 0, :])

        return result

class Star_Label_AM_att(nn.Module):
    def __init__(self, model_type, model_name_or_path, config, margin=-0.5):
        super(Star_Label_AM_att, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.lstm = nn.LSTM(768,768)
        self.star_emb = nn.Embedding(2, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.att_w = nn.Parameter(torch.randn(1, 768, 1))
        self.margin = margin

    def attention_net(self, lstm_output, input):
        batch_size, seq_len = input.shape

        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        if token_type_ids is None:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs, _ = self.lstm(embs)
        outputs = self.attention_net(outputs,input_ids)
        outputs = self.dense(outputs)
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))

        #make 2 data for match all2all data
        x1 = embs[:, 0, :].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:, 0, :].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        y = labels.unsqueeze(0).repeat(batch_size, 1).type(torch.FloatTensor).to(self.config.device)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=self.margin)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                        x2.view(-1, w2v_dim),
                        y.view(-1))

        #calculate loss with same label's represntation vector
        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs, embs[:, 0, :])

        return result

class Star_Label_ANN(nn.Module):
    def __init__(self, transformers_mode, model_type, model_name_or_path, config, labelNumber, margin=-0.5):
        super(Star_Label_ANN, self).__init__()
        self.emb = AutoModel.from_pretrained(transformers_mode)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, labelNumber)
        self.star_emb = nn.Embedding(labelNumber, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.labelNumber = labelNumber
        self.margin = margin

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        if token_type_ids is None:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        #print(outputs)
        #print(torch.argmax(outputs, axis=1))
        #print(labels)
        #print((torch.argmax(outputs, axis=1) == labels).float().mean())

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, self.labelNumber), labels.view(-1))
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=self.margin)
        loss2s = []
        for i in range(batch_size):
            diff_indexs = labels != labels[i].repeat(batch_size)
            diff_label_datas = embs[diff_indexs,0,:].squeeze()
            stretch_ori_datas = embs[i, 0, :].repeat(sum(diff_indexs),1)
            loss2s.append(loss_fn(diff_label_datas.view(-1, w2v_dim),
                        stretch_ori_datas.view(-1, w2v_dim),
                        -torch.ones(sum(diff_indexs)).to(self.config.device)))

        loss2 = sum(loss2s)/len(loss2s)

        #calculate loss with same label's represntation vector
        star = self.star_emb(labels)

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs, embs[:, 0, :])

        return result


class Star_Label_ANN_w_linear(nn.Module):
    def __init__(self, transformers_mode, model_type, model_name_or_path, config, labelNumber, margin=-0.5):
        super(Star_Label_ANN_w_linear, self).__init__()
        self.emb = AutoModel.from_pretrained(transformers_mode)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, labelNumber)
        self.star_emb = nn.Linear(labelNumber, 768)
        self.config = config
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.labelNumber = labelNumber
        self.margin = margin

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        if token_type_ids is None:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:, 0, :])
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        #print(outputs)
        #print(torch.argmax(outputs, axis=1))
        #print(labels)
        #print((torch.argmax(outputs, axis=1) == labels).float().mean())

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, self.labelNumber), labels.view(-1))
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=self.margin)
        loss2s = []
        for i in range(batch_size):
            diff_indexs = labels != labels[i].repeat(batch_size)
            diff_label_datas = embs[diff_indexs,0,:].squeeze()
            stretch_ori_datas = embs[i, 0, :].repeat(sum(diff_indexs),1)
            loss2s.append(loss_fn(diff_label_datas.view(-1, w2v_dim),
                        stretch_ori_datas.view(-1, w2v_dim),
                        -torch.ones(sum(diff_indexs)).to(self.config.device)))

        loss2 = sum(loss2s)/len(loss2s)

        #calculate loss with same label's represntation vector
        ont_hot_labels = torch.zeros((batch_size,self.labelNumber)).to(self.config.device)
        ont_hot_labels[range(batch_size),labels] = 1
        star = self.star_emb(ont_hot_labels)
        #spread = torch.zeros((self.labelNumber,self.labelNumber)).to(self.config.device)
        #spread[range(self.labelNumber), range(self.labelNumber)] = 1
        #print(self.star_emb(spread))

        loss3 = loss_fn(embs[:, 0, :].squeeze(),
                        star,
                        torch.ones(batch_size).to(self.config.device))

        result = ((loss1, 0.5 * loss2, 0.5 * loss3), outputs, embs[:, 0, :])

        return result

MODEL_LIST = {
    "Star_Label_AM": Star_Label_AM,
    "Star_Label_AM_w_linear": Star_Label_AM_w_linear,
    "Star_Label_ANN" : Star_Label_ANN,
    "Star_Label_ANN_w_linear" : Star_Label_ANN_w_linear,
    "Star_Label_AM_att": Star_Label_AM_att,
    "BaseModel": BaseModel
}