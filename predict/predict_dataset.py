import torch
from torch.utils.data import Dataset
import sys
sys.path.append("../lstm/")
from tokenize_util import tokenize
import pickle


class PredictDatasetLSTM(Dataset):
    def __init__(self, x_list, y_list, word2vec_path, batch_size_y, max_length=64):
        self.len = len(x_list)
        self.x_list = x_list
        self.y_list = y_list
        self.batch_size_y = batch_size_y
        with open(word2vec_path, "rb") as f:
            self.word2id = pickle.load(f)
        self.max_length = max_length

    def __getitem__(self, index):
        x, x_mask = self.tokenize(self.x_list[index])

        y = []
        y_mask = []
        x_id = []

        for target in self.y_list[index]:
            y_now, y_now_mask = self.tokenize(target)
            y.append(y_now)
            y_mask.append(y_now_mask)
            x_id.append(index)

        # 将被 reject 的y_list 补齐至 batch_size_y 的长度，x_id=-1 做标识，外部处理
        l = len(y)
        assert l > 0
        if l < self.batch_size_y:
            y += [y[0]] * (self.batch_size_y - l)
            y_mask += [y_mask[0]] * (self.batch_size_y - l)
            x_id += [-1] * (self.batch_size_y - l)

        return torch.LongTensor(x), torch.LongTensor(y), torch.BoolTensor(x_mask), torch.BoolTensor(y_mask), torch.IntTensor(x_id)

    def __len__(self):
        return self.len

    def tokenize(self, sentence):
        tokenized = tokenize([sentence], self.max_length, self.word2id)[0]
        mask = [1 if tok > 0 else 0 for tok in tokenized]
        return tokenized, mask
