import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, RandomSampler
#from data_util import load_train_test
from time import time
from tokenize_util import load_w2v, tokenize
#from transformers import AutoTokenizer
from random import sample
from topk import random_topk, dict_topk
import ipdb
import pickle
from numpy.random import choice
import numpy as np
import math


class MyDataset(Dataset):
    def __init__(self, x_list, y_list, word2vec_path, topk_dict_path, max_y_count=16, max_length=128):
        self.len = len(x_list)
        self.x_list = x_list
        self.y_list = y_list
        _, self.word2id, _ = load_w2v(word2vec_path)
        self.max_y_count = max_y_count
        self.max_length = max_length
        self.topk_dict_path = topk_dict_path
        self.load_dict()
        self.init_sample_prob()

    def __getitem__(self, index):
        x, x_mask = self.tokenize(self.x_list[index])

        y = []
        y_mask = []
        y_label = []

        for target in self.y_list[index]:
            y_now, y_now_mask = self.tokenize(target)
            y.append(y_now)
            y_mask.append(y_now_mask)
            y_label.append(1)

        #topk_y = rule_topk(self.x_list[index], 200)
        #print(index)
        topk_y = dict_topk(self.x_list[index], self.dict, 200)
        candidate_y = [y for y in topk_y if not y in self.y_list[index]]
        sample_neg_count = max(0, self.max_y_count - len(self.y_list[index]))
        if len(candidate_y) >= sample_neg_count:
            neg_y = choice(candidate_y,
                           sample_neg_count,
                           p=np.array(self.prob[0:len(candidate_y)]) / sum(self.prob[0:len(candidate_y)]),
                           replace=False).tolist()
        else:
            neg_y = candidate_y + random_topk(sample_neg_count - len(candidate_y))

        for target in neg_y:
            y_now, y_now_mask = self.tokenize(target)
            y.append(y_now)
            y_mask.append(y_now_mask)
            y_label.append(-1)       

        return (x, y, x_mask, y_mask, y_label)
        #return (self.tokenize(self.x_list[index]), [self.tokenize(y) for y in self.y_list[index]])

    def __len__(self):
        return self.len

    def tokenize(self, sentence):
        tokenized = tokenize([sentence], self.max_length, self.word2id)[0]
        mask = [1 if tok > 0 else 0 for tok in tokenized]
        return tokenized, mask

    def init_sample_prob(self):
        self.prob = [1/math.log(2+i) for i in range(1000)]

    def load_dict(self):
        with open(self.topk_dict_path, "rb") as f:
            self.dict = pickle.load(f)
        #self.dict = list(self.dict.values())
        return None


class FixedLengthBatchSampler(Sampler):
    def __init__(self, sampler, fixed_length, drop_last):
        self.sampler = sampler
        self.fixed_length = fixed_length
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        now_length = 0
        for idx in self.sampler:
            sample_length = len(self.sampler.data_source[idx][1]) + 1
            if now_length + sample_length > self.fixed_length:
                yield batch
                batch = []
                now_length = 0
            batch.append(idx)
            now_length += sample_length
        if len(batch) > 0 and not self.drop_last:
            yield batch

def my_collate_fn(batch):
    x = []
    y = []
    x_mask = []
    y_mask = []
    label = []

    for idx, item in enumerate(batch):
        x.append(item[0])
        x_mask.append(item[2])
        tmp_y = []
        tmp_y_mask = []
        for idx_y, term in enumerate(item[1]):
            tmp_y.append(term)
            tmp_y_mask.append(item[3][idx_y])
        y.append(tmp_y)
        y_mask.append(tmp_y_mask)
        label.append(item[4])

    output = (torch.LongTensor(x), torch.LongTensor(y), torch.FloatTensor(label), torch.BoolTensor(x_mask), torch.BoolTensor(y_mask))
    return output

def my_dataloader(my_dataset, fixed_length=32, num_workers=1):
    base_sampler = RandomSampler(my_dataset)
    batch_sampler = FixedLengthBatchSampler(sampler=base_sampler, fixed_length=fixed_length, drop_last=True)
    dataloader = DataLoader(my_dataset, batch_sampler=batch_sampler, collate_fn=my_collate_fn, num_workers=num_workers, pin_memory=True)
    return dataloader

if __name__ == "__main__":
    """
    x_train, y_train, x_test, y_test = load_train_test(test_size=100)
    embedding_path = "/media/sdd1/Hongyi_Yuan/CHIP2020/Final/result_dict_test/Dice_tfidfjson.pkl"
    my_dataset = MyDataset(x_train, y_train, "/media/sdc/GanjinZero/jiangsu_info/word2vec_5_300.model", embedding_path, max_length=128)
    my_dataloader = my_dataloader(my_dataset, fixed_length=64, num_workers=1)
    now_time = time()
    for index, batch in enumerate(my_dataloader):
        print(time() - now_time)
        now_time = time()
        if index < 3:
            for item in batch:
                #print(item)
                print(item.shape)
            print(item[-1])
        else:
            import sys
            sys.exit()
    """