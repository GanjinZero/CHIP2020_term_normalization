import os
from tqdm import tqdm
import sys
sys.path.append("..")
from load_icd_v3 import load_icd, load_additional_icd_for_train, load_icd_for_train


icd2str, match_list = load_icd()

def clean_text(text):
    return text

def train_test_split(test_size=100):
    with open("../train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    x_list = []
    y_list = []
    for line in lines:
        l = line.strip().split("\t")
        x_list.append(l[0].lower())
        y_list.append(l[1].split("##"))

    len_x = len(x_list)
    if test_size > 0:
        return x_list[0:(len_x-test_size)], y_list[0:(len_x-test_size)], x_list[(len_x-test_size):], y_list[(len_x-test_size):]
    else:
        return x_list, y_list, None, None
    
def load_train_test(test_size=100, use_icd10=False, use_additional_icd=False):
    x_train, y_train, x_test, y_test = train_test_split(test_size)

    if use_icd10:
        x_add, y_add = load_icd_for_train()
        x_train = x_train + x_add
        y_train = y_train + y_add
    if use_additional_icd:
        pass
        """
        x_add, y_add = load_additional_icd_for_train()
        x_train = x_train + x_add
        y_train = y_train + y_add
        """
    else:
        x_add, y_add = [], []

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_train_test(test_size=100)
    print(len(x_train), len(x_test))
    x_train, y_train, x_test, y_test = load_train_test(test_size=100, use_additional_icd=True)
    print(len(x_train), len(x_test))

    print(x_train[0:5], y_train[0:5])
    print(x_train[-5:], y_train[-5:])
