'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from predict_dataset import PredictDatasetBert, PredictDatasetLSTM
from predict_forward import predict_forward_bert, predict_forward_lstm
import os
import sys
sys.path.append("../")
sys.path.append("../rule_base")
sys.path.append("../bert_cover")
from metric import acc
from rule_base.rule_match import match
from rule_base.load_icd_v2 import load_train_icd, load_icd_code
from rule_base.split_v2 import split
from transformers import AutoTokenizer, AutoModel, AutoConfig
from bert_cover.tokenize_util import decode
from bert_cover.data_util import load_train_test
from tqdm import trange, tqdm
import pandas as pd
import datetime
from postprocess import reject_pred
from exact_match import exact_match
from jump_match import jump_match
import pickle


def get_y_option_online(x, k = 100):
    print(x)
    return [x]


def get_y_option(x_list, k = 100, use_match = True, top_k_dict = "/media/sdd1/Hongyi_Yuan/CHIP2020/Final/result_dict_test/Dice_tfidfjson.pkl"):
    dic = np.load(top_k_dict, allow_pickle = True)
    y_list = []
    for x in x_list:
        y = []
        count = k
        if use_match:
            match_y_option = match(x)
            # print(match_y_option)
            y += match_y_option
            count -= len(match_y_option)
        if x in dic:
            topk = []
            for term in dic.get(x)[:k]:
                if term not in y:
                    topk.append(term)
            y += topk[:count]
        else:
            topk = []
            for term in get_y_option_online(x, k):
                if term not in y:
                    topk.append(term)
            y += topk[:count]
        y_list.append(y)
    return y_list


def split_x(x_list):
    split_ids = []
    split_x_list = []
    for i in range(len(x_list)):
        temp_x_list = split(x_list[i])
        for x in temp_x_list:
            split_x_list.append(x)
            split_ids.append(i)
    return split_ids, split_x_list


def score(k, th, count, probability, x_ids, y_option_standard_list, y_list, x_list, y_option_standard, y_option_list, jump_match_y, standard2clean, clean2code, no_predict_default):
    # Count is not avaliable now.
    pred_y_list = [[] for i in range(len(y_list))]
    now_x_count = 0
    for i in range(len(x_ids)):
        if i == 0 or (i > 0 and x_ids[i] != x_ids[i - 1]):
            now_x_count = 0
        now_x_count += 1
        if now_x_count > k:
            continue
        if probability[i] >= th: # or exact_match(x_list[x_ids[i]], y_option_list[i]):
            pred_y_list[x_ids[i]].append(y_option_standard_list[i])

    for x_id in x_ids:
        for y in jump_match_y[x_id]:
            if y not in pred_y_list[x_id]:
                pred_y_list[x_id].append(y)

    pred_y_list = reject_pred(x_list, pred_y_list, icd_reject=True, standard2clean=standard2clean, clean2code=clean2code)

    #with open("same_char_count_dict_0.8_0.129.pkl", "rb") as f:
    with open("same_char_count_dict_0.8_0.20817843866171004.pkl", "rb") as f:
        same_char_count_dict = pickle.load(f)
    for i in range(len(pred_y_list)):
        if len(pred_y_list[i]) == 0:
            if  no_predict_default == 1:
                continue
            if no_predict_default == 2:
                pred_y_list[i].append(y_option_standard[i][0])
            if no_predict_default == 3:
                if x_list[i] in same_char_count_dict:
                    pred_y_list[i].append(same_char_count_dict[x_list[i]])

    return acc(pred_y_list, y_list)


def predict_pipeline_lstm(x_list, embedding_path, model_path, y_list = None, batch_size_x = 2, max_seq_length = 32, jump_max = -1, min_len_y = -1, no_predict_default = 1):
    model = torch.load(model_path, map_location=torch.device("cuda:1"))
    model.eval()
    clean2standard, origin_y, standard2clean, cleaned_y, clean2code = load_icd_code()
    #split_ids, split_x_list = split_x(x_list)
    split_ids = [i for i in range(len(x_list))]
    split_x_list = x_list

    print("Split done.")
    y_option = get_y_option(split_x_list, 200)

    y_option_standard = [[clean2standard.get(y_option[i][j]) for j in range(len(y_option[i]))] for i in range(len(y_option))]
    y_option_list = [y_option[i][j] for i in range(len(y_option)) for j in range(len(y_option[i]))]
    y_option_standard_list = [clean2standard.get(y) for y in y_option_list]
    print("Get standard done.")
    #y_option = y_list
    #print(y_option)
    dataset = PredictDatasetLSTM(split_x_list, y_option, embedding_path, 200, max_seq_length)
    #print(len(dataset))
    dataloader = DataLoader(dataset, batch_size = batch_size_x, drop_last = False)
    print("Load data done.")
    
    #pred_y_list = [[] for i in range(len(x_list))]
    x_ids = []
    split_x_result = []
    label = []
    sim = []
    enum = tqdm(dataloader)
    for index, batch in enumerate(enum):
        input_ids_x, input_ids_y, mask_x, mask_y, x_id = batch[0].to(model.device), batch[1].to(model.device), \
            batch[2].to(model.device), batch[3].to(model.device), batch[4]
        sim_y = predict_forward_lstm(model, input_ids_x, input_ids_y, mask_x, mask_y)
        sim_y = sim_y.reshape(-1, 1).squeeze()
        x_id = x_id.reshape(-1, 1).squeeze()
        x_id_removed = x_id != -1
        x_ids += list(x_id[x_id_removed].numpy())
        split_x_result += [split_x_list[i] for i in x_id[x_id_removed]]
        batch_sim = [sim.item() for sim in sim_y[x_id_removed]]
        sim += batch_sim

    jump_match_y = jump_match(x_list, cleaned_y, clean2standard, jump_max, min_len_y)

    best_acc = 0.0
    best_acc_k = -1
    best_acc_th = -1
    
    best_f1 = 0.0
    best_f1_k = -1
    best_f1_th = -1

    if y_list:
        for k in [1, 10, 20, 30, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
            for th in np.arange(-1, 1, 0.1):
                accuracy, p, r, f1 = score(k, th, None, sim, x_ids, y_option_standard_list, y_list, x_list, y_option_standard, y_option_list, jump_match_y, standard2clean, clean2code, no_predict_default)

                if accuracy > best_acc:
                    best_acc = accuracy
                    best_acc_k = k
                    best_acc_th = th

                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_k = k
                    best_f1_th = th

        print(best_acc, best_acc_k, best_acc_th)
        accuracy, p, r, f1 = score(best_acc_k, best_acc_th, None, sim, x_ids, y_option_standard_list, y_list, x_list, y_option_standard, y_option_list, jump_match_y, standard2clean, clean2code, no_predict_default)

        print(best_f1, best_f1_k, best_f1_th)
        accuracy, p, r, f1 = score(best_f1_k, best_f1_th, None, sim, x_ids, y_option_standard_list, y_list, x_list, y_option_standard, y_option_list, jump_match_y, standard2clean, clean2code, no_predict_default)

    return
  

def predict(model_path, config, tokenizer_path = None, batch_size = 200, max_seq_length = 32, jump_max = -1, min_len_y = -1, no_predict_default = 1, test = True):
    x_list, _, _, _, y_list = load_train_icd()
    if test:
        x_list = x_list[7000:]
        y_list = y_list[7000:]
    else:
        x_list = x_list[:7000]
        y_list = y_list[:7000]
    if config == "lstm":
        acc = predict_pipeline_lstm(x_list, tokenizer_path, model_path, y_list, batch_size, max_seq_length, jump_max, min_len_y, no_predict_default)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config = AutoConfig.from_pretrained(config))
        predict_pipeline_bert(x_list, tokenizer, model_path, y_list, batch_size, max_seq_length, jump_max, min_len_y, no_predict_default)


if __name__ == "__main__":
    
    model_path = "/media/sdd1/GanjinZero/CHIP2020_eval3/bert_cover_char/96_10_2e-05_5_chinese-roberta-wwm-ext_Dice_tfidfjson/model_last.pth"
    tokenizer_path = "/media/sdd1/GanjinZero/CHIP2020_eval3/bert_cover_char/96_10_2e-05_5_chinese-roberta-wwm-ext_Dice_tfidfjson/tok"
    predict(model_path, "hfl/chinese-roberta-wwm-ext", tokenizer_path, 150, 64)
    

    model_path = "/media/sdd1/GanjinZero/CHIP2020_eval3/char_att_lstm_neg/1024_150_0.001_5_word2vec_5_300_Dice_tfidfjson/model_last.pth"
    embedding_path = "/media/sdc/GanjinZero/jiangsu_info/word2vec_5_300.model"

    predict(model_path, "lstm", embedding_path, 15, 64, -1, 3, 1, True)
'''