import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from predict_dataset import PredictDatasetLSTM
from predict_forward import predict_forward_lstm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("../")
sys.path.append("../rule_base")
from utils.metric import acc
from rule_base.rule_match import match
from rule_base.load_icd_v3 import load_train_icd, load_icd_code
from rule_base.load_icd_v3 import load_test_samples as load_test_icd
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import trange, tqdm
import pandas as pd
import datetime
from postprocess import reject_pred
from jump_match import jump_match
import pickle
import ipdb


def get_y_option_online(x, k = 100):
    return [x] * k


def get_y_option(x_list, k = 100, use_match = True, top_k_dict = '../models/dice_test.pkl'):#"/media/sdd1/Hongyi_Yuan/CHIP2020/Final/result_dict_test/Dice_tfidfjson.pkl"):
    dic = np.load(top_k_dict, allow_pickle = True)
    #ipdb.set_trace()
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
            print(f"Not found {x}")
            topk = []
            for term in get_y_option_online(x, k):
                if term not in y:
                    topk.append(term)
            y += topk[:count]
        assert len(y) == k
        y_list.append(y)
    return y_list



def predict_pipeline_lstm(x_list, embedding_path, model_path, y_list = None, k = 100, batch_size_x = 2, 
threshold = 0, max_seq_length = 32, if_jump = None, no_predict_default = 4, mode='test'):
    device = torch.device("cuda")
    model = torch.load(model_path, map_location=device)
    model.eval()
    clean2standard, origin_y, standard2clean, cleaned_y, clean2code = load_icd_code()
    split_ids = [i for i in range(len(x_list))]
    split_x_list = x_list

    print("Split done.")
    y_option = get_y_option(split_x_list, k)

    y_option_standard = [[clean2standard.get(y_option[i][j]) for j in range(len(y_option[i]))] for i in range(len(y_option))]
    y_option_list = [y_option[i][j] for i in range(len(y_option)) for j in range(len(y_option[i]))]
    y_option_standard_list = [clean2standard.get(y) for y in y_option_list]
    print("Get standard done.")
    #y_option = y_list
    #print(y_option)
    dataset = PredictDatasetLSTM(split_x_list, y_option, embedding_path, k, max_seq_length)
    #print(len(dataset))
    dataloader = DataLoader(dataset, batch_size = batch_size_x, drop_last = False)
    print("Load data done.")

    pred_y_list = [[] for i in range(len(x_list))]
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
        label += [p >= threshold for p in batch_sim]

    for i in range(len(x_ids)):
        if label[i] == 1: # or exact_match(x_list[x_ids[i]], y_option_list[i]):
            pred_y_list[x_ids[i]].append(y_option_list[i])

    if if_jump:
        pred_y_list_jump = [[] for i in range(len(x_list))]
        jump_max, min_len_y = if_jump
        jump_match_y = jump_match(x_list, cleaned_y, clean2standard, jump_max, min_len_y)
        for i in range(len(x_list)):
            for y in jump_match_y[i]:
                if y not in pred_y_list[i]:
                    pred_y_list[i].append(standard2clean[y])
                    pred_y_list_jump[i].append(standard2clean[y])
    #ipdb.set_trace()

    pred_y_list_none = [[] for i in range(len(x_list))]
    for i in range(len(pred_y_list)):
        if len(pred_y_list[i]) == 0:
            if  no_predict_default == 1:
                pass
            if no_predict_default == 2:
                #pred_y_list[i].append(y_option_standard[i][0])
                #pred_y_list_none[i].append(y_option_standard[i][0])
                pass
            if no_predict_default == 3:
                '''
                with open("../models/same_char_count_dict.pkl", "rb") as f:
                    same_char_count_dict = pickle.load(f)
                if x_list[i] in same_char_count_dict:
                    pred_y_list[i].append(standard2clean[same_char_count_dict[x_list[i]]])
                    pred_y_list_none[i].append(standard2clean[same_char_count_dict[x_list[i]]])
                '''
                pass
            if no_predict_default == 4:
                with open("../models/rejected_dic_tmp.pkl", "rb") as f:
                    same_char_count_dict = pickle.load(f)
                if x_list[i] in same_char_count_dict:
                    pred_y_list[i] += [standard2clean[xx] for xx in same_char_count_dict[x_list[i]]]
                    pred_y_list_none[i] += [standard2clean[xx] for xx in same_char_count_dict[x_list[i]]]


    for i in range(len(pred_y_list_jump)):
        for y in pred_y_list_jump[i]:
            if y not in pred_y_list[i]:
                pred_y_list_jump[i].remove(y)

    pred_y_list = reject_pred(x_list, pred_y_list, icd_reject=True, standard2clean=standard2clean, clean2code=clean2code)

    #acc(pred_y_list, y_list)

    pred_y_list = [[clean2standard[yy] for yy in y] for y in pred_y_list]

    if y_list:
        if_pos = [y_option_standard[i][j] in y_list[i] for i in range(len(y_option_standard)) for j in range(len(y_option_standard[i]))]
        print("Matched term count: " + str(sum(if_pos)))
        results = pd.DataFrame({"x_id":x_ids, "split_x":split_x_result, "y_option":y_option_list, "y_option_standard":y_option_standard_list, 
            "if_pos":if_pos, "label":label, "similarity":sim})
        results = results[results["if_pos"] | results["label"]]

    else:
        results = pd.DataFrame({"x_id":x_ids, "split_x":split_x_result, "y_option":y_option_list, "y_option_standard":y_option_standard_list, 
            "label":label, "similarity":sim})

    
    results.to_csv("max" + mode + model_path.split("/")[-2] + f"_{k}_{threshold}_{no_predict_default}.csv")

    #with open("max" + mode + model_path.split("/")[-2] + f"_{k}_{threshold}_{if_jump}_{no_predict_default}.txt", "w", encoding="utf-8") as f:
    with open("final_results_tmp.txt", "w", encoding="utf-8") as f:
        for idx, x in enumerate(x_list):
            if y_list:
                f.write(x + "\t" + "##".join(pred_y_list[idx]) + "\t" + "##".join(y_list[idx]) + "\n")
            else:
                f.write(x + "\t" + "##".join(pred_y_list[idx]) + "\n")

    if if_jump:
        with open("max" + mode + model_path.split("/")[-2] + f"_{k}_{threshold}_{if_jump}_{no_predict_default}_jump.txt", "w", encoding="utf-8") as f:
            for idx, x in enumerate(x_list):
                if y_list:
                    if len(pred_y_list_jump[idx]) > 0:
                        f.write(x + "\t" + "##".join(pred_y_list_jump[idx]) + "\t" + "##".join(y_list[idx]) + "\n")
                else:
                    if len(pred_y_list_jump[idx]) > 0:
                        f.write(x + "\t" + "##".join(pred_y_list_jump[idx]) + "\n")

    with open("max" + mode + model_path.split("/")[-2] + f"_{k}_{threshold}_{if_jump}_{no_predict_default}_none.txt", "w", encoding="utf-8") as f:
        for idx, x in enumerate(x_list):
            if y_list:
                if len(pred_y_list_none[idx]) > 0:
                    f.write(x + "\t" + "##".join(pred_y_list_none[idx]) + "\t" + "##".join(y_list[idx]) + "\n")
            else:
                if len(pred_y_list_none[idx]) > 0:
                    f.write(x + "\t" + "##".join(pred_y_list_none[idx]) + "\n")

    if y_list:
        return acc(pred_y_list, y_list)
    else:
        return pred_y_list
    

def predict(model_path, config, tokenizer_path = None, top_k = 100, batch_size = 200, threshold = 0.99, 
max_seq_length = 32, mode = 'test', if_jump = None, no_predict_default = 4):
    if mode != 'test':
        x_list, _, _, _, y_list = load_train_icd()
        if mode == 'train':
            x_list = x_list[7000:]
            y_list = y_list[7000:]
        if mode == 'dev':
            x_list = x_list[:7000]
            y_list = y_list[:7000]
    if mode == 'test':
        x_list = load_test_icd()
        y_list = None
    if config == "lstm":
        acc = predict_pipeline_lstm(x_list, tokenizer_path, model_path, y_list, top_k, batch_size, threshold, max_seq_length, if_jump, no_predict_default)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config = AutoConfig.from_pretrained(config))
        #acc = predict_pipeline_bert(x_list, tokenizer, model_path, y_list, top_k, batch_size, threshold, max_seq_length, if_jump, no_predict_default)


if __name__ == "__main__":

    model_path = "../models/model_last.pth"
    embedding_path = "../models/word2id.pkl"
    #predict(model_path, "lstm", embedding_path, 90, 30, 0.7, 64)
    #predict(model_path, "lstm", embedding_path, 150, 15, 0.8, 64, 'test', (-1, 3), 1)
    #predict(model_path, "lstm", embedding_path, 150, 15, 0.8, 64, 'test', (-1, 3), 3)
    predict(model_path, "lstm", embedding_path, 150, 15, 0.8, 64, 'test', (-1, 3), 4)
    #predict(model_path, "lstm", embedding_path, 80, 15, 0.8, 64, 'test', (-1, 3), 3)
    #predict(model_path, "lstm", embedding_path, 90, 15, 0.8, 64, 'test', (-1, 3), 3)
    #predict(model_path, "lstm", embedding_path, 90, 15, 0.7, 64, 'test', (-1, 3), 3)
    #predict(model_path, "lstm", embedding_path, 150, 10, 0.8, 64, 'test', (2, 5), 3)
    #predict(model_path, "lstm", embedding_path, 80, 15, 0.8, 64, 'test', (2, 5), 3)
    #predict(model_path, "lstm", embedding_path, 90, 15, 0.7, 64, 'test', (2, 5), 3)
    #predict(model_path, "lstm", embedding_path, 150, 30, 0.8, 64, True, (-1, 3), 2)
    
