import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys
sys.path.append("../")
from predict.same_char_count_3 import find_same_char_max
from predict.predict_dataset import PredictDatasetLSTM
from predict.predict_forward import predict_forward_lstm
from predict.jump_match import jump_match
from rule_base.load_icd_v3 import load_train_icd, load_icd_code, clean
from rule_base.rule_match import match
from rule_base.bm25 import get_word_tfidf, fill_co_exsit_dict2, string_clean_seg, set_find
from rule_base.postprocess import reject_pred
from utils.syndrone import syndrome
import json
import numpy as np


@st.cache
def init():
    #x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list = load_train_icd()
    #clean2standard, origin_y, standard2clean, cleaned_y, clean2code = load_icd_code()
    with open("data/clean2standard.json", "r", encoding="utf-8") as f:
        clean2standard = json.load(f)
    with open("data/standard2clean.json", "r", encoding="utf-8") as f:
        standard2clean = json.load(f)
    with open("data/clean2code.json", "r", encoding="utf-8") as f:
        clean2code = json.load(f)
    x_list = np.load("data/x_list.npy").tolist()
    y_clean_list = np.load("data/y_clean_list.npy", allow_pickle=True).tolist()
    y_list = np.load("data/y_list.npy", allow_pickle=True).tolist()
    y_clean_icd10_list = np.load("data/y_clean_icd10_list.npy", allow_pickle=True).tolist()
    y_icd10_list = np.load("data/y_icd10_list.npy", allow_pickle=True).tolist()
    origin_y = np.load("data/origin_y.npy", allow_pickle=True).tolist()
    cleaned_y = np.load("data/cleaned_y.npy", allow_pickle=True).tolist()
    return x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list, \
           clean2standard, origin_y, standard2clean, cleaned_y, clean2code

@st.cache
def load_lstm_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    return model


@st.cache
def calculate_co_exist():
    with open("data/co_exist.json", "r", encoding="utf-8") as f:
        co_exist = json.load(f)
    return co_exist


def get_y_option_online(x, cleaned_y, co_exist, k=150, use_match=True):
    tmp_x = ",".join(string_clean_seg(x))
    now_y = {}
    word_cut = list(set(tmp_x))
    for idx, y in enumerate(cleaned_y):
        tfidf_score = 0
        dice_score = set_find(tmp_x, y, 'dice')
        now_y[y] = dice_score
        for word in word_cut:
            if word + '-' + y in co_exist.keys():
                tfidf_score +=  co_exist[word + '-' + y] * 2000
        now_y[y] += tfidf_score
    pred_y = [a[0] for a in sorted(now_y.items(), key = lambda kv: -kv[1]) if a[1] > 0]

    if use_match:
        match_y_option = match(x)
        k -= len(match_y_option)
    if len(pred_y) > k:
        pred_y = pred_y[0:k]
    if use_match:
        return match_y_option + pred_y
    return pred_y

@st.cache
def get_set_y(cleaned_y):
    return [set(y) for y in cleaned_y]


def main():
    st.title("中文医学术语标准化：ICD-10")
    st.write("清华大学统计中心")

    text = st.text_input("", "大隐静脉曲张并且血栓性静脉炎")
    run = st.button("Normalize!")

    x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list, \
        clean2standard, origin_y, standard2clean, cleaned_y, clean2code = init()
    co_exist = calculate_co_exist()
    model_path = "../models/model_last.pth"
    model = load_lstm_model(model_path)

    # Define parameters
    k = 150
    max_seq_length = 32
    embedding_path = "../models/word2id.pkl"

    if run:
        # step 1: dice + tfidf
        clean_text = clean(text)
        y_option = get_y_option_online(clean_text, cleaned_y, co_exist, k, use_match=True)
        y_option_standard = [clean2standard.get(y_option[j]) for j in range(len(y_option))]

        # step 2: lstm
        lstm_option_y = []
        dataset = PredictDatasetLSTM([clean_text], [y_option], embedding_path, k, max_seq_length)
        input_ids_x, input_ids_y, x_mask, y_mask, x_id = dataset[0]
        input_ids_x = input_ids_x.unsqueeze(0)
        input_ids_y = input_ids_y.unsqueeze(0)
        x_mask = x_mask.unsqueeze(0)
        y_mask = y_mask.unsqueeze(0)
        sim_y = predict_forward_lstm(model, input_ids_x, input_ids_y, x_mask, y_mask)
        y_count = len(y_option)
        sim_y = sim_y[0][0:y_count]
        for idx, y in enumerate(y_option):
            if sim_y[idx] >= 0.8:
                lstm_option_y.append(y)
        #st.write(y_option)
        #st.write(lstm_option_y)

        # step 3: jump
        jump_match_y = jump_match([clean_text], cleaned_y, clean2standard, jump_max=-1, min_len_y=3)[0]

        # step 4: find same char count
        set_y = get_set_y(cleaned_y)
        t, s = find_same_char_max(clean_text, cleaned_y, clean2standard, 0.8, set(clean_text), set_y)

        # step 5: reject
        pred_y = reject_pred([clean_text], [list(set(lstm_option_y + jump_match_y + t))], icd_reject=True, standard2clean=standard2clean, clean2code=clean2code)

        # step 6: syndrome
        out_y = syndrome([clean_text], pred_y, origin_y)
        for i in range(len(out_y[0])):
            st.write(f"- {out_y[0][i]}")

main()
