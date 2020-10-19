from random import sample
import sys
sys.path.append("..")
from rule_base.load_icd_v3 import load_icd

clean2standard, origin_y, standard2clean, cleaned_y = load_icd()


def random_topk(k=200):
    return sample(cleaned_y, k)

def dict_topk(idx, d, k=200):
    return d[idx][0:min(len(d[idx]), k)]