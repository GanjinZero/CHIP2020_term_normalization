from gensim.models import Word2Vec
import numpy as np


def load_w2v(file_path):
    W = Word2Vec.load(file_path)
    word_lst = ["[PAD]"] + W.wv.index2word + ["[UNK]"]
    word2id = {word:idx for idx, word in enumerate(word_lst)}
    id2word = {idx:word for word, idx in word2id.items()}
    vector = W.wv.vectors
    vector_mean = np.mean(vector, axis=0)
    vector_word = np.concatenate((np.zeros_like(vector_mean).reshape(1, -1), vector, vector_mean.reshape(1, -1)), axis=0)
    print(vector_word.shape)
    return vector_word, word2id, id2word

def tokenize(string_lst, max_length, word2id):
    unk_id = len(word2id) - 1
    input_ids = []
    for string in string_lst:
        input_id = [word2id.get(ch, unk_id) for ch in string]
        if len(input_id) > max_length:
            input_ids.append(input_id[0:max_length])
        else:
            input_ids.append(input_id + [0] * (max_length - len(input_id)))
    return input_ids

def decode(input_ids, id2word):
    unk_id = len(id2word) - 1
    decoded = []
    for input_id in input_ids:
        now = "".join([id2word[id] for id in input_id if id > 0 and id != unk_id])
        decoded.append(now)
    return decoded

if __name__ == "__main__":
    vector_word, word2id, id2word = load_w2v("/media/sdc/GanjinZero/jiangsu_info/word2vec_5_300.model")
    string_lst = ["疾病", "感冒，发烧"]
    input_ids = tokenize(string_lst, 16, word2id)
    print([id2word[i] for i in range(10)])
    print(input_ids)
    decoded = decode(input_ids, id2word)
    print(decoded)
