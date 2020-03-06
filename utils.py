import pandas as pd
import numpy as np


pad = '[PAD]'
unk = '[UNK]'
amino_acid = [pad, 'A','I','L','V','F','W','Y','N','C','Q','M','S',
              'T','D','E','R','H','K','G','P','O','U','X','B','Z', unk]
seq_dic = {w: i for i,w in enumerate(amino_acid)}


def pro2idx(pro, max_seq_len = 700):
    ids = []
    for w in pro:
        if w in seq_dic:
            ids.append(seq_dic[w])
        else:
            ids.append(seq_dic[unk])
    while len(ids) < max_seq_len:
        ids.append(seq_dic[pad])
    return np.array(ids[:max_seq_len])


def load_data(max_seq_len = 700):
    train = pd.read_csv('../data/deepbio/coreseed.train.tsv', sep='\t')
    test = pd.read_csv('../data/deepbio/coreseed.test.tsv', sep='\t')
    
    train_pro = train['protein']
    test_pro = test['protein']

    train_pro = [i for i in train_pro if len(i) <= max_seq_len]
    test_pro = [i for i in test_pro if len(i) <= max_seq_len]
    print("{:.2f}% training proteins ({}) selected with length <= {}".format(
        len(train_pro) / len(train['protein']) * 100, len(train_pro), max_seq_len))
    print("{:.2f}% testing proteins ({}) selected with length <= {}".format(
        len(test_pro) / len(test['protein']) * 100, len(test_pro), max_seq_len))
    
    return train_pro, test_pro
