from torch.utils.data import Dataset
import os
import scipy.sparse
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch

class TextData(Dataset):
    def __init__(self, data_dir, partition='train'):
        self.partition = partition
        self.texts_en, self.bow_matrix_en, self.vocab_en, self.word2id_en, self.id2word_en = self.read_data(data_dir, lang='en')
        self.texts_cn, self.bow_matrix_cn, self.vocab_cn, self.word2id_cn, self.id2word_cn = self.read_data(data_dir, lang='cn')
        
        self.size_en = len(self.texts_en)
        self.size_cn = len(self.texts_cn)
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)
        
        self.trans_dict, self.trans_matrix_en, self.trans_matrix_cn = self.parse_dictionary()
        
        self.Map_en2cn = self.get_Map(self.trans_matrix_en, self.bow_matrix_en)
        self.Map_cn2en = self.get_Map(self.trans_matrix_cn, self.bow_matrix_cn)


    def __getitem__(self, idx):
        batch_en = self.bow_matrix_en[idx]
        batch_cn = self.bow_matrix_cn[idx]
        return torch.tensor(batch_en, dtype=torch.float32), torch.tensor(batch_cn, dtype=torch.float32)

    def __len__(self):
        return self.size_en
        
    def read_text(self, path):
        texts = []
        with open(path, 'r') as f:
            for line in f: texts.append(line.strip())
        return texts

    def read_data(self, data_dir, lang):
        texts = self.read_text(os.path.join(data_dir, '{}_texts_{}.txt'.format(self.partition,lang)))
        vocab = self.read_text(os.path.join(data_dir, 'vocab_{}'.format(lang)))
        word2id = dict(zip(vocab, range(len(vocab))))
        id2word = dict(zip(range(len(vocab)), vocab))

        bow_matrix = scipy.sparse.load_npz(os.path.join(data_dir, '{}_bow_matrix_{}.npz'.format(self.partition,lang))).toarray()
        return texts, bow_matrix, vocab, word2id, id2word
    
    def parse_dictionary(self):
        trans_dict = defaultdict(set)
        trans_matrix_en = np.zeros((self.vocab_size_en, self.vocab_size_cn), dtype='int32')
        trans_matrix_cn = np.zeros((self.vocab_size_cn, self.vocab_size_en), dtype='int32')
        
        with open('./ch_en_dict.dat') as f:
            for line in f:
                terms = (line.strip()).split()
                if len(terms) == 2:
                    cn_term = terms[0]
                    en_term = terms[1]
                    if cn_term in self.word2id_cn and en_term in self.word2id_en:
                        trans_dict[cn_term].add(en_term)
                        trans_dict[en_term].add(cn_term)
                        cn_term_id = self.word2id_cn[cn_term]
                        en_term_id = self.word2id_en[en_term]

                        trans_matrix_en[en_term_id][cn_term_id] = 1
                        trans_matrix_cn[cn_term_id][en_term_id] = 1

        return trans_dict, trans_matrix_en, trans_matrix_cn
    
    def get_Map(self, trans_matrix, bow_matrix):
        Map = (trans_matrix * bow_matrix.sum(0)[:, np.newaxis]).astype('float32')
        Map = Map + 1
        Map_sum = np.sum(Map, axis=1)
        t_index = Map_sum > 0
        Map[t_index, :] = Map[t_index, :] / Map_sum[t_index, np.newaxis]
        return torch.tensor(Map)