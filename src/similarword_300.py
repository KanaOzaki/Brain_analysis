#!=usr/bin/python
#-*- coding: utf-8 -*-
#意味表象ベクトルと最も近い単語（品詞ごとも可）上位5個出力してファイルに保存
from gensim import corpora, matutils
import gensim
import h5py
import numpy as np
import scipy.stats as sp


w2v_filename = '../data/entity_vector/nwjc_word_skip_300_8_25_0_1e4_6_1_0_15.txt.vec'
W2V_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_filename)

dict_srm = np.load("../data/Dict/VB/Dict_SN_pred0.55_base900_sec6_sample2.pickle")

dict_srm = np.array(dict_srm)
dict_srm = dict_srm.T #(1097, 1100)

dict_srm = dict_srm[-300:]
dict_srm = dict_srm.T #(1100, 200)
print(dict_srm.shape)

vocab_file=open('../original_data/TV/jawiki160111S1000W10SG_vocab.txt','r')

vocab_list = {}
for line in vocab_file:
    line = line.strip()
    word_list = line.split(' ')
    word_list = word_list[0].split('.')
    vocab_list[word_list[0]] = word_list[1]


for i,vector in enumerate(dict_srm):
    words = W2V_model.most_similar([vector], [], 40)
    print("基底 " + str(i) + "-------")
    for word in words:
        if(word[0] in vocab_list.keys()):
            word_type = vocab_list[word[0]]
            if(word_type == "n" or word_type == "a" or word_type == "v"):
                print (word_type + " " + word[0] + ":" + str(word[1]))
    print("-------------------------------")
