#!=usr/bin/python
#-*- coding: utf-8 -*-
#get_similarword_train.py

##このプログラムは、指定された基底に対して
##意味表象辞書ベクトルとcos類似度が高い単語を出力する.
## コマンド引数
##1.被験者名 2.予測精度の閾値 3.基底数 4.time lag 5.間引き数(何sampleに1枚抜くか)

import gensim
import numpy as np
import sys
import pickle

def main():

    w2v_filename = '../data/entity_vector/nwjc_word_skip_300_8_25_0_1e4_6_1_0_15.txt.vec'
    W2V_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_filename)

    args = sys.argv
    sub = args[1]
    threshold = args[2]
    dimention = int(args[3])
    shift = int(args[4])
    sample = int(args[5])

    # 対象基底番号の辞書を読み込み、基底番号だけのリストにする
    sample_base_dict = np.load("../data/base/base_" + sub + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    base_nums = [a for (a,b) in sample_base_dict.values()]

    # 訓練で作った辞書の読み込み
    Dict_train = np.load("../data/Dict/VB/Dict_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    # 意味表象辞書の部分だけ取り出す
    SRM_dict = Dict_train.T[-300:]
    SRM_dict = SRM_dict.T
    print("意味表象辞書")
    print(SRM_dict.shape)

    # 語彙ファイルの読み込み
    vocab_file=open('../original_data/NishidaVimeo/jawiki160111S1000W10SG_vocab.txt','r')

    # 語彙リストの作成
    vocab_list = {}
    for line in vocab_file:
        line = line.strip()
        word_list = line.split(' ')
        word_list = word_list[0].split('.')
        vocab_list[word_list[0]] = word_list[1]


    for num in base_nums:
        words = W2V_model.most_similar([SRM_dict[num]], [], 20)
        print("基底" + str(num) + "-------")
        for word in words:
            if(word[0] in vocab_list.keys()):
                word_type = vocab_list[word[0]]
                if(word_type == "n" or word_type == "a" or word_type == "v"):
                    print (word_type + " " + word[0] + ":" + str(word[1]))
        print("-------------------------------")

if __name__ == '__main__':
    main()
