#!=usr/bin/python
#-*- coding: utf-8 -*-
#推定意味表象行列の各サンプルごとの類似単語を出力

import sys
import gensim
import numpy as np

def main():

    w2v_filename = '../data/entity_vector/nwjc_word_skip_300_8_25_0_1e4_6_1_0_15.txt.vec'
    W2V_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_filename)

    args = sys.argv
    sub = args[1]
    threshold = args[2]
    dimention = int(args[3])
    shift = int(args[4])
    sample = int(args[5])

    # 推定意味表象行列の読み込み
    Estimate_SRM = np.load("../data/Test/VB/ESRM_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    Estimate_SRM = Estimate_SRM.T

    # 語彙ファイルの読み込み
    vocab_file=open('../original_data/NishidaVimeo/jawiki160111S1000W10SG_vocab.txt','r')

    # 語彙リストの作成
    vocab_list = {}
    for line in vocab_file:
        line = line.strip()
        word_list = line.split(' ')
        word_list = word_list[0].split('.')
        vocab_list[word_list[0]] = word_list[1]

    for i,vector in enumerate(Estimate_SRM):
        words = W2V_model.most_similar([vector], [], 20)
        print("サンプル " + str(i) + "-------")
        for word in words:
            if(word[0] in vocab_list.keys()):
                word_type = vocab_list[word[0]]
                if(word_type == "n" or word_type == "a" or word_type == "v"):
                    print (word_type + " " + word[0] + ":" + str(word[1]))
        print("-------------------------------")

if __name__ == '__main__':
    main()
