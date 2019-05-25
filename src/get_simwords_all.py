#!=usr/bin/python
#-*- coding: utf-8 -*-
#get_simwords_all.py

##このプログラムは、推定意味表象行列の各サンプルごとの類似単語を出力する.
##単語の探索範囲はword2vec空間全てである.(ストップワード設けず)
##コマンド入力
##1.VB/TV 2.被験者名 3.予測精度の閾値 4.基底数 5.time lag 6.間引き数(何sampleに1枚抜くか)

import sys
import gensim
import numpy as np

def main():

    w2v_filename = '../data/entity_vector/nwjc_word_skip_300_8_25_0_1e4_6_1_0_15.txt.vec'
    W2V_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_filename)

    args = sys.argv
    target = args[1]
    sub = args[2]
    threshold = args[3]
    dimention = int(args[4])
    shift = int(args[5])
    sample = int(args[6])

    # 推定意味表象行列の読み込み
    Estimate_SRM = np.load("../data/Test/" + target + "/ESRM_" + sub + "_pred" + threshold + "_base" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    Estimate_SRM = Estimate_SRM.T

    # 語彙ファイルの読み込み
    vocab_file=open('../original_data/TV/jawiki160111S1000W10SG_vocab.txt','r')

    # 語彙リストの作成
    vocab_list = {}
    for line in vocab_file:
        line = line.strip()
        word_list = line.split(' ')
        word_list = word_list[0].split('.')
        vocab_list[word_list[0]] = word_list[1]

    # 推定意味表象行列から1サンプルごとに類似単語を出力する
    for i,vector in enumerate(Estimate_SRM):
        # word2vec空間の中から上位20単語を持ってくる.
        words = W2V_model.most_similar([vector], [], 20)
        print("サンプル " + str(i) + "-------")
        for word in words:
            # 語彙の中に含まれている単語で、名詞、動詞、形容詞のみを出力.
            if(word[0] in vocab_list.keys()):
                word_type = vocab_list[word[0]]
                if(word_type == "n" or word_type == "a" or word_type == "v"):
                    print (word_type + " " + word[0] + ":" + str(word[1]))
        print("-------------------------------")

if __name__ == '__main__':
    main()
