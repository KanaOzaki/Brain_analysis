#!=usr/bin/python
#-*- coding: utf-8 -*-
#get_reconstruct_sample_train.py

##このプログラムは、指定された基底に対して
##その基底を多く使うサンプル上位5つを出力するプログラムである
## コマンド引数
##1.被験者名 2.予測精度の閾値 3.基底数 4.time lag 5.間引き数(何sampleに1枚抜くか)

import sys
import numpy as np
import gensim
import pickle

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

    # 対象基底番号の辞書を読み込み、基底番号だけのリストにする
    sample_base_dict = np.load("../data/base/" + target + "/base_" + sub + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    base_nums = [a for (a,b) in sample_base_dict.values()]

    # 訓練で作った係数の読み込み
    Coef_train = np.load("../data/Dict/" + target + "/Coef_" + sub + "_pred" + threshold + "_base" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    print(Coef_train.shape)

    # 指定された基底について係数の大きい順に5つを見ていく
    for num in base_nums:
        print("基底 " + str(num) + "-------")
        # 基底番号numの係数を見る
        coef_per_base = Coef_train[:,num]
        print(len(coef_per_base))
        # ソート
        coef_sorted_index = np.argsort(coef_per_base)[::-1]
        coef_sorted = np.sort(coef_per_base)[::-1]
        # 上位5つを見る
        for i in range(5):
            print("サンプル : {0},  係数 : {1}".format(coef_sorted_index[i] * sample, coef_sorted[i]))
        print("-------------------------------")

if __name__ == '__main__':
    main()
