#!=usr/bin/python
#-*- coding: utf-8 -*-
#get_similarword_train_annotation.py

##このプログラムは、指定された基底に対して
##アノテーション内に含まれる単語の中で意味表象辞書ベクトルとcos類似度が高い単語を出力する.
## コマンド引数
##1.被験者名 2.予測精度の閾値 3.基底数 4.time lag 5.間引き数(何sampleに1枚抜くか)

import gensim
import numpy as np
import sys
import pickle

def cos_similar(vector1, vector2):

	# 与えられた2つのvectorのcos類似度を計算
	l = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

	return l


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

    # アノテーション内の名詞のみを取り出したデータ(key:単語, value:ベクトル の辞書)
    with open('../data/annotation/noun_in_annotation.pickle', mode = 'rb') as f:
            noun_in_annotation = pickle.load(f)

    for num in base_nums:
        # アノテーションの中から取り出した名詞とのcos類似度を計算し、ソート.
        cos_similar_dict ={}
        for word, candidate_vector in noun_in_annotation.items():
                cos_similar_dict[word] = cos_similar(SRM_dict[num], candidate_vector)
                cos_similar_sorted = sorted(cos_similar_dict.items(), key=lambda x: -x[1])

        print("基底" + str(num) + "-------")
        for top_word, top_vector in cos_similar_sorted[:5]:
                print("単語 : {0},  cos類似度 : {1}".format(top_word, top_vector))
        print("-------------------------------")

if __name__ == '__main__':
    main()
