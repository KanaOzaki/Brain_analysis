#!=usr/bin/python
#-*- coding: utf-8 -*-
#get_high_cossim_base.py

##このプログラムは、
## 1.サンプルごとのcos類似度のリストから、
## cos類似度が高かった上位10サンプルの番号とcos類似度を表示する。(time_lagを考慮)
## 2.そのサンプルを復元するのに多く使われている基底はきっと良い基底のはずという仮定のもと、
## もっとも多く使われている基底の番号をpickleファイルに出力する.(10個になる)
## コマンド引数
##1.被験者名 2.予測精度の閾値 3.基底数 4.time lag 5.間引き数(何sampleに1枚抜くか)


import numpy as np
import sys
import pickle

def main():

    # コマンド引数
    args = sys.argv
    sub = args[1]
    threshold = args[2]
    dimention = int(args[3])
    shift = int(args[4])
    sample = int(args[5])

    if sub == 'DK':
        time_lag = int(shift)
    else:
        time_lag = int(shift/2)

    ##1.cos類似度の高かったサンプルを上位10個得る
    # cos類似度のリスト読み込み
    cos_sim_list = np.load("../data/cos_sim/cos_sim" + sub + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")

    # 降順にソート
    sorted_sample_num = np.argsort(cos_sim_list)[::-1]
    sorted_sample_cos = np.sort(cos_sim_list)[::-1]

    # 時間差のズレを考慮して出力
    for i in range(10):
        print("サンプル : {0},  cos類似度 : {1}".format(sorted_sample_num[i]+time_lag, sorted_sample_cos[i]))

    ##2.1で求めたサンプルの復元にもっとも寄与している基底をそれぞれ求める
    # テスト係数の読み込み
    Coef_test = np.load("../data/Test/VB/TestCoef_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    Coef_test = Coef_test.T

    # key:サンプル番号、value:そのサンプルを復元するのにもっとも使われている基底と係数の大きさのペア  という辞書を作成する.
    sample_base_dict = {}
    for sample_num in sorted_sample_num[:10]:
        # sample_num番目の係数を取り出す
        coef_per_sample = Coef_test[:,sample_num]
        # もっとも大きく寄与した基底番号と係数の大きさを求める
        base_num = np.argmax(coef_per_sample)
        coef = np.max(coef_per_sample)
        sample_base_dict[sample_num] = (base_num, coef)

    # 保存
    with open("../data/base/base_" + sub + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle", 'wb') as f:
        pickle.dump(sample_base_dict, f)

if __name__ == '__main__':
    main()
