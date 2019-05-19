#!=usr/bin/python
#-*- coding: utf-8 -*-
#DL.py

##このプログラムは、テスト用脳活動行列をスパースコーディングし、
##得られた係数に意味表象辞書をかけて推定意味表象行列を作成する.
##コマンド入力
##1.被験者名 2.予測精度の閾値 3.基底数 4.time lag 5.間引き数(何sampleに1枚抜くか)



from numpy import *
from scipy import *
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
import pickle
import time
import sys

def main():

    start = time.time()

    args = sys.argv
    sub = args[1]
    threshold = args[2]
    dimention = int(args[3])
    shift = int(args[4])
    sample = int(args[5])

    #テスト用脳活動行列の読み込み
    with open('../data/Brain/VB/' + sub + '_test_reduced_' + threshold +'.pickle', 'rb') as f:
        test_brain_data = pickle.load(f)

    print("テスト脳活動行列")
    print(test_brain_data.shape)

    # 辞書データ読み込み(学習のとき作ったやつ)
    Dict = np.load("../data/Dict/VB/Dict_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
    # 脳活動辞書の部分だけ取り出す
    print(Dict.shape[1]-300)
    Brain_dict = Dict.T[0:Dict.shape[1]-300]
    Brain_dict = Brain_dict.T
    print ("脳活動辞書")
    print (Brain_dict.shape)

    # 意味表象辞書の部分だけ取り出す
    SRM_dict = Dict.T[-300:]
    SRM_dict = SRM_dict.T
    print("意味表象辞書")
    print(SRM_dict.shape)

    #スパースコーディング
    SC_model = SparseCoder(dictionary = Brain_dict, transform_algorithm = 'lasso_lars')

    #係数
    test_coef = SC_model.transform(test_brain_data)
    print("テスト係数")
    print(test_coef.shape)

    #係数保存
    f = open("../data/Test/VB/TestCoef_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle","wb")
    pickle.dump(test_coef, f)
    f.close()

    #テスト意味表象行列の推定
    estimated_SRM = np.dot(SRM_dict.T, test_coef.T)
    print("推定意味表象行列")
    print(estimated_SRM.shape)

    #推定意味表象行列保存
    f = open("../data/Test/VB/ESRM_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle","wb")
    pickle.dump(estimated_SRM, f)
    f.close()

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



if __name__ == '__main__':
    main()
