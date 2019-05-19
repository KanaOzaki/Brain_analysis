#!=usr/bin/python
#-*- coding: utf-8 -*-
#ランダムなベクトル（2種類）を推定意味表象ベクトルとし、正解意味表象ベクトルとのcos類似度を計算（チャンスレベル）
import time
from numpy import *
from scipy import *
import scipy as sp
from scipy.io import loadmat
import scipy.stats as sp
from scipy.stats import pearsonr
import numpy as np
import pickle
from sklearn import linear_model
import h5py
from numpy.random import *

def Normalization(X, X_max, X_min, M, m):

    return (float((X-X_min)*(M-m))/(X_max-X_min) + m)

def main():

    # 正解意味表象行列の読み込み
    Correct_SRM = np.load("../data/srm/VB_srm300.pickle")

    # テストサンプル数
    num_sample = 300

    #次元数(300,600)の乱数ベクトルと正解意味表象行列とのcos類似度
    randomlst=[]
    for i in range(num_sample):
        random=rand(300)
        randomlst.append(random)

    ans_lst=[]
    #cossim
    ans_all=0.0
    for i in range(num_sample - 2): #268
        random = randomlst[i+2]
        correct = Correct_SRM[i]
        # ランダムベクトルの値の範囲を正解に合わせる
        M = np.amax(correct)
        m = np.amin(correct)
        X_max = np.amax(random)
        X_min = np.amin(random)
        random = np.array([Normalization(X, X_max, X_min, M, m) for X in random])
        ans = np.dot(random, correct)/(np.linalg.norm(random)*np.linalg.norm(correct))
        ans_lst.append(ans)
    ave=sum(ans_lst)/len(ans_lst)
    var=np.var(ans_lst)

    with open("../data/cos_sim/random.pickle","wb") as f:
        pickle.dump(ans_lst,f)

    print ("マクロ平均")
    print (ave)
    print ("分散")
    print (var)


if __name__ == '__main__':
    main()
