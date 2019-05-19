#!=usr/bin/python
#-*- coding: utf-8 -*-
#make_SNtest.py

import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import pickle


if __name__ == '__main__':

    newSN = []

    num_list = ['04', '08', '12', '16', '20']

    for num in num_list:
        SN = loadmat('../original_data/VB/ROW/VB_SN.' + num + '.mat')
        SN = SN['resp']
        for i in range(60):
            newSN_row = (SN[i] + SN[i+60] + SN[i+60*2] + SN[i+60*3] + SN[i+60*4])/5
            newSN.append(newSN_row)

    newSN = np.array(newSN)
    print(newSN.shape)

    # mat保存
    savemat('../original_data/VB/ROW/VB_SN.Val01.mat', {'resp':newSN})
