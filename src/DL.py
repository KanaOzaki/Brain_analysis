#!=usr/bin/python
#-*- coding: utf-8 -*-
#DL.py

##このプログラムは、与えられた脳活動行列と意味表象行列からtimelagを考慮して結合行列を作成し、
##辞書学習を行い、辞書と係数を保存するプログラムである.
##コマンド入力
##1.被験者名 2.予測精度の閾値 3.基底数 4.time lag 5.間引き数(何sampleに1枚抜くか)

import sys
import pickle
import time
import os
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder


def main():

	start = time.time()

	args = sys.argv
	sub = args[1]
	threshold = args[2]
	dimention = int(args[3])
	shift = int(args[4])
	sample = int(args[5])

	print('{} secずらし'.format(shift))

	#脳活動データ読み込み
	with open('../data/Brain/VB/' + sub + '_train_reduced_' + threshold +'.pickle', 'rb') as f:
		brain_data = pickle.load(f)

	#意味表象データ読み込み
	with open('../data/srm/VB_srm300.pickle', 'rb') as f:
		semantic_data = pickle.load(f)

	# 秒ずらし
	if sub != 'DK':
		start = int(shift/2)
		end = int(4500-shift/2)
		brain_data = brain_data[start:]
		semantic_data = semantic_data[::2]
		semantic_data = semantic_data[0:end]
	else:
		start = int(shift)
		end = int(9000-shift)
		brain_data = brain_data[start:]
		semantic_data = semantic_data[0:end]

	print(len(semantic_data))
	print(len(brain_data))


	#2つを結合した合成行列を作成
	brainw2vdata = np.c_[brain_data, semantic_data]
	brainw2vdata = np.array(brainw2vdata)

	brainw2vdata = brainw2vdata[::sample]

	print("次元：")
	print(brainw2vdata.shape)

	#辞書学習
	dict_model = DictionaryLearning(n_components = dimention, alpha = 1.0, transform_algorithm = 'lasso_lars', transform_alpha = 1.0, fit_algorithm = 'lars', verbose = True)
	dict_model.fit(brainw2vdata)

	#辞書
	Dict = dict_model.components_
	print("辞書：")
	print(Dict.shape)

	#係数　
	coef = dict_model.transform(brainw2vdata)
	print("係数：")
	print(coef.shape)

	#辞書保存
	f = open("../data/Dict/VB/Dict_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle", "wb")
	pickle.dump(Dict, f)
	f.close()

	#係数保存
	f = open("../data/Dict/VB/Coef_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle", "wb")
	pickle.dump(coef, f)
	f.close()

	#計算時間出力
	elapsed_time = time.time() - start
	print (("elapsed_time:{0}".format(elapsed_time)) + "[sec]")

if __name__ == '__main__':
	main()
