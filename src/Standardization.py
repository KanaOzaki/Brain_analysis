#!=usr/bin/python
#-*- coding: utf-8 -*-
#Standardization.py

##このプログラムは、TrainとTestが同じ値の範囲になるように
##どちらもそれぞれ平均0、分散1に正規化する(標準化).
##いただいたtrainデータとtestデータは最初から大きく分散が違ったので
##今回はそれぞれのデータを標準化した.
##コマンド入力 : 被験者名

import sys
import numpy as np
from scipy.io import loadmat
import pickle

DATA_DIR = '../original_data'

def Standardization(X, mean, sd):

	return (float(X - mean)/sd)

def main():

	args = sys.argv
	sub = args[1]

	##1. 訓練データを標準化
	#大脳皮質のみのpickleデータを読み込み
	with open(DATA_DIR + '/VB/preprocess/' + sub + '_VB_brain_train_preprocess.pickle', 'rb') as f:
		train_data = pickle.load(f, encoding='latin1')

	if sub == 'DK':
		with open(DATA_DIR + '/VB/preprocess/' + sub + '_VB_brain_train_preprocess2.pickle', 'rb') as f:
			train_data2 = pickle.load(f, encoding='latin1')
		train_data = np.r_[train_data, train_data2]

	# 平均0、分散1に正規化
	mean = np.mean(train_data)
	sd = np.std(train_data)
	train_data = np.array([[Standardization(X, mean, sd) for X in row] for row in train_data])

	print(np.mean(train_data))
	print(np.std(train_data))
	print(train_data.shape)

	# 標準化済みtrain_data保存
	if sub == 'DK':
		train_data1 = train_data[:4500]
		train_data2 = train_data[4500:]
		with open(DATA_DIR + '/VB/' + sub + '_VB_brain_train_std1.pickle', 'wb') as f:
			pickle.dump(train_data1, f)
		with open(DATA_DIR + '/VB/' + sub + '_VB_brain_train_std2.pickle', 'wb') as f:
			pickle.dump(train_data2, f)
	else:
		with open(DATA_DIR + '/VB/' + sub + '_VB_brain_train_std.pickle', 'wb') as f:
			pickle.dump(train_data, f)

	##2. テストデータを標準化
	test_data = loadmat(DATA_DIR + '/VB/ROW/VB_' + sub + '.Val01.mat')
	test_data = test_data['resp']

	#マスキングデータで大脳皮質のみ取り出す
	vset = loadmat(DATA_DIR + '/VB/vset/vset_' + sub + '_099.mat')
	vset = vset['tvoxels']
	vset = [ flatten for inner in vset for flatten in inner ]
	test_data = test_data[:, vset]

	# 平均0、分散1に正規化
	mean = np.mean(test_data)
	sd = np.std(test_data)
	test_data = np.array([[Standardization(X, mean, sd) for X in row] for row in test_data])

	print(np.mean(test_data))
	print(np.std(test_data))
	print(test_data.shape)

	# 標準化したtest_data保存
	with open(DATA_DIR + '/VB/' + sub + '_VB_brain_test_std.pickle', 'wb') as f:
		pickle.dump(test_data, f)


if __name__ == '__main__':
	main()
