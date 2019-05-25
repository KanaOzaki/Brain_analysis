#!=usr/bin/python
#-*- coding: utf-8 -*-
#eval_ESRM.py

##このプログラムは、推定意味表象行列と正解意味表象行列とのcos類似度を計算し、
##マクロ平均を出す.
##この時、time lagを考慮して正解と比較する.
##また、すべてのサンプルにおけるcos類似度をpickleで保存しておく.
##コマンド入力
##1.VB/TV 2.被験者名 3.予測精度の閾値 4.基底数 5.time lag 6.間引き数(何sampleに1枚抜くか)

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sys


def main():

	args = sys.argv
	target = args[1]
	sub = args[2]
	threshold = args[3]
	dimention = int(args[4])
	shift = int(args[5])
	sample = int(args[6])

	# 正解意味表象行列の読み込み
	Correct_SRM = np.load("../data/srm/" + target + "_srm300_test.pickle")

	# 推定意味表象行列の読み込み
	Estimate_SRM = np.load("../data/Test/" + target +  "/ESRM_" + sub + "_pred" + threshold + "_base" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
	Estimate_SRM = Estimate_SRM.T

	answer_list = []

	# テストのサンプル数とサンプルを何個ずらすかをtime lagを何秒考慮したかで決める
	if (target == 'TV') or (sub == 'DK'): #TVまたはVBのDKだったら600サンプル
		test_sample_num = int(600 - shift)
		shift_num = int(shift)
	else:
		test_sample_num = int(300 - shift/2) #VBのSTとSNは2秒に1サンプルなので300
		shift_num = int(shift/2)

	# time lagを考慮して正解とのcos類似度をサンプルごとに求めてリストに
	for i in range(0,test_sample_num):
		estimate = Estimate_SRM[i+shift_num]
		correct = Correct_SRM[i]
		answer = np.dot(correct, estimate)/(np.linalg.norm(correct) * np.linalg.norm(estimate))
		answer_list.append(answer)

	# マクロ平均を出す
	average = sum(answer_list)/len(answer_list)
	print ("マクロ平均")
	print (average)

	# サンプルごとのcos類似度保存
	with open("../data/cos_sim/" + target +  "/cos_sim" + sub + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle", "wb") as f:
		pickle.dump(answer_list,f)


if __name__ == '__main__':
	main()
