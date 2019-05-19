#!=usr/bin/python
#-*- coding: utf-8 -*-
#推定意味表象行列と正解意味表象行列とのcos類似度を計算
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sys



def main():

	args = sys.argv
	sub = args[1]
	threshold = args[2]
	dimention = int(args[3])
	shift = int(args[4])
	sample = int(args[5])

	# 正解意味表象行列の読み込み
	Correct_SRM = np.load("../data/srm/VB_srm300.pickle")

	Estimate_SRM = np.load("../data/Test/VB/ESRM_pred" + threshold + "_basis" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")

	Estimate_SRM = Estimate_SRM.T

	answer_list = []

	# テストのサンプル数とサンプルを何個ずらすかをtime lagを何秒考慮したかで決める
	if sub == 'DK':
		test_sample_num = int(600 - shift)
		shift_num = int(shift)
	else:
		test_sample_num = int(300 - shift/2)
		shift_num = int(shift/2)

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
	with open("../data/cos_sim/cos_sim" + sub + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle", "wb") as f:
		pickle.dump(answer_list,f)

	# 0~29サンプルまでをプロット
	#plt.plot(answer_list[:30])
	#plt.show()
	#plt.savefig("../data/cossim_0-30.png")


if __name__ == '__main__':
	main()
