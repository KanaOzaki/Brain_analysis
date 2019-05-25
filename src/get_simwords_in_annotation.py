#!=usr/bin/python
#-*- coding: utf-8 -*-

##このプログラムは、アノテーション内の名詞のみを取り出したデータと辞書と推定意味表象行列を読み込み、
##辞書における各意味表象基底ベクトルとcos類似度が高い上位5単語をそのリストの中から見つける.
##コマンド入力
##1.VB/TV 2.被験者名 3.予測精度の閾値 4.基底数 5.time lag 6.間引き数(何sampleに1枚抜くか) 7. train/test

import pickle
import sys
import numpy as np

def cos_similar(vector1, vector2):

	# 与えられた2つのvectorのcos類似度を計算
	l = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

	return l

def main():

	args = sys.argv
	target = args[1]
	sub = args[2]
	threshold = args[3]
	dimention = int(args[4])
	shift = int(args[5])
	sample = int(args[6])
	phase = args[7]

	if (target == 'TV') and (sub == 'DK'):
		sample_lag = shift
	else:
		sample_lag = int(shift/2)

	# アノテーション内の名詞のみを取り出したデータ(key:単語, value:ベクトル の辞書)
	with open('../data/annotation/' + target + '/noun_in_annotation_' + target + '_' + phase + '.pickle', mode = 'rb') as f:
		noun_in_annotation = pickle.load(f)

	# 推定意味表書行列の読み込み
	Estimate_SRM = np.load("../data/Test/" + target + "/ESRM_" + sub + "_pred" + threshold + "_base" + str(dimention) + "_sec" + str(shift) + "_sample" + str(sample) + ".pickle")
	Estimate_SRM = Estimate_SRM.T

	# 推定意味表象行列から1サンプルごとに類似単語を出力する
	for i, target_vector in enumerate(Estimate_SRM):
		cos_similar_dict ={}

		# アノテーションの中から取り出した名詞とのcos類似度を計算し、ソート.
		for word, candidate_vector in noun_in_annotation.items():
			cos_similar_dict[word] = cos_similar(target_vector, candidate_vector)
		cos_similar_sorted = sorted(cos_similar_dict.items(), key=lambda x: -x[1])

		# サンプルごとに類似単語上位5つを出力
		print("サンプル " + str(i + sample_lag) + "-------")
		for top_word, top_vector in cos_similar_sorted[:5]:
			print("単語 : {0},  cos類似度 : {1}".format(top_word, top_vector))
		print("-------------------------------")

if __name__ == '__main__':
	main()
