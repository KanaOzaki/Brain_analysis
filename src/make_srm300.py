#!=usr/bin/python
#-*- coding: utf-8 -*-
#make_srm300.py

# このプログラムは、アノテーションデータから意味表象行列を作成(word2vec : 300)する.
# 浅原先生がお作りになられた分散表現を使用.
# 訓練データに対して、テストデータに対してどちらも対応可能.

import re
import gensim
import pickle
import sys
import numpy as np
from janome.tokenizer import Tokenizer

w2v_filename = '../data/entity_vector/nwjc_word_skip_300_8_25_0_1e4_6_1_0_15.txt.vec'
W2V_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_filename)

def make_word_list(file_name):

	input_file = open(file_name, 'r')


	# 対象とする単語のリストを作る
	word_list = ['</s>']

	for line in input_file:
		line = line.strip()
		line = line.replace('.', '\t')
		item_list = line.split('\t')
		word_list.append(item_list[0])
	input_file.close()

	return word_list

def read_annotation_data(filename):

	with open(filename, 'r') as f:
		data = f.read().splitlines()

	return data

def make_word2vec(sentence, word_list):

	# まずは文を携帯素解析
	t = Tokenizer()
	W2V_sum=np.zeros(300)

	for token in t.tokenize(sentence, stream=True):
		hinshi = token.part_of_speech.split(',')[0]

		# 動詞、形容詞または数字以外の名詞
		if(hinshi == "動詞" or hinshi == "形容詞" or (hinshi == "名詞" and token.part_of_speech.split(',')[1] != "数")):

			# 語彙リストに含まれている単語のみ
			if(token.base_form in word_list):
				# 分散表現の辞書の中になかったら、それは無視する.
				try:
					W2V_sum += W2V_model[token.base_form]
				except:
					continue

	return (W2V_sum)

def main():

	# VB/TV
	args = sys.argv
	target = args[1]

	# 語彙ファイル読み込み
	file_name = '../original_data/NishidaVimeo/jawiki160111S1000W10SG_vocab.txt'
	word_list = make_word_list(file_name)

	# アノテーションデータ読み込み
	file_name = '../data/annotation/annotation_' + target + '.txt'
	annotation_data = read_annotation_data(file_name)


	# アノテーションデータの各サンプルについて単語の意味表象の和を求める.
	srm_all = []
	for sentence in annotation_data:
		srm = make_word2vec(sentence, word_list)
		print(srm)
		srm_all.append(srm)


	# 行列を保存
	with open('../data/srm/' + target + '_srm300.pickle', 'wb') as f:
		pickle.dump(srm_all, f)


if __name__ == '__main__':
	main()
