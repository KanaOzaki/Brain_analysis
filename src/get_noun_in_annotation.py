#!=usr/bin/python
#-*- coding: utf-8 -*-
#get_noun_in_annotation.py

##このプログラムは、アノテーションデータに含まれる名詞とその300次元の分散表現を辞書として作り、保存するプログラム.
##形態素解析はJanome
##stopwords.txtに含まれる単語は除く.
##コマンド入力 1. TV/VB 2. train/test

import re
import sys
import pickle
import gensim
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

def read_annotation_data(file_name, target):

	if target == 'VB':
		end = 9000
	else:
		end = 7200

	data = open(file_name)
	data = data.read()

	#文章からスペース削除
	data = data.replace(' ', '')
	line = data.split()
	line = line[:end]

	return line

def make_noun_list(sentence, word_list):

	# まずは文を携帯素解析
	t = Tokenizer()
	W2V_sum=np.zeros(300)
	noun_in_sentence =[]

	for token in t.tokenize(sentence, stream=True):
		hinshi = token.part_of_speech.split(',')[0]

		# 数字以外の名詞に限定
		if (hinshi == "名詞" and token.part_of_speech.split(',')[1] != "数"):

			if(token.base_form in word_list):
				noun_in_sentence.append(token.base_form)

	return noun_in_sentence

def main():

	# コマンド引数
	args = sys.argv
	target = args[1] # VB/TV
	phase = args[2] #train/test

	# 語彙ファイル読み込み
	file_name = '../original_data/TV/jawiki160111S1000W10SG_vocab.txt'
	word_list = make_word_list(file_name)

	# アノテーションデータ読み込み
	file_name = '../data/annotation/' + target + '/annotation_' + target + '_' + phase + '.txt'
	annotation_data = read_annotation_data(file_name, target)


	# ストップワードを読み込んでリストへ
	stopwords = []
	with open('../data/annotation/stopwords.txt', 'r') as stopwords_file:
		for line in stopwords_file:
			line = line.strip()
			stopwords.append(line)

	# アノテーションから取り出した名詞リスト作成
	target_noun_list = []
	for sentence in annotation_data:
		lst = make_noun_list(sentence, word_list)
		target_noun_list.extend(lst)

	# 違う文で同じ単語が出てきても、1つとしてしまう.
	target_noun_list = list(set(target_noun_list))
	# ストップワード除去
	target_noun_list = [word for word in target_noun_list if word not in stopwords]

	# 名詞リストの中の単語を全て分散表現にして、{名詞：ベクトル}の辞書作成
	target_noun_dict = {}
	for word in target_noun_list:
		try:
			target_noun_dict[word] = W2V_model[word]
		except:
			continue

	print(target_noun_dict)

	# 作成した辞書を保存
	with open('../data/annotation/' + target + '/noun_in_annotation_' + target + '_' + phase + '.pickle', mode = 'wb') as f:
		pickle.dump(target_noun_dict, f)

if __name__ == '__main__':
	main()
