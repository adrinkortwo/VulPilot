import re
import csv
import numpy as np
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
args = get_args_parser().parse_args(['-model_dir', 'uncased_L-2_H-768_A-12',
                                     '-pooling_strategy', 'NONE',
                                     '-max_seq_len','512'])

path1 = "D:\\extract_data1.csv"
path2 = "D:\\extract_data2.csv"
path3 = "D:\\clean_data.csv"
sort_word_list = []
word_sentence_similarity1 = []  # 存储MASK句子与原始句子相似度以及对应的关键词
word_sentence_similarity2 = []


def read_data(path, mode):
    if mode == 'word':
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.read().splitlines()
        return lines
    elif mode == 'sentence':
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def word_tokenize(mode):
    word_list = []
    if mode == 'one':
        for i in range(lens):
            word = line_word1[i].split(',')
            word_list.append(word)  # 是一个二维数组
    elif mode == 'two':
        for i in range(lens):
            word = line_word2[i].split(',')
            word_list.append(word)
    return word_list



def MDERank1(word_list, line_sentence):
    for i in range(len(line_sentence)):
        original_sentence = bc.encode([line_sentence[i]])  # 编码原始句子
        lenss = len(line_sentence[i].split())  # 每句话的长度
        len_list = len(word_list[i])  # 每句话关键词个数
        word_index = []  # 用于存储相似度
        for j in range(len_list):
            word_repalce = []
            sentence_word = line_sentence[i].split()  # 句子分词
            for k in range(lenss):  # 遍历原始句子每个单词 ，是否是关键词
                if word_list[i][j] == sentence_word[k]:
                    sentence_word[k] = 'MASK'
                word_repalce.append(sentence_word[k])
            # print(word_repalce)#所有相同关键词都被替换掉，可用于编码

            sentence = ""
            for wd in word_repalce:  # 拼接成句子
                sentence += " " + wd
            # print( sentence)#接下来编码计算相似度
            masked_sentence = bc.encode([sentence])
            text_similarity = cosine_similarity(np.array(masked_sentence), np.array(original_sentence))
            word_index.append((word_list[i][j], text_similarity))
        word_sentence_similarity1.append(word_index)


def MDERank2(word_list, line_sentence):
    for i in range(len(line_sentence)):
        original_sentence = bc.encode([line_sentence[i]])  # 编码原始句子
        lenss = len(line_sentence[i].split())  # 每句话的长度
        len_list = len(word_list[i])  # 每句话关键词个数,每组个数不一定一样
        word_index = []  # 用于存储相似度
        for j in range(len_list):
            word_repalce = []
            sentence_word = line_sentence[i].split()  # 句子分词
            two_word_list = word_list[i][j].split()  # 定位到一个关键词组
            for k in range(lenss):
                if two_word_list[0] == sentence_word[k]:
                    if k + 1 < lenss and two_word_list[1] == sentence_word[k + 1]:
                        sentence_word[k] = 'MASK'
                        sentence_word[k + 1] = 'MASK'

                    elif k + 2 < lenss and two_word_list[1] == sentence_word[k + 2]:
                        sentence_word[k] = 'MASK'
                        sentence_word[k + 2] = 'MASK'

                word_repalce.append(sentence_word[k])
                # print(word_repalce)
            sentence = ""
            for wd in word_repalce:  # 拼接成句子
                sentence += " " + wd
            # print( sentence)#接下来编码计算相似度
            masked_sentence = bc.encode([sentence])
            text_similarity = cosine_similarity(np.array(masked_sentence), np.array(original_sentence))
            word_index.append((word_list[i][j], text_similarity))
        word_sentence_similarity2.append(word_index)


if __name__ == '__main__':
    line_word1 = read_data(path1, 'word')
    line_word2 = read_data(path2, 'word')
    print("1")
    for i in range(len(line_word1)):  # 去掉右边逗号
        line_word2[i] = line_word2[i].rstrip(',')

    for i in range(len(line_word1)):  # 去掉右边逗号
        line_word1[i] = line_word1[i].rstrip(',')

    lens = len(line_word2)
    print("2")

    line_sentence = read_data(path3, 'sentence')

    word_list1 = word_tokenize('one')
    word_list2 = word_tokenize('two')
    print("3")

    bc = BertClient(check_length=False)

    print("4")

    MDERank1(word_list1, line_sentence)

    MDERank2(word_list2, line_sentence)

    print("sort start!")

    # 排序算法
    for i in range(len(line_sentence)):
        word = word_sentence_similarity1[i] + word_sentence_similarity2[i]
        sort_word = sorted(word, key=(lambda x: x[1]))
        w_v = []
        for j in range(len(word)):
            w_v.append(sort_word[j][0])
        sort_word_list.append(w_v)

    with open("D:\\mderank_data.csv", 'w', newline='') as f:
        data_writer = csv.writer(f, lineterminator='\n')
        for wd in sort_word_list:
            data_writer.writerow(wd)