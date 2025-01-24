from keybert import KeyBERT
from nltk import data
data.path.append(r'/root/nltk_data')

import re
import csv


def read_data(path):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def write_data(path,sentence_list):
    with open(path, 'w', newline='') as f:
        data_writer = csv.writer(f, lineterminator='\n')
        for wd in sentence_list:
            data_writer.writerow(wd)


if __name__ == '__main__':
    kw_model = KeyBERT()
    key_can_word1 = []
    key_can_word2 = []
    path1 = "D:\\clean_data.csv"

    line_sentence1 = read_data(path1)
    lens = len(line_sentence1)

    for i in range(lens):
        list1 = []
        list2 = []
        list_sentence = []
        word1 = kw_model.extract_keywords(line_sentence1[i], keyphrase_ngram_range=(1, 1), top_n=20)
        word2 = kw_model.extract_keywords(line_sentence1[i], keyphrase_ngram_range=(2, 2), top_n=20)
        for j in range(len(word1)):
            list1.append(word1[j][0])
        for k in range(len(word2)):
            list2.append(word2[k][0])
        key_can_word1.append(list1)
        key_can_word2.append(list2)

    with open("D:\\extract_data1.csv", 'w',
              newline='') as f:
        data_writer = csv.writer(f, lineterminator='\n')
        for wd in key_can_word1:
            data_writer.writerow(wd)

    with open("D:\\extract_data2.csv", 'w',
              newline='') as f:
        data_writer = csv.writer(f, lineterminator='\n')
        for wd in key_can_word2:
            data_writer.writerow(wd)




