# -*- encoding:utf-8 -*-

"""
@software: PyCharm
@file: generate-vocab
@time: 2018/12/05 10:19
@author: yaa
"""

import sys
import os
import pprint

BASE_DIR = r"D:\Yaa\AI\DL\NN\datasets\image-to-text"
input_description_file = os.path.join(BASE_DIR, 'results_20130124.token')
output_vocab_file = os.path.join(BASE_DIR, 'vocab.txt')


def generate_vocab(input_description_file, verbose=True):
    """Generate vocab from input_description file's description message and return word length."""
    with open(input_description_file, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
    max_length = 0
    length_dict = {}
    vocab_dict = {}
    for line in lines:
        image_id, image_description = line.strip("\n").split("\t")
        words = image_description.strip(" ").split()  # default split by ' '
        words_length = len(words)
        length_dict[words_length] = length_dict.setdefault(words_length, 0) + 1
        max_length = max(max_length, words_length)

        for word in words:
            vocab_dict[word] = vocab_dict.setdefault(word, 0) + 1

    if verbose:
        print(max_length)
        pprint.pprint(length_dict)

    return vocab_dict


def save_vocal_file(sorted_vocab_dict, output_vocab_file, verbose=True):
    """save file to output_vocab_file from vocab_dict"""
    with open(output_vocab_file, "w", encoding='utf-8') as fw:
        fw.write("<UNK>\t1000000\n")  # un know word
        for item in sorted_vocab_dict:
            fw.write("%s\t%d\n" % (item[0], item[1]))

    if verbose:
        print("Write file already successful..")


vocab_dict = generate_vocab(input_description_file)
sorted_vocab_dict = sorted(vocab_dict.items(), key=lambda d: d[1], reverse=True)
save_vocal_file(sorted_vocab_dict, output_vocab_file)



