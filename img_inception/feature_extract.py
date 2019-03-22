# -*- encoding:utf-8 -*-

"""
@software: PyCharm
@file: feature_extract
@time: 2018/12/05 14:10
@author: yaa
"""

import os
import math
import pickle
import numpy as np
import tensorflow as tf
import pprint
from tensorflow import logging
from tensorflow import gfile

logging.set_verbosity(logging.INFO)

BASE_DIR = r"D:\Yaa\AI\DL\NN\datasets\image-to-text"
inception_v3_model_file = os.path.join(BASE_DIR, r"checkpoint_inception_v3\inception_v3_graph_def.pb")
input_description_file = os.path.join(BASE_DIR, r'results_20130124.token')
output_folder = os.path.join(BASE_DIR, r'image_features')

if not gfile.Exists(output_folder):
    gfile.MakeDirs(output_folder)


def extract_img_description_from_token(token_file):
    """extract image name and image description."""
    with gfile.GFile(token_file, "r") as fr:
        lines = fr.readlines()

    img2description_dict = {}
    for line in lines:
        img_intro, description = line.strip("\n").split("\t")
        img_filename, img_filename_id = img_intro.split("#")

        img2description_dict.setdefault(img_filename, [])
        img2description_dict[img_filename].append(description)

    logging.info("num of image is %d" % len(img2description_dict))
    pprint.pprint([img2description_dict.keys()][:10])
    pprint.pprint(img2description_dict[list(img2description_dict.keys())[0]])
    return img2description_dict


img2description_dict = extract_img_description_from_token(input_description_file)
img_names = img2description_dict.keys()


def add_default_graph_from_pb_file(model_file):
    """Add model graph to tensor-flow's default graph."""
    assert "pb" in model_file, "model file type must is pb file"
    with gfile.FastGFile(model_file, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        _ = tf.import_graph_def(graph_def, name="")


add_default_graph_from_pb_file(inception_v3_model_file)

batch_size = 1000
num_batchs = math.ceil(len(img_names) / batch_size)

with tf.Session() as sess:
    """Extract image feature from feature map by this layer."""
    second_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")
    for i in range(num_batchs):
        batch_img_names = list(img_names)[i * batch_size: (i + 1) * batch_size]
        batch_feature = []
        for img_name in batch_img_names:
            img_path = os.path.join(BASE_DIR, img_name)
            if not gfile.Exists(img_path):
                logging.info("%s file is not exists" % img_path)
                continue
            img_data = gfile.FastGFile(img_path, "rb").read()
            img_feature = sess.run(second_to_last_tensor, feed_dict={"DecodeJpeg/contents:0": img_data})
            print(type(img_feature))
            batch_feature.append(img_feature)
        if [] == batch_feature:
            continue
        batch_features = np.vstack(batch_feature)
        output_file = os.path.join(output_folder, "image_feature-%d.pickle" % i)
        logging.info("write to %s" % output_file)
        with gfile.GFile(output_file, "w") as fw:
            # shape is [m,] [m, n]
            pickle.dump((batch_img_names, batch_feature), fw)
