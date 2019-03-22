# -*- encoding:utf-8 -*-

"""
    1. Data generate
        a. Load Vocab
        b. Load image feature
        c. proving data for train
    2. Build image caption model
    3. Train the model
"""

"""
@software: PyCharm
@file: image_caption_train
@time: 19:39
@author: yaa
"""
import numpy as np
import os
import pickle
import tensorflow as tf
import random
from tensorflow import logging
from tensorflow import gfile
from image_to_text_utils_v3 import *
from pprint import pprint

logging.set_verbosity(logging.INFO)


def get_default_params():
    return tf.contrib.training.HParams(
        num_word_thresholds=3,
        num_embedding_nodes=32,
        num_timesteps=10,
        num_lstm_nodes=[64, 64],
        num_fc_nodes=32,
        cell_type="lstm",
        clip_gradient_norm=1,
        learning_rate=0.01,
        batch_size=50,
        keep_prob=0.8,
        log_frequent=100,
        save_frequent=1000,
    )


BASE_DIR = r"D:\Yaa\AI\DL\NN\datasets\image-to-text"

input_description_file = os.path.join(BASE_DIR, "results_20130124.token")
vocab_file = os.path.join(BASE_DIR, "vocab.txt")
output_pickle_folder = r"D:\Yaa\AI\DL\NN\datasets\image_caption_data\feature_extraction_inception_v3"
saver_folder = "./ckpt"

if not gfile.Exists(saver_folder):
    gfile.MakeDirs(saver_folder)

hps = get_default_params()
vocab = Vocab(vocab_file, hps.num_word_thresholds)
vocab_size = vocab.size()

logging.info("vocab size is %d" % vocab_size)

img2token = extract_img_description_from_token(input_description_file)
img2token_ids = conver_token_to_id(img2token, vocab)

# logging.info("num of all images: %d" % len(img2token))
# pprint(img2token['2778832101.jpg'])
# logging.info("num of all images: %d" % len(img2token_ids))
# pprint(img2token_ids['2778832101.jpg'])

# pprint(list(img2token_ids.keys()))

image_caption_data = ImageCaptionData(img2token_ids, output_pickle_folder, hps.num_timesteps, vocab)
image_feature_size = image_caption_data.size()
image_feature_data_size = image_caption_data.img_feature_size()
logging.info("image_feature_size is %d", image_feature_size)
logging.info("image feature_data_size is %d", image_feature_data_size)

batch_img_filenames, batch_img_feature, batch_img_feature_ids, batch_img_feature_weight \
    = image_caption_data.next_batch(2)

# pprint(batch_img_filenames)
# pprint(batch_img_feature)
# pprint(batch_img_feature_ids)
# pprint(batch_img_feature_weight)

from image_to_text_utils_v2 import create_model

placeholders, metrics, bp = create_model(hps, vocab_size, image_feature_data_size)

loss, accuracy, prediction, prediction_word = metrics
train_op, global_steps = bp

# summary all need show variable.
summary_op = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=3)

global_variables_init = tf.global_variables_initializer()
train_steps = int(1e4)

with tf.Session() as sess:
    sess.run(global_variables_init)
    file_writer = tf.summary.FileWriter(saver_folder)
    img_name_fixed_list = [list(img2token_ids.keys())[random.randrange(len(img2token_ids.keys()))]
                           for i in range(hps.batch_size - 1)]

    img_name_fixed_list.append("1363924449.jpg")

    for i in range(train_steps):
        batch_data = image_caption_data.next_batch(hps.batch_size)

        image_filenames, image_feature_data, image_feature_sentences_ids, image_feature_sentences_mask = batch_data

        feed_value = [image_feature_data, image_feature_sentences_ids, image_feature_sentences_mask, hps.keep_prob]
        feed_dict = dict(zip(placeholders, feed_value))

        log_state = not (i + 1) % hps.log_frequent
        save_state = not (i + 1) % hps.save_frequent

        run_list = [loss, accuracy, global_steps, prediction, prediction_word, train_op]

        if log_state:
            # train summary variable operation.
            run_list.append(summary_op)

        outputs = sess.run(run_list, feed_dict=feed_dict)

        loss_val, accuracy_val, global_steps_val, prediction_val, prediction_word_val = outputs[:5]

        if log_state:
            summary_res = outputs[-1]
            file_writer.add_summary(summary_res, global_steps_val)
            logging.info("[Step] %5d, loss_val:%4.5f, accuracy_val:%4.5f" % (global_steps_val, loss_val, accuracy_val))

            img_name, img_feature_data, img_sentence_ids, img_sentence_mask \
                = image_caption_data.get_many_by_image_name_list(img_name_fixed_list)
            feed_value = [img_feature_data, img_sentence_ids, img_sentence_mask, hps.keep_prob]
            feed_dict = dict(zip(placeholders, feed_value))
            prediction_word_val = sess.run(prediction_word, feed_dict=feed_dict)

            print(img_name_fixed_list[-1])
            print(vocab.decode(prediction_word_val[-1]))

        if save_state:
            try:
                saver.save(sess, os.path.join(saver_folder, "image_feature"), global_step=global_steps_val)
            except Exception:
                continue
            logging.info("[Step] %5d model saved" % global_steps_val)

    """
        test model flow:
            1. get feature of a image
            2. input the feature
            3. get prediction word
            4. print the text of the image
    """
