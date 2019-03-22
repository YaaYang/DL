# -*- encoding:utf-8 -*-

import tensorflow as tf
import math
import pickle
import numpy as np
import os
from tensorflow import logging
from tensorflow import gfile

"""
    Main task: 
@software: PyCharm
@file: image_to_text_utils
@time: 18:57
@author: yaa
"""


class Vocab(object):
    """Encode and Decode implementation by vocab file."""

    def __init__(self, filename, word_thresholds):
        self._eos = -1
        self._unk = -1
        self._word2id = {}
        self._id2word = {}
        self._word_thresholds = word_thresholds
        self._data_dict(filename)

    def _data_dict(self, filename):
        assert gfile.Exists(filename), "FileName is not Exists."
        with gfile.GFile(filename) as fr:
            lines = fr.readlines()
        for line in lines:
            word, frequency = line.strip('\n').split("\t")
            frequency = int(frequency)
            if word != "<UNK>" and frequency < self._word_thresholds:
                continue
            idx = len(self._word2id)
            if word == "<UNK>":
                self._unk = idx
            elif word == '.':
                self._eos = idx
            self._word2id[word] = idx
            self._id2word[idx] = word

    @property
    def unk(self):
        return self._unk

    @property
    def eos(self):
        return self._eos

    def size(self):
        return len(self._word2id)

    def _word2id_single(self, word):
        return self._word2id.get(word, self._unk)

    def _id2word_single(self, id):
        return self._id2word.get(id, "<UNK>")

    def encode(self, sentence):
        return [self._word2id_single(word) for word in sentence.split()]

    def decode(self, sentence_id):
        return " ".join([self._id2word_single(id) for id in sentence_id])


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

    return img2description_dict


def conver_token_to_id(img2description_dict, vocab):
    """return {<image_filename>: [[], [], []],...}."""
    assert isinstance(vocab, Vocab), "vocab type must is Vocab."
    img2token_ids = {}

    for img_name in img2description_dict:
        for description in img2description_dict[img_name]:
            img2token_ids.setdefault(img_name, [])
            # [[sentence_id1], [sentence_id2], [sentence_id3], ....]
            img2token_ids[img_name].append(vocab.encode(description))

    return img2token_ids


class ImageCaptionData(object):
    """
        Main task:
            input: image
            output: text

            functions:
                data_dict:
                    1. read pickle file by img_feature_dir and this is a directory.
                    2. return (image_file_names, feature_file)
                img_description:
                    1. transfer value data
                        if length > num_timesteps: split length to timesteps
                        else: fill padding
                    2. return image_feature_value and image_weight
                next_batch:
                    1. batch data from all data and batch size is a attribute.
                    2. return (image_file_names, image_feature_data).
    """

    def __init__(self, img2token_ids, img_feature_dir, num_timesteps, vocab, is_shuffle=True):
        """

            :param img2token_ids: {<image_name>: [[first sentence id], [second sentence id]]}
        """
        assert isinstance(vocab, Vocab), " attribute vocab type must is Vocab."
        self._num_timesteps = num_timesteps
        self._img2token_ids = img2token_ids
        self._is_shuffle = is_shuffle
        self._img_feature_filenames = []
        self._img_feature_data = []
        self._indicator = 0
        self._vocab = vocab

        self._img2feature_dict = {}

        self._data_dict(img_feature_dir)

        if is_shuffle:
            self._random_shuffle()

    def _data_dict(self, img_feature_dir):
        for file in gfile.ListDirectory(img_feature_dir):
            with open(os.path.join(img_feature_dir, file), "rb") as fr:
                """
                    img_feature_name is a list and the length is <feature_extract.py>--batch_size and this is 1000 so 
                        summary: the type is list and the length is [1000]
                    img_feature_data is a ndarray and the shape is [batch_size, 1, 1, 2048] 
                        because when we saving pickle model the image feature is [1, 1, 2048]
                    summary:
                        img_feature_name:
                            shape: [batch_size, ]
                            type: list
                        img_feature_data:
                            shape: [batch_size, 1, 1, 2048] each image shape is [1, 1, 2048]
                            type: ndarray

                """
                img_feature_name, img_feature_data = pickle.load(fr, encoding="latin")

                for i in range(len(img_feature_name)):
                    img_feature_single_data = img_feature_data[i]
                    img_feature_data_shape = img_feature_single_data.shape
                    self._img2feature_dict[img_feature_name[i]] = \
                        np.reshape(img_feature_single_data, [img_feature_data_shape[0], img_feature_data_shape[-1]])

                self._img_feature_filenames += img_feature_name
                self._img_feature_data.append(img_feature_data)

        self._img_feature_filenames = np.asarray(self._img_feature_filenames)

        self._img_feature_data = np.vstack(self._img_feature_data)  # shape is [batch_size, 1, 1, 2048]

        feature_data_shape = self._img_feature_data.shape
        # shape transfer [batch_size, 2048]
        self._img_feature_data = np.reshape(self._img_feature_data, [feature_data_shape[0], feature_data_shape[-1]])

        print(self._img_feature_filenames.shape)
        print(self._img_feature_data.shape)

    def _random_shuffle(self):
        """Random change data."""
        p = np.random.permutation(self.size())
        self._img_feature_filenames = self._img_feature_filenames[p]
        self._img_feature_data = self._img_feature_data[p]

    def size(self):
        return len(self._img_feature_filenames)

    def img_feature_size(self):
        assert isinstance(self._img_feature_data, np.ndarray), "Must run __init__ method."
        return self._img_feature_data.shape[1]

    def _img_description(self, batch_img_feature_filenames):
        """
            input: [filename1, filename2,...]

        :param batch_img_feature_filenames: type is iterable and value is filename
        :return: ([sentence_id], [sentence_weight])
        """
        img_sentence_ids = []
        img_sentence_weight = []
        for filename in batch_img_feature_filenames:
            # 1. result is [[sentence_id1], [sentence_id2], ...]
            img_token_ids = self._img2token_ids[filename]
            # 2. get the first sentence id and the type is list
            img_token_id = img_token_ids[0]
            img_token_id_len = len(img_token_id)
            img_token_weight = [1 for i in range(img_token_id_len)]

            if img_token_id_len >= self._num_timesteps:
                img_token_id = img_token_id[:self._num_timesteps]
                img_token_weight = img_token_weight[:self._num_timesteps]
            else:
                remain_length = self._num_timesteps - img_token_id_len
                img_token_id += [self._vocab.eos for i in range(remain_length)]
                img_token_weight += [0 for i in range(remain_length)]

            assert len(img_token_weight) == len(img_token_id)
            img_sentence_ids.append(img_token_id)
            img_sentence_weight.append(img_token_weight)

        img_sentence_weight = np.asarray(img_sentence_weight)
        img_sentence_ids = np.asarray(img_sentence_ids)
        return img_sentence_ids, img_sentence_weight

    def next_batch(self, batch_size):
        """
            Get data.
        :param batch_size: data size the type is int
        :return: batch_img_feature_filenames, batch_img_feature_data, batch_img_sentence_ids, batch_img_sentence_weight
        """
        assert isinstance(batch_size, int), "batch_size type must is int."

        end_indicator = self._indicator + batch_size
        if end_indicator > self.size():
            if self._is_shuffle:
                self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size
        assert end_indicator <= self.size(), "Don't have more data."

        batch_img_feature_filenames = self._img_feature_filenames[self._indicator: end_indicator]
        batch_img_feature_data = self._img_feature_data[self._indicator: end_indicator]
        batch_img_sentence_ids, batch_img_sentence_mask = self._img_description(batch_img_feature_filenames)
        self._indicator = end_indicator

        return batch_img_feature_filenames, batch_img_feature_data, batch_img_sentence_ids, batch_img_sentence_mask

    def _get_one_by_image_name(self, img_name):
        """Get feature by <img_name> the type is a str"""
        img_sentence_ids, img_sentence_mask = self._img_description([img_name])
        img_feature_data = self._img2feature_dict[img_name]
        return img_name, img_feature_data, img_sentence_ids, img_sentence_mask

    def get_many_by_image_name_list(self, img_name_fixed_list):
        """<img_name_fixed_list> type is a can iterable object."""
        img_name_list = []
        img_feature_data_list = []
        img_sentence_ids_list = []
        img_sentence_mask_list = []
        for img_name_single in img_name_fixed_list:
            img_name, img_feature_data, img_sentence_ids, img_sentence_mask = \
                self._get_one_by_image_name(img_name_single)
            img_name_list.append(img_name)
            img_feature_data_list.append(img_feature_data)
            img_sentence_ids_list.append(img_sentence_ids)
            img_sentence_mask_list.append(img_sentence_mask)
        img_name_nd = np.vstack(img_name_list)
        img_feature_data_nd = np.vstack(img_feature_data_list)
        img_sentence_ids_nd = np.vstack(img_sentence_ids_list)
        img_sentence_mask_nd = np.vstack(img_sentence_mask_list)

        return img_name_nd, img_feature_data_nd, img_sentence_ids_nd, img_sentence_mask_nd


def _rnn_util(hidden_dim, cell_type='lstm'):
    if cell_type == "lstm":
        return tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True)
    elif cell_type == 'gru':
        return tf.nn.rnn_cell.GRUCell(hidden_dim)


def _drop_out(cell, keep_prob):
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def create_model(hps, vocab_size, img_feature_dim):
    """
    Create model (many to many):
        flow:
            1. create placeholders
            2. image input embedding and text input embedding concat transfer embed_input
            3. build network
            4. calculate metrics
            5. set up optimizer
            6. return ((placeholder), (metrics data and optimizer), (steps))
        Main task:
            sentence：(a, b, c, d, e)
            input：(img, a, b, c, d)
            image_feature：eg: (0.2, 0.3, 0.4, 0.5)
            prediction #1: image_feature -> image_embedding -> (a)
            prediction #2: (a) -> word_embedding -> (b)
            prediction #3: (b) -> word_embedding -> (c)
            ...

            True prediction:
                input: (img, a, b, c, d)
                output: (a, b, c, d, e)
                all embedding is a same embedding and the embedding initializer at code begins.
    """
    batch_size = hps.batch_size
    num_timesteps = hps.num_timesteps
    num_embedding_nodes = hps.num_embedding_nodes
    num_lstm_nodes = hps.num_lstm_nodes

    with tf.name_scope("placeholders"):
        image_feature = tf.placeholder(shape=[batch_size, img_feature_dim],
                                       dtype=tf.float32,
                                       name="image_feature")
        # [[word_id1], [word_id2], [word_id3], ....,[word_id_num_timesteps]]
        sentences = tf.placeholder(shape=[batch_size, num_timesteps],
                                   dtype=tf.int32,
                                   name="sentences")
        # 因为 有的 句子 不够 word_threshold 所以 补充 了 padding  这个 不去 计算 这个 补充 padding 对应 的 word
        sentences_mask = tf.placeholder(shape=[batch_size, num_timesteps],
                                        dtype=tf.float32,
                                        name="sentences_mask")
        # summary global step
        global_steps = tf.get_variable(name="global_steps", dtype=tf.int32, initializer=0,
                                       trainable=False)

        keep_prob = tf.placeholder(shape=[], dtype=tf.float32, name="keep_prob")

    """Sets up embedding start."""
    embed_init = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope("embedding",
                           initializer=embed_init):
        word_embed = tf.get_variable(name="word_embed",
                                     shape=[vocab_size, num_embedding_nodes])

        # shape is [batch_size, num_timesteps-1, num_embedding_nodes]
        sentences_embed = tf.nn.embedding_lookup(word_embed, sentences[:, :-1])

    image_transfer_init = tf.uniform_unit_scaling_initializer()
    with tf.variable_scope("image_transfer_init",
                           initializer=image_transfer_init):
        # shape is [batch_size, num_embedding_nodes]
        image_embed = tf.layers.dense(image_feature, num_embedding_nodes)
        # shape transfer [batch_size, 1, num_embedding_nodes] by add dimension to index = 1
        image_embed = tf.expand_dims(image_embed, axis=1)
        logging.info(sentences_embed.get_shape().as_list())
        logging.info(image_embed.get_shape().as_list())
        # input is [image_feature, sentence_embed] and shape is [batch_size, num_timesteps, num_embedding_nodes]
        embed_input = tf.concat([image_embed, sentences_embed], axis=1)

    """Sets up embedding end."""

    """Sets up network start."""
    scale = 1.0 / math.sqrt(num_embedding_nodes + num_lstm_nodes[-1]) / 3.0
    lstm_nn_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope("lstm_nn",
                           initializer=lstm_nn_init):
        cells = []
        for i in range(len(num_lstm_nodes)):
            cell = _rnn_util(num_lstm_nodes[i], hps.cell_type)
            cell = _drop_out(cell, keep_prob)
            cells.append(cell)

        lstm_nn = tf.nn.rnn_cell.MultiRNNCell(cells)

        state_init = lstm_nn.zero_state(batch_size, dtype=tf.float32)

        # run the lstm model
        # outputs shape is [batch_size, num_timesteps, num_lstm_nodes[-1]]
        outputs, outputs_state = tf.nn.dynamic_rnn(lstm_nn, embed_input, initial_state=state_init)

    fc_init = tf.uniform_unit_scaling_initializer()
    with tf.variable_scope("fc",
                           initializer=fc_init):
        # shape is [batch_size * num_timesteps, num_lstm_nodes[-1]]
        output_flatten = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[-1]])
        # two fc connect
        fc1 = tf.layers.dense(output_flatten, hps.num_fc_nodes, name="fc1")
        fc1_dropout = tf.layers.dropout(fc1, keep_prob)
        fc1_activation = tf.nn.relu(fc1_dropout)

        # shape is [batch_size * num_timesteps, vocab_size]
        logits = tf.layers.dense(fc1_activation, vocab_size, name="logits")
    """Sets up network end."""

    """Calculate metrics start."""
    with tf.name_scope("metrics"):
        """注意： 要考虑 padding 对应 word 的值"""
        # shape is [batch_size * num_timesteps]
        sentences_flatten = tf.reshape(sentences, [-1])
        sentences_mask_flatten = tf.reshape(sentences_mask, [-1])

        # predict value * weight = last predict value
        sentences_mask_sum = tf.reduce_sum(sentences_mask_flatten)

        prediction_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=sentences_flatten)

        prediction_loss_with_mask = tf.multiply(prediction_loss, sentences_mask_flatten)

        # Calculate accuracy but why do not use BLEU because can't calculate.....
        prediction = tf.argmax(logits, 1, output_type=tf.int32)
        prediction_equal = tf.equal(prediction, sentences_flatten)
        prediction_with_mask = tf.multiply(tf.cast(prediction_equal, dtype=tf.float32), sentences_mask_flatten)

        loss = tf.reduce_sum(prediction_loss_with_mask) / sentences_mask_sum
        accuracy = tf.reduce_sum(prediction_with_mask) / sentences_mask_sum

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)

    """Calculate metrics end."""

    """Back propagate start."""
    with tf.variable_scope("optimizer"):
        # Get all the trainable variables.
        tvars = tf.global_variables()

        gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), hps.clip_gradient_norm)

        optimizer = tf.train.AdamOptimizer(hps.learning_rate)

        train_op = optimizer.apply_gradients(zip(gradients, tvars), global_step=global_steps)

    """Back propagate end."""

    return ((image_feature, sentences, sentences_mask, keep_prob),
            (loss, accuracy, prediction),
            (train_op, global_steps))
