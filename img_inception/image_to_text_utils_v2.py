# -*- encoding:utf-8 -*-

"""
@software: PyCharm
@file: image_to_text_utils
@time: 18:57
@author: yaa
"""
import tensorflow as tf
from tensorflow import logging
import math


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
        # shape is [bach_size, num_timesteps]
        prediction_word = tf.argmax(tf.reshape(logits, [batch_size, num_timesteps, vocab_size]), axis=2)

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
            (loss, accuracy, prediction, prediction_word),
            (train_op, global_steps))
