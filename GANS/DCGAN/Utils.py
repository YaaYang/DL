# -*- encoding:utf-8 -*-

"""

"""

import tensorflow as tf
import numpy as np
import math
import pickle


def load_cifar_data(filename):
    """ Only return features."""
    with open(filename, 'rb') as fr:
        data = pickle.load(fr, encoding='latin')
    return data['data']


def conv2d_transpose(input, output_channel, name,
                     training=True, conv_kernel_size=(5, 5), conv_strides=(2, 2), with_bn_relu=True):
    """Wrapper transpose conv2d"""
    with tf.variable_scope(name):
        output = tf.layers.conv2d_transpose(input,
                                            output_channel,
                                            conv_kernel_size,
                                            conv_strides,
                                            padding="SAME")
        """the last of G need pass <tanh>"""
        if with_bn_relu:
            output = tf.layers.batch_normalization(output, training=training)
            return tf.nn.relu(output)
        else:
            return output


def conv2d(input, output_channel, name,
           training=True, conv_kernel_size=(5, 5), conv_strides=(2, 2)):
    def _leaky_relu(x, name, leaky_rate=0.2):
        return tf.maximum(x, x * leaky_rate, name=name)

    with tf.variable_scope(name):
        conv_output = tf.layers.conv2d(input,
                                       output_channel,
                                       kernel_size=conv_kernel_size,
                                       strides=conv_strides,
                                       padding="SAME")
        bn = tf.layers.batch_normalization(conv_output, training=training)
        return _leaky_relu(bn, name="output")


class Generator(object):
    """Generator implementation."""

    def __init__(self, g_channels, init_conv_size, conv_kernel_size, conv_strides, reuse=False):
        self._g_channels = g_channels
        self._init_conv_size = init_conv_size
        self._reuse = reuse  # can share train variables
        self.variables = None
        self._conv_kernel_size = conv_kernel_size
        self._conv_strides = conv_strides

    def __call__(self, inputs, training=True):
        """
            1. inputs need transfer tensor
            2. each input pass fc layer reshape 3D
            3. pass transpose convolution as img
            4. return img
        """
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        with tf.variable_scope("generator", reuse=self._reuse):
            with tf.variable_scope("fc"):
                fc_output = tf.layers.dense(inputs, self._g_channels[0] * math.pow(self._init_conv_size, 2))

                """
                    [None, z_dim<100>] --> [None, self._init_conv_size, self._init_conv_size, self._g_channels[0]]
                    then passed transpose convolution get a generate image  
                """
                fc_output = tf.reshape(fc_output, [-1, self._init_conv_size, self._init_conv_size, self._g_channels[0]])

                fc_bn = tf.layers.batch_normalization(fc_output, training=training)
                fc_relu = tf.nn.relu(fc_bn)

            conv_transpose = fc_relu
            # with tf.variable_scope("conv2d_transpose_ops"):
            for i in range(1, len(self._g_channels)):
                conv_transpose = conv2d_transpose(conv_transpose,
                                                  self._g_channels[i],
                                                  name="generate_conv2d_tranpose_%d" % i,
                                                  conv_kernel_size=self._conv_kernel_size,
                                                  conv_strides=self._conv_strides,
                                                  with_bn_relu=bool(i % (len(self._g_channels) - 1)))

            generate_imgs = conv_transpose
            with tf.variable_scope("generate_img"):
                generate_imgs_tanh = tf.nn.tanh(generate_imgs, name="imgs")  # shape is [None,  32, 32, 1]

        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

        return generate_imgs_tanh


class Discriminator(object):
    """Wrapper Discriminator implementation."""

    def __init__(self, d_channels, conv_kernel_size, conv_strides, reuse=False):
        self._d_channels = d_channels
        self._conv_kernel_size = conv_kernel_size
        self._conv_strides = conv_strides
        self.variables = None
        self._reuse = reuse

    def __call__(self, inputs, training=True):
        """
            Main tasks:
                2 classifier inputs
        """
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        with tf.variable_scope("discriminator", reuse=self._reuse):
            # with tf.variable_scope("convolution"):
            for i in range(len(self._d_channels)):
                inputs = conv2d(inputs,
                                self._d_channels[i],
                                conv_kernel_size=self._conv_kernel_size,
                                conv_strides=self._conv_kernel_size,
                                name="discriminator_convolution_conv2d_%d" % i)
            output = inputs
            with tf.variable_scope("fc"):
                flatten = tf.layers.flatten(output)
                logits = tf.layers.dense(flatten, 2, name="logits")

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        self._reuse = True
        return logits


class DCGAN(object):
    """Wrapper DC GAN network."""

    def __init__(self, hps):
        self._z_dim = hps.z_dim
        self._g_channels = hps.g_channels
        self._d_channels = hps.d_channels
        self._img_size = hps.img_size
        self._batch_size = hps.batch_size
        self._generator = Generator(self._g_channels, hps.init_conv_size, hps.conv_kernel_size, hps.conv_strides)
        self._discriminator = Discriminator(self._d_channels, hps.conv_kernel_size, hps.conv_strides)

        self._learning_rate = hps.learning_rate
        self._beta1 = hps.beta1
        self._hps = hps

    def build(self):
        """Builds model."""
        with tf.variable_scope("placeholders"):
            real_imgs = tf.placeholder(dtype=tf.float32, name="real_img",
                                       shape=[self._batch_size, self._img_size[0],
                                              self._img_size[1], self._g_channels[-1]])
            fake_vecs = tf.placeholder(dtype=tf.float32, name="fake_vec",
                                       shape=[self._batch_size, self._z_dim])

        # with tf.variable_scope("forward_propagate"):
        fake_imgs = self._generator(fake_vecs)
        real_imgs_logits = self._discriminator(real_imgs)
        fake_imgs_logits = self._discriminator(fake_imgs)

        with tf.variable_scope("metrics"):
            losses_fake2real = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones(shape=[self._batch_size], dtype=tf.int32), logits=fake_imgs_logits))

            losses_fake2fake = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros(shape=[self._batch_size], dtype=tf.int32), logits=fake_imgs_logits))

            losses_real2real = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones(shape=[self._batch_size], dtype=tf.int32), logits=real_imgs_logits))

            tf.add_to_collection("generator_loss", losses_fake2real)
            tf.add_to_collection("discriminator_loss", losses_fake2fake)
            tf.add_to_collection("discriminator_loss", losses_real2real)

            losses = {
                "g_loss": tf.add_n(tf.get_collection("generator_loss"), name="total_generator_loss"),
                "d_loss": tf.add_n(tf.get_collection("discriminator_loss"), name="total_discriminator_loss")
            }

        return fake_vecs, real_imgs, fake_imgs, losses

    def build_train_op(self, losses):
        generator_optimizer = tf.train.AdamOptimizer(self._learning_rate, beta1=self._beta1)
        discriminator_optimizer = tf.train.AdamOptimizer(self._learning_rate, beta1=self._beta1)

        generator_train_op = generator_optimizer.minimize(losses["g_loss"],
                                                          var_list=self._generator.variables)
        discriminator_train_op = discriminator_optimizer.minimize(losses["d_loss"],
                                                                  var_list=self._discriminator.variables)

        """cross train implementation."""
        # 先 执行 control_dependencies -> <control_inputs> 然后 执行 context manager 中的 op 保证程序的流程
        # 因为 我们 需要 g 和 d 交叉 训练 所以 使用 control_dependencies
        with tf.control_dependencies([generator_train_op, discriminator_train_op]):
            return tf.no_op(name="train")  # Does nothing. Only useful as a placeholder for control edges.


def combine_imgs(batch_imgs, img_size, rows=8, cols=16):
    """8 * 16 = 128"""
    row_imgs = []
    for r in range(rows):
        col_imgs = []
        for c in range(cols):
            img = batch_imgs[r * cols + c]
            # [batch_size, 32, 32]
            img = img.reshape(img_size)
            img = (img + 1) * 127.5
            col_imgs.append(img)
        col_imgs = np.hstack(col_imgs)
        row_imgs.append(col_imgs)
    # [8 * 32, 16 * 32]
    all_img = np.vstack(row_imgs)
    all_img = np.asarray(all_img, dtype=np.uint8)
    return all_img
