# -*- encoding:utf-8 -*-


"""
    1. Data provider
        1. image read
    2. build network
        1. generator
        2. discriminator
        3. DCGAN
            1. real image
            2. fake image
    3. train 
"""

import os
from tensorflow import logging
from tensorflow import gfile
from tensorflow.examples.tutorials.mnist import input_data
#
from Utils import *
from DataProvider import *

logging.set_verbosity(logging.INFO)

DATA_DIR = r'D:\Yaa\AI\DL\NN\datasets\mnist'

OUTPUT_DIR = './output_dir'
saver_path = "./ckpt"

if not gfile.Exists(saver_path):
    gfile.MakeDirs(saver_path)

if not gfile.Exists(OUTPUT_DIR):
    gfile.MakeDirs(OUTPUT_DIR)

IMAGE_CHANNEL = 1

mnist_data = input_data.read_data_sets(DATA_DIR)


def get_default_params():
    return tf.contrib.training.HParams(
        z_dim=100,  # random vector length
        init_conv_size=4,  # the height and width of image passed first transpose convolution
        g_channels=[64, 32, 16, IMAGE_CHANNEL],  #
        d_channels=[32, 64, 128, 256],
        img_size=(32, 32),  # need 4 transpose convolution
        conv_kernel_size=(5, 5),
        conv_strides=(2, 2),
        batch_size=128,
        learning_rate=0.001,
        beta1=0.5,
    )


hps = get_default_params()
dc_gan = DCGAN(hps)
mnist_provider = MnistData(mnist_data.train.images, hps.z_dim, hps.img_size)
fake_vecs, real_imgs, fake_imgs, losses = dc_gan.build()
train_op = dc_gan.build_train_op(losses)
init_op = tf.global_variables_initializer()
train_steps = int(3e3)

saver = tf.train.Saver(max_to_keep=3)

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(train_steps):
        save_state = not (step + 1) % 50
        save_model_state = not (step + 1) % 1000
        fetch = [losses["g_loss"], losses["d_loss"], train_op]

        if save_state:
            fetch.append(fake_imgs)
        batch_imgs, batch_fake = mnist_provider.next_batch(hps.batch_size)
        output = sess.run(fetch, feed_dict={fake_vecs: batch_fake, real_imgs: batch_imgs})
        g_loss_val, d_loss_val = output[:2]

        if save_state:
            fake_imgs_val = output[-1]

            fake_big_img = combine_imgs(fake_imgs_val, hps.img_size)
            real_big_img = combine_imgs(batch_imgs, hps.img_size)

            cv.imwrite(os.path.join(OUTPUT_DIR, "%05d-gen.png" % (step + 1)), fake_big_img)
            cv.imwrite(os.path.join(OUTPUT_DIR, "%05d-real.png" % (step + 1)), real_big_img)

            logging.info("[Step]:%d, g_loss:%4.5f, d_loss:%4.5f" % (step + 1, g_loss_val, d_loss_val))

        if save_model_state:
            saver.save(sess, os.path.join(saver_path, "saved_model"), global_step=step + 1)
            logging.info("[Step]:%d model already saved" % (step + 1))
