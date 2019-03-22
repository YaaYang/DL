# -*- encoding:utf-8 -*-

"""
@software: PyCharm
@file: DataProvider
@time: 20:45
@author: yaa
"""

import numpy as np
import cv2 as cv
from Utils import load_cifar_data


class MnistData(object):
    """Mnist Data provider implementation."""

    def __init__(self, mnist_data, z_dim, img_size):

        self._data = mnist_data  # shape is [None, 784]
        self._example_num = len(mnist_data)
        self._z_data = np.random.standard_normal((self._example_num, z_dim))
        self._indicator = 0  #

        self._random_shuffled()
        self._reshape_img(img_size)

    def _random_shuffled(self):
        p = np.random.permutation(self._example_num)
        self._data = self._data[p]
        self._z_data = self._z_data[p]

    def _reshape_img(self, img_size):
        """Transfer data shape to [img_size_list[0], img_size_list[1]] by cv resize"""
        data = np.reshape(self._data, [-1, 28, 28])
        data = data * 255  # the data set already normalization this op is inverse normalization.
        new_data_list = []
        for i in range(len(data)):
            new_data = cv.resize(data[i], img_size)
            new_data_list.append(new_data)
        # range: [-1, 1]
        new_data = np.asarray(new_data_list, dtype=np.uint8).reshape([-1, img_size[0], img_size[1], 1]) / 127.5 - 1
        self._data = new_data
        # from matplotlib import pyplot as plt
        # plt.imshow(new_data[55].reshape((32, 32)))
        # plt.show()

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffled()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > self._example_num:
            raise Exception("Don't have more samples")

        batch_data = self._data[self._indicator: end_indicator]
        batch_z_data = self._z_data[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_z_data


class Cifar10Provider(object):
    """Cifar10 Data provider implementation."""

    def __init__(self, filenames, z_dim, img_size):

        all_features = []

        for filename in filenames:
            features = load_cifar_data(filename)
            all_features.append(features)

        # the shape is [m, 3072]
        self._data = np.vstack(all_features)
        self._example_num = len(self._data)
        # Generate random vector and shape is [m, z_dim]
        self._z_data = np.random.standard_normal((self._example_num, z_dim))
        self._indicator = 0  #

        self._random_shuffled()
        self._reshape_img(img_size)

    def _random_shuffled(self):
        p = np.random.permutation(self._example_num)
        self._data = self._data[p]
        self._z_data = self._z_data[p]

    def _reshape_img(self, img_size):
        """Transfer data shape to [img_size_list[0], img_size_list[1]] by cv resize"""
        # reshape data from [m, 3072] to [m, 32, 32, 3]
        data = np.transpose(np.reshape(self._data, [-1, 3, 32, 32]), [0, 2, 3, 1])
        new_data_list = []
        for i in range(len(data)):
            new_data = cv.resize(data[i], img_size[:-1])
            new_data_list.append(new_data)
        # range: [-1, 1]  because already normalization.
        new_data = np.asarray(new_data_list, dtype=np.uint8).reshape([-1, img_size[0], img_size[1], 3]) / 127.5 - 1
        self._data = new_data

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffled()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > self._example_num:
            raise Exception("Don't have more samples")

        batch_data = self._data[self._indicator: end_indicator]
        batch_z_data = self._z_data[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_z_data




