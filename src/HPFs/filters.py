#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2018.03.28
Finished on 2018.03.28
Modified on 2018.08.27

@author: Yuntao Wang
"""

import numpy as np
import tensorflow as tf
from text_preprocess import text_read


def kv_kernel_generator():
    """
    kv kernel for image steganalysis
    :return:
        kv kernel tensor
    """
    kv_kernel_matrix = tf.constant(value=[-1, 2, -2, 2, -1, 2, -6, 8, -6, 2, -2, 8, -12, 8, -2, 2, -6, 8, -6, 2, -1, 2, -2, 2, -1],
                                   dtype=tf.float32,
                                   shape=[5, 5, 1, 1],
                                   name="kv_kernel_matrix")
    kv_kernel = tf.multiply(1 / 12, kv_kernel_matrix, name="kv_kernel")

    return kv_kernel


def srm_kernels_generator():
    """
    SRM kernels for image steganalysis
    :return:
        SRM kernels tensor
    """
    srm_kernels_np = np.load("./SRM_Kernels.npy")
    [height, width, channel, filter_num] = np.shape(srm_kernels_np)
    srm_kernels = tf.constant(value=srm_kernels_np,
                              dtype=tf.float32,
                              shape=[height, width, channel, filter_num],
                              name="srm_kernels")

    return srm_kernels


def dct_kernel_generator(kernel_size=4):
    """
    DCT kernel for image steganalysis
    :param kernel_size: size of DCT kernel, e.g. 2, 3, 4, 5, 6, 7, 8
    :return:
        a DCT kernel tensor
    """
    kernel_sizes = [2, 3, 4, 5, 6, 7, 8]
    if kernel_size not in kernel_sizes:
        print("Wrong kernel size")
        dct_kernel_np = np.ones(kernel_size * kernel_size)
        dct_kernel = tf.constant(value=dct_kernel_np,
                                 dtype=tf.float32,
                                 shape=[kernel_size, kernel_size, 1, 1],
                                 name="all_one_kernel")
    else:
        dct_kernel_file_path = "./dct_kernels/DCT" + str(kernel_size) + ".txt"
        dct_kernel_np = text_read(dct_kernel_file_path, channel=0, separator=" ")

        dct_kernel = tf.constant(value=dct_kernel_np,
                                 dtype=tf.float32,
                                 shape=[kernel_size, kernel_size, 1, kernel_size * kernel_size],
                                 name="dct_kernel")
    return dct_kernel


def point_high_pass_kernel_generator():
    """
    point high-pass kernel for image steganalysis
    :return:
        point high-pass kernel tensor
    """
    point_high_pass_kernel = tf.constant(value=[-0, 0, 0.0199, 0, 0, 0, 0.0897, 0.1395, 0.0897, 0, -0.0199, 0.1395, -1, 0.1395, 0.0199,
                                                0, 0.0897, 0.1395, 0.0897, 0, 0, 0, 0.0199, 0, 0],
                                         dtype=tf.float32,
                                         shape=[5, 5, 1, 1],
                                         name="point_high_pass_kernel")

    return point_high_pass_kernel


def gabor_2d_horizontal_kernel_generator():
    """
    horizontal 2D Gabor Filter for image steganalysis
    :return:
        horizontal 2D Gabor Filter
    """
    gabor_2d_horizontal_kernel = tf.constant(value=[0.0562, -0.1354, 0, 0.1354, -0.0562, 0.0818, -0.1970, 0, 0.1970, -0.0818, 0.0926, -0.2233, 0, 0.2233, -0.0926,
                                                    0.0818, -0.1970, 0, 0.1970, -0.818, 0.0562, -0.1354, 0, 0.1354, -0.0562],
                                             dtype=tf.float32,
                                             shape=[5, 5, 1, 1],
                                             name="gabor_2d_horizontal_kernel")

    return gabor_2d_horizontal_kernel


def gabor_2d_vertical_kernel_generator():
    """
    vertical 2D Gabor Filter for image steganalysis
    :return:
        vertical 2D Gabor Filter
    """
    gabor_2d_vertical_kernel = tf.constant(value=[-0.0562, -0.0818, -0.0926, -0.0818, -0.0562, 0.1354, 0.1970, 0.2233, 0.1970, 0.1354, 0, 0, 0, 0, 0,
                                                  -0.1354, -0.1970, -0.2233, -0.1970, -0.1354, 0.0562, 0.0818, 0.0926, 0.0818, 0.0562],
                                           dtype=tf.float32,
                                           shape=[5, 5, 1, 1],
                                           name="gabor_2d_vertical_kernel")

    return gabor_2d_vertical_kernel
