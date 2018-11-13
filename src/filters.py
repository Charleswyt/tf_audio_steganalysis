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
    srm_kernels_np = np.load("SRM_Kernels.npy")
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
                                 shape=[kernel_size * kernel_size, kernel_size * kernel_size, 1, 1],
                                 name="dct_kernel")
    return dct_kernel
