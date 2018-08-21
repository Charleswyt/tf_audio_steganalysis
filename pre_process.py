#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.01.03
Finished on 2018.01.03
@author: Wang Yuntao
"""

import math
import numpy as np

"""
The pre-processing of the QMDCT matrix:
    truncate(matrix, threshold)                                                                 truncation of matrix
    down_sampling(matrix, mode, mode_number)                                                    downsampling in single mode
    get_down_sampling(matrix, mode_number)                                                      downsampling in 
"""


def truncate(matrix, threshold=None, threshold_left=None, threshold_right=None):
    """
    truncation (数据截断)
    :param matrix: the input matrix (numpy.ndarray)
    :param threshold: threshold
    :param threshold_left: threshold (for minimum)
    :param threshold_right: threshold (for maximum)
    :return:
    """
    if threshold_left is not None and threshold_right is not None:
        matrix[matrix > threshold_left] = threshold_left
        matrix[matrix > threshold_right] = threshold_right
    else:
        matrix[matrix > threshold] = threshold
        matrix[matrix < -threshold] = -threshold

    return matrix


def down_sampling(matrix, mode, mode_number):
    """
    the downsampling of the matrix (矩阵下采样)
    :param matrix: the input matrix
    :param mode: the current mode
    :param mode_number: the total number of the modes
    :return: down sampling matrix
    """
    stride = int(math.sqrt(mode_number))
    mask = list(range(mode_number))
    mask = np.reshape(mask, [stride, stride])
    index = np.argwhere(mask == mode)[0]
    i, j = index[0], index[1]
    output = matrix[i::stride, j::stride]

    return output


def get_down_sampling(matrix, mode_number):
    """
    the downsampling of the matrix (矩阵下采样)
    :param matrix: the input matrix
    :param mode_number: the total number of the modes
    """
    shape = np.shape(matrix)
    height, width = shape[0], shape[1]
    matrix = np.reshape(matrix, [height, width])
    stride = math.sqrt(mode_number)
    sub_height, sub_width = int(height // stride), int(width // stride)
    output = np.zeros([sub_height, sub_width, mode_number])
    for i in range(mode_number):
        output[:, :, i] = down_sampling(matrix, i, mode_number)

    return output


def preprocess(content, is_abs=False, is_diff=False, is_diff_abs=False, order=2, direction=0, is_trunc=False, threshold=15):
    if is_abs is True:
        content = abs(content)
    if is_trunc is True:
        content = truncate(content, threshold=threshold)
    if is_diff is True:
        content = np.diff(content, n=order, axis=direction)
        if is_diff_abs is True:
            content = abs(content)

    return content
