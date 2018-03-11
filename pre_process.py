#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2018.01.03
Finished on 
@author: Wang Yuntao
"""

import numpy as np


"""
QMDCT系数矩阵预处理:
    cut(matrix, height_start_idx=0, width_start_idx=0, height_end_idx=-1, width_end_idx=-1)     取子矩阵
    diff(matrix, direction, order)                                                              差分处理
    truncate(matrix, threshold)                                                                 截断处理
    downsampling(matrix, stride)                                                                下采样(200 x 576 -> 50 x 144 x 4)
"""


def cut(matrix, height_start_idx=0, width_start_idx=0, height_end_idx=-1, width_end_idx=-1):
    return matrix[height_start_idx:height_end_idx, width_start_idx:width_end_idx]


def truncate(matrix, threshold, threshold_left=None, threshold_right=None):
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
        matrix[matrix < threshold] = -threshold

    return matrix


def downsampling(matrix, stride):
    shape = np.shape(matrix)
    matrix = np.reshape(matrix, [shape[0], shape[1]])
    height = np.shape(matrix)[0]
    width = np.shape(matrix)[1]
    height_new = height // stride
    width_new = width // stride
    depth = stride ** 2

    output = np.zeros([height_new, width_new, depth], dtype=float)

    i = 0
    while i < depth:
        row = i // stride
        col = i % stride
        temp = matrix[row:height:stride, col:width:stride]
        output[:, :, i] = temp
        i += 1

    return output
