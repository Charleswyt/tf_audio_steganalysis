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


def diff(matrix, direction, order):
    """
    差分运算
    输入数据类型:
    :param matrix: 输入矩阵
    :param direction: 差分运算方向"0 - row, 1 - col"
    :param order: 阶数
    :return: matrix_dif
    """

    height = np.shape(matrix)[0]                                # 矩阵高度
    width = np.shape(matrix)[1]                                 # 矩阵宽度

    w, h = 0, 0
    if direction == "row" and order == 1:
        output = np.zeros([height-1, width], dtype=float)
        while h < height-1:
            output[h, :] = matrix[h+1, :] - matrix[h, :]
            h += 1

    elif direction == "col" and order == 1:
        output = np.zeros([height, width-1], dtype=float)
        while w < width-1:
            output[:, w] = matrix[:, w+1] - matrix[:, w]
            w += 1

    elif direction == "row" and order == 2:
        output = np.zeros([height-2, width], dtype=float)
        while h < height-2:
            output[h, :] = 2 * matrix[h+1, :] - matrix[h, :] - matrix[h+2, :]
            h += 1

    elif direction == "col" and order == 2:
        output = np.zeros([height, width-2], dtype=float)
        while w < width-2:
            output[:, w] = 2 * matrix[:, w+1] - matrix[:, w] - matrix[:, w+2]
            w += 1

    else:
        output = matrix

    return output


def truncate(matrix, threshold):
    height = np.shape(matrix)[0]
    width = np.shape(matrix)[1]

    h = 0
    while h < height:
        w = 0
        while w < width:
            if abs(matrix[h, w]) > threshold:
                matrix[h, w] = threshold

            w += 1
        h += 1

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
