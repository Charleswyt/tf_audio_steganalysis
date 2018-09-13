#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from pre_process import preprocess

"""
Created on 2017.11.27
Finished on 2017.11.27
Modified on 2018.08.23

@author: Wang Yuntao
"""

"""
    function:
        text_read(text_file_path, height=200, width=576, separator=",")                             read single QMDCT coefficients matrix into memory
        text_read_all(text_files_dir, height=200, width=576, separator=",")                         read all QMDCT coefficients matrix read into memory
        text_read_batch(text_files_list, height=200, width=576, separator=",")                      read QMDCT coefficients matrix in batch
"""


def text_read(text_file_path, height=200, width=576, separator=",",
              is_abs=False, is_diff=False, is_abs_diff=False, is_diff_abs=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    data read from one text file

    :param text_file_path: the file path
    :param height: the height of QMDCT matrix
    :param width: the width of QMDCT matrix
    :param separator: separator of each elements in text file
    :param is_abs: whether make abs or not
    :param is_diff: whether make difference or not
    :param is_abs_diff: whether make abs and difference or not
    :param is_diff_abs: whether make difference and abs or not
    :param order: the order of difference
    :param direction: the direction of difference, 0 - inter frame, 1 - intra frame
    :param is_trunc: whether make truncation or not
    :param threshold: the threshold of truncation

    :return
        content: QMDCT matrix  ndarray, shape: [height, width, 1]
    """
    file = open(text_file_path)
    content = []

    # read data line by line
    lines = file.readlines()
    for line in lines:
        numbers = [int(character) for character in line.split(separator)[:-1]]
        content.append(numbers)

    content = np.array(content, dtype=np.int32)

    content = preprocess(content=content, is_abs=is_abs, is_diff=is_diff, is_abs_diff=is_abs_diff, is_diff_abs=is_diff_abs,
                         order=order, direction=direction, is_trunc=is_trunc, threshold=threshold)

    # reshape
    [h, w] = np.shape(content)
    content = np.reshape(content, [h, w, 1])

    height_new = None if h < height else height
    width_new = None if w < width else width
    content = content[:height_new, :width_new, :]

    return content


def text_read_batch(text_files_list, height=200, width=576, separator=",",
                    is_abs=False, is_diff=False, is_abs_diff=False, is_diff_abs=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    read all txt files into the memory

    :param text_files_list: text files list
    :param height: the height of QMDCT matrix
    :param width: the width of QMDCT matrix
    :param separator: separator of each elements in text file
    :param is_abs: whether make abs or not
    :param is_diff: whether make difference or not
    :param is_abs_diff: whether make abs and difference or not
    :param is_diff_abs: whether make difference and abs or not
    :param order: the order of difference
    :param direction: the direction of difference, 0 - inter frame, 1 - intra frame
    :param is_trunc: whether make truncation or not
    :param threshold: the threshold of truncation

    :return:
        data: QMDCT matrixs, ndarry, shape: [files_num, height, width, 1]
    """

    files_num = len(text_files_list)
    data = np.zeros([files_num, height, width, 1], dtype=np.float32)

    i = 0
    for text_file_path in text_files_list:
        content = text_read(text_file_path, height=height, width=width, separator=separator, is_abs=is_abs, is_diff=is_diff, is_abs_diff=is_abs_diff,
                            is_diff_abs=is_diff_abs, order=order, direction=direction, is_trunc=is_trunc, threshold=threshold)
        data[i] = content
        i = i + 1

    return data
