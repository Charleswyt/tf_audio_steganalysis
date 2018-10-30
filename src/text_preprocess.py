#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

"""
Created on 2017.11.27
Finished on 2017.11.27
Modified on 2018.08.23

@author: Wang Yuntao
"""

"""
    function:
        text_read(text_file_path, height=200, width=576, channel=1, separator=",")           read single QMDCT coefficients matrix into memory
        text_read_batch(text_files_list, height=200, width=576, channel=1, separator=",")    read QMDCT coefficients matrix in batch
"""


def text_read(text_file_path, height=200, width=576, channel=1, separator=","):
    """
    data read from one text file

    :param text_file_path: the file path
    :param height: the height of QMDCT matrix
    :param width: the width of QMDCT matrix
    :param channel: the channel of QMDCT matrix
    :param separator: separator of each elements in text file

    :return
        content: QMDCT matrix  ndarray, shape: [height, width, 1]
    """
    content = []
    try:
        file = open(text_file_path)

        # read data line by line
        lines = file.readlines()
        for line in lines:
            numbers = [int(character) for character in line.split(separator)[:-1]]
            content.append(numbers)

        content = np.array(content, dtype=np.int32)

        # reshape
        [h, w] = np.shape(content)
        content = np.reshape(content, [h, w, channel])

        height_new = None if h < height else height
        width_new = None if w < width else width
        content = content[:height_new, :width_new, :]
    except ValueError:
        print(text_file_path)

    return content


def text_read_batch(text_files_list, height=200, width=576, channel=1, separator=","):
    """
    read all txt files into the memory

    :param text_files_list: text files list
    :param height: the height of QMDCT matrix
    :param width: the width of QMDCT matrix
    :param channel: the channel of QMDCT matrix
    :param separator: separator of each elements in text file

    :return:
        data: QMDCT matrixs, ndarry, shape: [files_num, height, width, 1]
    """

    files_num = len(text_files_list)
    data = np.zeros([files_num, height, width, 1], dtype=np.float32)

    i = 0
    for text_file_path in text_files_list:
        content = text_read(text_file_path, height=height, width=width, channel=channel, separator=separator)
        data[i] = content
        i = i + 1

    return data
