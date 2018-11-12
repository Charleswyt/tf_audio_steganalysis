#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

"""
Created on 2017.11.27
Finished on 2017.11.27
Modified on 2018.08.23

@author: Yuntao Wang
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
        with open(text_file_path) as file:
            # read data line by line
            lines = file.readlines()
            for line in lines:
                try:
                    numbers = [int(character) for character in line.split(separator)[:-1]]
                except ValueError:
                    numbers = [float(character) for character in line.split(separator)[:-1]]
                content.append(numbers)

            content = np.array(content)

            # reshape
            [h, w] = np.shape(content)

            height_new = None if h < height else height
            width_new = None if w < width else width

            if channel == 0:
                content = content[:height_new, :width_new]
            else:
                content = np.reshape(content, [h, w, channel])
                content = content[:height_new, :width_new, :channel]

    except ValueError:
        print("Error read: %s" % text_file_path)

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
    data = np.zeros([files_num, height, width], dtype=np.float32) if channel == 0 else np.zeros([files_num, height, width, channel], dtype=np.float32)

    i = 0
    for text_file_path in text_files_list:
        content = text_read(text_file_path, height=height, width=width, channel=channel, separator=separator)
        data[i] = content
        i = i + 1

    return data
