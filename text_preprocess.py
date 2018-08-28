#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from utils import get_files_list

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


def text_read(text_file_path, height=200, width=576, separator=","):
    """
    data read from one text file

    :param text_file_path: the file path
    :param height: the height of QMDCT matrix
    :param width: the width of QMDCT matrix
    :param separator: separator of each elements in text file
    :return
        content: QMDCT matrix  ndarray, shape: [height, width, 1]
    """
    file = open(text_file_path)
    content = []

    # read data line by line (逐行读取数据)
    lines = file.readlines()
    for line in lines:
        numbers = [int(character) for character in line.split(separator)[:-1]]
        content.append(numbers)

    # reshape
    content = np.reshape(content, [height, width, 1])

    return content


def text_read_all(text_files_dir, height=200, width=576, separator=","):
    """
    read all txt files into the memory (not recommend)

    :param text_files_dir: the folder of txt files (txt文件存储路径)
    :param height: the height of QMDCT matrix (QMDCT矩阵的高度)
    :param width: the width of QMDCT matrix (QMDCT矩阵的宽度)
    :param separator: separator of each elements in text file
    :return:
        data: QMDCT matrices, ndarry, shape: [files_num, height, width, 1]
    """
    text_files_list = get_files_list(text_files_dir)                            # get the files list
    files_num = len(text_files_list)                                            # get the number of files in the folder

    data = np.zeros([files_num, height, width, 1], dtype=np.float32)

    i = 0
    for text_file in text_files_list:
        content = text_read(text_file, height, width, separator)
        data[i] = content
        i = i + 1
    
    return data


def text_read_batch(text_files_list, height=200, width=576, separator=","):
    """
    read all txt files into the memory

    :param text_files_list: text files list
    :param height: the height of QMDCT matrix
    :param width: the width of QMDCT matrix
    :param separator: separator of each elements in text file
    :return:
        data: QMDCT matrixs, ndarry, shape: [files_num, height, width, 1]
    """

    files_num = len(text_files_list)
    data = np.zeros([files_num, height, width, 1], dtype=np.float32)

    i = 0
    for text_file_path in text_files_list:
        content = text_read(text_file_path, height, width, separator)
        data[i] = content
        i = i + 1

    return data
