#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from pre_process import *
from config import get_files_list

"""
Created on 2017.11.27
Finished on 2017.11.27
@author: Wang Yuntao
"""


def text_read(text_file_path, height=200, width=576,
              is_abs=False, is_diff=False, is_diff_abs=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    data read from one text file (从单个txt文本中读取数据)
    read_text(text_file_path, separator=",", is_abs=False, is_dif=False, order=2, direction=0, is_trunc=False, threshold=15)

    :param text_file_path: the file path (文本文件路径)
    :param height: the height of QMDCT matrix (QMDCT矩阵的高度)
    :param width: the width of QMDCT matrix (QMDCT矩阵的宽度)
    :param is_abs: whether abs or not (是否对矩阵数据取绝对值, default: False)
    :param is_diff: whether difference or not (是否对矩阵数据差分处理, default: False)
    :param is_diff_abs: whether abs after difference or not (是否对差分处理后的数据取绝对值, default: False)
    :param order: the order of the difference (差分阶数, default: 2)
    :param direction: the direction of the difference (差分方向, 0 - row, 1 - col, default: 0)
    :param is_trunc: whether truncation or not (是否对矩阵数据截断处理, default: False)
    :param threshold: threshold (截断阈值, default: 15)

    :return content: QMDCT matrix (读取的QMDCT矩阵) shape: [height, width, 1]
    """
    file = open(text_file_path)
    content = []

    # read data line by line (逐行读取数据)
    lines = file.readlines()
    for line in lines:
        numbers = [int(character) for character in line.split(",")[:-1]]
        content.append(numbers)

    # pre-process (more pre-process method can be added here)
    content = np.array(content)                                                 # list -> numpy.ndarray (将list类型的数据转为numpy.ndarray)
    content = content[:height, :width]                                          # cut (矩阵裁剪)

    if is_abs is True:
        content = abs(content)
    if is_trunc is True:
        content = truncate(content, threshold=threshold)
    if is_diff is True:
        content = np.diff(content, n=order, axis=direction)
        if is_diff_abs is True:
            content = abs(content)

    # reshape
    h = np.shape(content)[0]
    w = np.shape(content)[1]
    content = np.reshape(content, [h, w, 1])

    return content


def text_read_all(text_files_dir, height=200, width=576,
                  is_abs=False, is_diff=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    read all txt files into the memory (not recommend)

    :param text_files_dir: the folder of txt files (txt文件存储路径)
    :param height: the height of QMDCT matrix (QMDCT矩阵的高度)
    :param width: the width of QMDCT matrix (QMDCT矩阵的宽度)

    the methods of pre-process
    :param is_abs: whether abs or not (是否对矩阵数据取绝对值, default: False)
    :param is_diff: whether difference or not (是否对矩阵数据差分处理, default: False)
    :param order: the order of the difference (差分阶数, default: 2)
    :param direction: the direction of the difference (差分方向, 0 - row, 1 - col, default: 0)
    :param is_trunc: whether truncation or not (是否对矩阵数据截断处理, default: False)
    :param threshold: threshold (截断阈值, default: 15)
    :return: QMDCT matrixs (QMDCT矩阵) shape: [files_num, height, width, 1]
    """
    text_files_list = get_files_list(text_files_dir)                            # get the files list
    files_num = len(text_files_list)                                            # get the number of files in the folder

    if is_diff is True and direction == 0:
        h, w = height - order, width
    elif is_diff is True and direction == 1:
        h, w = height, width - order
    else:
        h, w = height, width
    data = np.zeros([files_num, h, w, 1], dtype=np.float32)

    i = 0
    for text_file in text_files_list:
        content = text_read(text_file, height, width, is_abs, is_diff, is_diff_abs, order, direction, is_trunc, threshold)
        data[i] = content
        i = i + 1
    
    return data


def text_read_batch(text_files_list, height=200, width=576,
                    is_abs=False, is_diff=False, order=2, direction=0, is_diff_abs=True, is_trunc=False, threshold=15):
    """
    read all txt files into the memory (not recommend)

    :param text_files_list: text files list (txt文件存储路径)
    :param height: the height of QMDCT matrix (QMDCT矩阵的高度)
    :param width: the width of QMDCT matrix (QMDCT矩阵的宽度)

    the methods of pre-process
    :param is_abs: whether abs or not (是否对矩阵数据取绝对值, default: False)
    :param is_diff: whether difference or not (是否对矩阵数据差分处理, default: False)
    :param order: the order of the difference (差分阶数, default: 2)
    :param direction: the direction of the difference (差分方向, 0 - row, 1 - col, default: 0)
    :param is_diff_abs: whether abs after difference or not (是否对经过高通滤波器的数据取绝对值, default: False)
    :param is_trunc: whether truncation or not (是否对矩阵数据截断处理, default: False)
    :param threshold: threshold (截断阈值, default: 15)
    :return: QMDCT matrixs (QMDCT矩阵) shape: [files_num, height, width, 1]
    """

    files_num = len(text_files_list)
    if is_diff is True and direction == 0:
        h, w = height - order, width
    elif is_diff is True and direction == 1:
        h, w = height, width - order
    else:
        h, w = height, width
    data = np.zeros([files_num, h, w, 1], dtype=np.float32)

    i = 0
    for text_file in text_files_list:
        content = text_read(text_file, height, width, is_abs, is_diff, is_diff_abs, order, direction, is_trunc, threshold)
        data[i] = content
        i = i + 1

    return data


def read_csv(csv_file_path):
    csv_data = pd.read_csv(csv_file_path)
    csv_data_list = csv_data.values.tolist()

    data_list, label_list = list(), list()
    for csv_data in csv_data_list:
        data_list.append(csv_data[0])
        label_list.append(csv_data[1])

    return data_list, label_list


if __name__ == "__main__":
    data_list, label_list = read_csv(r"C:\Users\Charles_CatKing\Desktop\123.csv")
    print(data_list, label_list)
