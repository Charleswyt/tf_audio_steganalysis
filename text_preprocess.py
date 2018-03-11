#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pre_process import *

"""
Created on 2017.11.27
Finished on 2017.11.27
@author: Wang Yuntao
"""


def get_files_list(files_dir):
    """
    # get the files list
    :param files_dir: the folder of the files
    :return: files list
    """

    filename = os.listdir(files_dir)
    files_list = [os.path.join(files_dir, file) for file in filename]

    return files_list


def read_text(text_file_path, separator=",", height=200, width=576,
              is_abs=False, is_diff=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    data read from one text file (从单个txt文本中读取数据)
    read_text(text_file_path, separator=",", is_abs=False, is_dif=False, order=2, direction=0, is_trunc=False, threshold=15)

    :param text_file_path: the file path (文本文件路径)
    :param separator: separator (数据分隔符)
    :param height: the height of QMDCT matrix (QMDCT矩阵的高度)
    :param width: the width of QMDCT matrix (QMDCT矩阵的宽度)
    :param is_abs: whether abs or not (是否对矩阵数据取绝对值, default: False)
    :param is_diff: whether difference or not (是否对矩阵数据差分处理, default: False)
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
        numbers = [int(character) for character in line.split(separator)[:-1]]
        content.append(numbers)

    # pre-process (more pre-process method can be added here)
    content = np.array(content)                                                 # list -> numpy.ndarray (将list类型的数据转为numpy.ndarray)
    content = content[:height, :width]                                          # cut (矩阵裁剪)

    if is_abs is True:
        content = abs(content)
    if is_diff is True:
        content = np.diff(content, n=order, axis=direction)
    if is_trunc is True:
        content = truncate(content, threshold)

    # reshape
    h = np.shape(content)[0]
    w = np.shape(content)[1]
    print("height = %d, width = %d" % (h, w))

    content = np.reshape(content, [h, w, 1])

    return content


def read_text_all(text_files_dir, separator=",", height=200, width=576,
                  is_abs=False, is_diff=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    read all txt files into the memory (not recommend)

    :param text_files_dir: the folder of txt files (txt文件存储路径)
    :param separator: separator (数据分隔符)
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
    files_num = len(text_files_list)                                            # get the number of files in the foloder
    data = np.zeros([files_num, height, width, 1], dtype=np.float32)
    i = 0
    for text_file in text_files_list:
        content = read_text(text_file, separator, height, width, is_abs, is_diff, order, direction, is_trunc, threshold)
        data[i, :, :, 1] = content
        i = i + 1
    
    return data


def read_text_batch(text_files_list, separator=",", height=200, width=576,
                    is_abs=False, is_diff=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    read all txt files into the memory (not recommend)

    :param text_files_list: text files list (txt文件存储路径)
    :param separator: separator (数据分隔符)
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

    files_num = len(text_files_list)
    data = np.zeros([files_num, height-2, width, 1], dtype=np.float32)

    i = 0
    for text_file in text_files_list:
        content = read_text(text_file, separator, height, width, is_abs, is_diff, order, direction, is_trunc, threshold)
        data[i] = content
        i = i + 1

    return data


# def csv_read(filelist):
#     # 构建文件队列
#     Q = tf.train.string_input_producer(filelist)
#     # 构建读取器
#     reader = tf.TextLineReader()
#     # 读取队列
#     key, value = reader.read(Q)
#     # 构建解码器
#     x1, y = tf.decode_csv(value, record_defaults=[["None"], ["None"]])
#     # 进行管道批处理
#     #x1_batch, y_batch = tf.train.batch([x1, y], batch_size=1, num_threads=1, capacity=12)
#     # 开启会话
#     with tf.Session() as sess:
#         # 创建线程协调器
#         coord = tf.train.Coordinator()
#         # 开启线程
#         threads = tf.train.start_queue_runners(sess, coord=coord)
#         # 执行任务
#         print(sess.run(x1))
#         print(sess.run(y))
#         # print(sess.run([x1_batch, y_batch]))
#         # 线程回收
#         coord.request_stop()
#         coord.join(threads)


if __name__ == "__main__":
    file_path = "E:/Myself/2.database/10.QMDCT/1.txt/APS/128_01/wav10s_00689.txt"
    if os.path.exists(file_path):
        read_text(file_path, is_abs=True, is_diff=True, order=2, direction=1, is_trunc=True, threshold=3)
    else:
        print("This file does not exist.")
