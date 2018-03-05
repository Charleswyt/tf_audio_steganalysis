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
    # 获取文件列表
    filename = os.listdir(files_dir)
    files_list = [os.path.join(files_dir, file) for file in filename]

    return files_list


def read_text(text_file_path, content_type="array", separator=",", ):
    """
    从txt文本中读取数据
    content = read_text(text_file_path, content_type="array", separator=",")

    :param text_file_path: 文本文件路径
    :param content_type: 读取形式(str, strlist, list, array)
    :param separator: 数据分隔符
    :return content: 数据
    """
    file = open(text_file_path)
    content = []

    if content_type == "str":
        content = file.read()

    elif content_type == "strlist":
        lines = file.readlines()
        for line in lines:
            content.append(line)

    elif content_type == "list":
        lines = file.readlines()
        for line in lines:
            numbers = [int(character) for character in line.split(separator)[:-1]]
            content.append(numbers)

    elif content_type == "array":
        lines = file.readlines()
        for line in lines:
            numbers = [int(character)
                       for character in line.split(separator)[:-1]]
            content.append(numbers)
        content = np.array(content)
        content = diff(content, "row", 2)
        height = np.shape(content)[0]
        width = np.shape(content)[1]
        content = np.reshape(content, [height, width, 1])

    return content


def read_text_all(text_files_dir, height=200, width=576, content_type="array", separator=",",
                  is_abs=False, is_trunc=False, threshold=3, is_downsampling=False, stride=2):
    """
    将所有txt一次性读入内存(不建议)
    :param text_files_dir: txt文件存储路径
    :param height: QMDCT系数矩阵高度
    :param width: QMDCT系数矩阵宽度
    :param content_type: 类型
    :param separator: txt文本分隔符
    预处理方式
    :param is_abs: 是否取绝对值
    :param is_trunc: 是否做截断处理
    :param threshold: 截断阈值
    :param is_downsampling: 是否做下采样
    :param stride: 降采样间隔
    :return:
    """
    text_files_list = get_files_list(text_files_dir)
    files_num = len(text_files_list)
    if is_downsampling is True:
        depth = stride ** 2
        height_new = height // stride
        width_new = width // stride
        data = np.zeros([files_num, height_new, width_new, depth], dtype=np.float32)
    else:
        data = np.zeros([files_num, height, width, 1], dtype=np.float32)
    i = 0
    for text_file in text_files_list:
        content = read_text(text_file, content_type, separator)

        # 预处理(加绝对值，做截断，降采样)
        if is_abs is True:
            content = abs(content)
        if is_trunc is True:
            content = truncate(content, threshold)
        if is_downsampling is True:
            data[i] = downsampling(content, stride)
        else:
            data[i, :, :, 1] = content
        i = i + 1
    
    return data


def read_text_batch(text_files_list, height=200, width=576, content_type="array", separator=",",
                    is_abs=False,
                    is_trunc=False, threshold=3,
                    is_downsampling=False, stride=2):
    files_num = len(text_files_list)
    if is_downsampling is True:
        depth = stride ** 2
        height_new = height // stride
        width_new = width // stride
        data = np.zeros([files_num, height_new, width_new, depth], dtype=np.float32)
    else:
        data = np.zeros([files_num, height-2, width, 1], dtype=np.float32)

    i = 0
    for text_file in text_files_list:
        content = read_text(text_file, content_type, separator)     # shape: [height, width, channel]

        # 预处理(加绝对值，做截断，降采样)
        if is_downsampling is True:
            data[i] = downsampling(content, stride)
        elif is_abs is True:
            data[i] = abs(data[i])
        elif is_trunc is True:
            data[i] = truncate(data[i], threshold)
        else:
            data[i] = content
        i = i + 1

    return data


files_dir = "C:/Users/Charles_CatKing/Desktop/steganalysis/cover"
# data = read_text_all(files_dir)
# print(data)
# print(np.shape(data))
# print(len(data))


# def read_text_batch(text_files_dir, batch_size, content_type="array", separator=","):
#     text_files_list = get_files_list(text_files_dir)


# import tensorflow as tf
# import os


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

read_text("E:/Myself/2.database/10.QMDCT/1.txt\EECS/128_W_4_H_7_ER_10/wav10s_00001.txt")
