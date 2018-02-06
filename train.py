#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from model import *
from utils import read_text, get_files_list

"""
Created on 2017.11.27
Finished on 2017.11.30
@author: Wang Yuntao
"""

"""
    function:
        test_steganalysis_batch: 批量文件测试
        def test_steganalysis(model_dir, file_path, file_label) 单个文件测试
"""


def test_steganalysis_batch(model_dir, files_dir):
    """
    单个文件测试
    :param model_dir: 模型存储路径
    :param files_dir: 待检测文件目录
    :param file_label: 待检测文件label
    :return: NULL
    """
    height = 198
    width = 576
    label = ["cover", "stego"]

    # 设定占位符
    x = tf.placeholder(tf.float32, [1, height, width, 1], name="QMDCTs")
    y_ = tf.placeholder(tf.int32, [1, ], name="label")
    network1(x, 2)

    # 加载模型
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, model_file)

    files_list = get_files_list(files_dir)  # 获取文件列表
    data = np.zeros([1, 198, 576, 1], dtype=np.float32)
    labels = []
    for file in files_list:
        data[0, :, :, :] = read_text(file)
        # labels.append(file_label)
        labels = np.asarray(labels, np.float32)
        ret = sess.run(y_, feed_dict={x: data, y_: labels})


def test_steganalysis(model_dir, file_path, file_label):
    """
    单个文件测试
    :param model_dir: 模型存储路径
    :param file_path: 待检测文件路径
    :param file_label: 待检测文件label
    :return: NULL
    """
    height = 198
    width = 576
    label = ["cover", "stego"]

    # 设定占位符
    x = tf.placeholder(tf.float32, [1, height, width, 1], name="QMDCTs")
    y_ = tf.placeholder(tf.int32, [1, ], name="label")
    network1(x, 2)

    # 加载模型
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, model_file)

    # 读取数据
    data = np.zeros([1, 198, 576, 1], dtype=np.float32)
    labels = []
    data[0, :, :, :] = read_text(file_path)
    labels.append(file_label)
    labels = np.asarray(labels, np.float32)
    ret = sess.run(y_, feed_dict={x: data, y_: labels})

    print("测试结果: %s, 实际结果: %s" % (label[ret[0]], label[file_label]))


def steganalysis_batch(model_dir, files_dir):
    """
    单个文件测试
    :param model_dir: 模型存储路径
    :param files_dir: 待检测文件目录
    :return: NULL
    """
    height = 198
    width = 576
    label = ["cover", "stego"]

    # 设定占位符
    x = tf.placeholder(tf.float32, [1, height, width, 1], name="QMDCTs")
    y = network1_test(x, 2)
    logits = tf.nn.softmax(y, 1)

    # 加载模型
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model_file = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, model_file)
    print("The model is loaded successfully.")

    files_list = get_files_list(files_dir)  # 获取文件列表
    data = np.zeros([1, 198, 576, 1], dtype=np.float32)
    for file in files_list:
        data[0, :, :, :] = read_text(file)
        ret = sess.run(logits, feed_dict={x: data})
        print(ret)
        result = ret.argmax()
        file_name = file.split(sep="/")[-1]
        print("文件: %s, 分析结果: %s" % (file_name, label[int(result)]))

    sess.close()


steganalysis_batch("E:/Myself/13.project/python/steganalysis_CNN/models/128_W_4_H_7_ER_10",
                   "E:/Myself/2.database/10.QMDCT/1.txt/Test")
