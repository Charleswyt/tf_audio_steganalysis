#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2017.11.20
Finished on 2017.11.20
@author: Wang Yuntao
"""

import csv
import tensorflow as tf
from operator import mul
from functools import reduce
from image_preprocess import *
from text_preprocess import *
from matplotlib.pylab import plt

"""
    function:
        read_data(cover_files_path, stego_files_path, start_idx=0, end_idx=10000, is_shuffle=True)              获取文件列表与标签列表
        minibatches(cover_datas=None, cover_labels=None, stego_datas=None, stego_labels=None, batchsize=None)   批次读取数据
        
"""


def fullfile(file_dir, file_name):
    """
    fullfile as matlab
    :param file_dir: file dir
    :param file_name: file name
    :return: a full file path
    """
    full_file_path = os.path.join(file_dir, file_name)
    full_file_path = full_file_path.replace("\\", "/")

    return full_file_path


def get_time(_unix_time_stamp=None):
    """
    unix时间戳 -> "%Y-%m-%d %H:%M:%S"格式的时间
    e.g. 1522048036 -> 2018-03-26 15:07:16
    :param _unix_time_stamp: unix时间戳
    :return:
        "%Y-%m-%d %H:%M:%S"格式的时间
    """
    _format = "%Y-%m-%d %H:%M:%S"
    if _unix_time_stamp is None:
        value = time.localtime()
    else:
        value = time.localtime(_unix_time_stamp)
    _time_string = time.strftime(_format, value)

    return _time_string


def get_unix_stamp(_time_string="1970-01-01 08:01:51", _format="%Y-%m-%d %H:%M:%S"):
    """
    time expression with "%Y-%m-%d %H:%M:%S" format -> unix time stamp
    :param _time_string: time expression with "%Y-%m-%d %H:%M:%S" format
    :param _format:
    :return: unix time stamp
    """
    _unix_time_stamp = time.mktime(time.strptime(_time_string, _format))

    return int(_unix_time_stamp)


def read_data(cover_files_path, stego_files_path, start_idx=0, end_idx=10000, is_shuffle=True):
    """
    read file names from the storage
    :param cover_files_path: the folder name of cover files
    :param stego_files_path: the folder name of stego files
    :param start_idx: the start index
    :param end_idx: the end index
    :param is_shuffle: whether shuffle or not (default is True)
    :return:
        data_list: the file name list
        label_list: the label list
    """
    cover_files_list = get_files_list(cover_files_path)                     # cover文件列表
    stego_files_list = get_files_list(stego_files_path)                     # stego文件列表
    sample_num = len(cover_files_list)                                      # 样本对数
    cover_label_list = np.zeros(sample_num, np.int32)                       # cover标签列表
    stego_label_list = np.ones(sample_num, np.int32)                        # stego标签列表
    
    temp_cover = np.array([cover_files_list, cover_label_list])
    temp_cover = temp_cover.transpose()

    temp_stego = np.array([stego_files_list, stego_label_list])
    temp_stego = temp_stego.transpose()

    if is_shuffle is True:
        np.random.shuffle(temp_cover)
        np.random.shuffle(temp_stego)

    if start_idx > sample_num:
        start_idx = 0
    if end_idx > sample_num:
        end_idx = sample_num + 1

    cover_data_list = list(temp_cover[start_idx:end_idx, 0])
    stego_data_list = list(temp_stego[start_idx:end_idx, 0])
    cover_label_list = list(temp_cover[start_idx:end_idx, 1])
    stego_label_list = list(temp_stego[start_idx:end_idx, 1])

    return cover_data_list, cover_label_list, stego_data_list, stego_label_list


def minibatches(cover_datas=None, cover_labels=None, stego_datas=None, stego_labels=None, batchsize=None):
    """
    read data batch by batch
    :param cover_datas: file name list (cover)
    :param cover_labels: label list(cover)
    :param stego_datas: file name list (stego)
    :param stego_labels: label list (stego)
    :param batchsize: batch size
    :return:
        yield datas and labels
    """
    for start_idx in range(0, len(cover_datas) - batchsize // 2 + 1, batchsize // 2):
        excerpt = slice(start_idx, start_idx + batchsize // 2)
        datas = cover_datas[excerpt]
        datas.extend(stego_datas[excerpt])
        labels = cover_labels[excerpt]
        labels.extend(stego_labels[excerpt])

        yield datas, labels


def get_data(files_list, height, width, carrier="audio", is_abs=False, is_diff=False, is_diff_abs=False, order=2, direction=0,
             is_trunc=False, threshold=15, threshold_left=0, threshold_right=255):
    """
    read data
    :param files_list: files list (audio | image | text)
    :param height: the height of the data matrix
    :param width: the width of the data matrix
    :param carrier: the type of carrier (audio | image, here if choose audio, use QMDCT matrix)
    :param is_abs: whether abs or not (default: False)
    :param is_diff: whether difference or not (default: False)
    :param is_diff_abs: whether abs after difference or not (default: False)
    :param order: the order of difference
    :param direction: the direction of difference (default: row)
    :param is_trunc: whether truncation or not (default: False)
    :param threshold: the threshold of truncation
    :param threshold_left: the threshold of truncation
    :param threshold_right: the threshold of truncation
    :return:
        the data list 4-D tensor [batch_size, height, width, channel]
    """
    if carrier == "audio":
        data = read_text_batch(text_files_list=files_list, height=height, width=width, is_abs=is_abs, is_diff=is_diff, order=order, direction=direction,
                               is_diff_abs=is_diff_abs, is_trunc=is_trunc, threshold=threshold)
    elif carrier == "image":
        data = read_image_batch(image_files_list=files_list, height=height, width=width, is_diff=is_diff, order=order, direction=direction,
                                is_trunc=is_trunc, threshold=threshold, threshold_left=threshold_left, threshold_right=threshold_right)
    else:
        data = read_text_batch(text_files_list=files_list, height=height, width=width, is_abs=is_abs, is_diff=is_diff, order=order, direction=direction,
                               is_trunc=is_trunc, threshold=threshold)

    return data


def get_model_parameters():
    """
    calculate the number of parameters of the network
    :return:
        num_params: the number of parameters of the network
    """

    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)

    return num_params


def get_weights(model, name):
    """
    get the weights
    :param model: model
    :param name: the name of the layer
    :return: the weights
    """
    return tf.constant(model[name][0], name="weights")


def get_biases(model, name):
    """
    get the biases
    :param model: model
    :param name: the name of the layer
    :return: the biases
    """
    if np.shape(model[name]) == 1:
        return tf.constant(0, name="biases")
    else:
        return tf.constant(model[name][1], name="biases")


def update_chekpoint_info(model_dir):
    """
    checkpoint format
        model_checkpoint_path: "Path1/audio_steganalysis-5797"
        all_model_checkpoint_paths: "Path1/audio_steganalysis-2618"
        all_model_checkpoint_paths: "Path1/audio_steganalysis-3553"
        all_model_checkpoint_paths: "Path1/audio_steganalysis-5797"
                                ->

    modify the model files directory
    :param model_dir: model files dir
    :return: NULL
    """
    check_point_path = os.path.join(model_dir, "checkpoint")
    with open(check_point_path) as file:
        content, new_content = list(), list()
        for line in file.readlines():
            content.append(line)

        latest_model_file_name = content[0].split("/")[-1][:-2]                                 # -2 is used to remove the punctuation "
        original_dir = content[0].replace(latest_model_file_name, "").split(":")[1][2:-3]
        if original_dir != model_dir:
            for path in content:
                new_path = path.replace(original_dir, model_dir)
                new_content.append(new_path)

            # with open(check_point_path, "w") as file:
            #     for path in new_content:
            #         file.writelines(path)
        else:
            pass


def get_model_info(model_file_path):
    graph_file_path = model_file_path + ".meta"
    saver = tf.train.import_meta_graph(graph_file_path)
    with tf.Session() as sess:
        saver.restore(sess, model_file_path)
        reader = tf.pywrap_tensorflow.NewCheckpointReader("stegshi/audio_steganalysis-5797")
        var_to_shape_map = reader.get_variable_to_shape_map()
        keys = var_to_shape_map.keys()
        var_to_shape_map_keys = sorted(keys)
        for key in var_to_shape_map_keys:
            print("tensor_name: ", key)
            # print(reader.get_tensor(key))
        print(var_to_shape_map["fc7/weight"])

        print("The model is loaded successfully.")
