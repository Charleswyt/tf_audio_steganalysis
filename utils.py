#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2017.11.20
Finished on 2017.11.20
@author: Wang Yuntao
"""

import tensorflow as tf
from image_preprocess import *
from text_preprocess import *
from matplotlib.pylab import plt

"""
    function:
        def read_data(cover_files_path, stego_files_path, start_idx=0, end_idx=10000, is_shuffle=True)          获取文件列表与标签列表
        minibatches(cover_datas=None, cover_labels=None, stego_datas=None, stego_labels=None, batchsize=None)   批次读取数据
        
"""


def read_data(cover_files_path, stego_files_path, start_idx=0, end_idx=10000, is_shuffle=True):
    """
    读取数据(当前数据为文件名)
    :param cover_files_path: cover文件路径
    :param stego_files_path: stego文件路径
    :param start_idx: 起始文件下标
    :param end_idx: 终止文件下标
    :param is_shuffle: 是否乱序
    :return:
        data_list: 数据列表
        label_list: 标签列表
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
    批次读取数据
    :param cover_datas: data list (cover)
    :param cover_labels: data label (cover)
    :param stego_datas: data list (stego)
    :param stego_labels: data label (stego)
    :param batchsize: batch size
    :return:
    """
    for start_idx in range(0, len(cover_datas) - batchsize // 2 + 1, batchsize // 2):
        excerpt = slice(start_idx, start_idx + batchsize // 2)
        datas = cover_datas[excerpt]
        datas.extend(stego_datas[excerpt])
        labels = cover_labels[excerpt]
        labels.extend(stego_labels[excerpt])
        yield datas, labels


def model_load(model_file):
    if not os.path.exists(model_file):
        print("There is no such file, try again please.")
    else:
        model = np.load(model_file, encoding="latin1").item()
        print(type(model))
        model_keys = model.keys()
        keys = sorted(model_keys)
        print(model["fc6"][3])

        for key in keys:
            weights = model[key][0]
            biases = model[key][1]
            print(key)
            print("weights shape: ", weights.shape)
            print("biases  shape: ", biases.shape)

        fc6 = get_weights(model, "fc6")
        print(np.shape(fc6))


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


def get_model_info(model_dir):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        print(saver)
        update_chekpoint_info(model_dir)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("model path: %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The model is loaded successfully.")


if __name__ == "__main__":
    # train files
    # cover_files_path_train = "/home/zhanghong/data/image/train/512_cover"
    # stego_files_path_train = "/home/zhanghong/data/image/train/512_stego"

    # cover_data_train_list, cover_label_train_list, stego_data_train_list, stego_label_train_list = read_data(cover_files_path_train, stego_files_path_train)
    # print(len(cover_data_train_list), len(stego_data_train_list))

    # valid files
    # cover_files_path_valid = "/home/zhanghong/data/image/val/512_cover"
    # stego_files_path_valid = "/home/zhanghong/data/image/val/512_stego"

    # cover_data_valid_list, cover_label_valid_list, stego_data_valid_list, stego_label_valid_list = read_data(cover_files_path_valid, stego_files_path_valid)
    # print(len(cover_data_valid_list), len(stego_data_valid_list))

    update_chekpoint_info("E:/Myself/1.source_code/tf_audio_steganalysis/stegshi/")
