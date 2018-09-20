#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2017.11.20
Finished on 2017.11.20
Modified on 2018.08.29

@author: Wang Yuntao
"""

import os
import numpy as np
from glob import glob
import tensorflow as tf
from file_preprocess import get_path_type
from text_preprocess import text_read, text_read_batch
from image_preprocess import image_read, image_read_batch
from audio_preprocess import audio_read, audio_read_batch

"""
    function:
        fullfile(file_dir, file_name)                                                                           concatenate the file path (实现路径连接，功能类似于matlab中的fullfile)
        get_files_list(file_dir, start_idx=None, end_idx=None)                                                  get files list
        get_time(_unix_time_stamp=None)                                                                         get calender time via unix time stamp (根据Unix时间戳获取日历时间)
        get_unix_stamp(_time_string="1970-01-01 08:01:51", _format="%Y-%m-%d %H:%M:%S")                         get unix time stamp via calender time (根据日历时间获取Unix时间戳)
        read_data(cover_files_path, stego_files_path, start_idx=None, end_idx=None, is_shuffle=True)              get file list and  corresponding label list (获取文件列表与标签列表)
        minibatches(cover_datas=None, cover_labels=None, stego_datas=None, stego_labels=None, batchsize=None)   get minibatch for training (批次读取数据, 此处的数据仍为文件列表)
        get_data_batch(files_list, height, width, carrier="audio", is_abs=False, is_diff=False, is_diff_abs=False, order=2, direction=0,
                       is_trunc=False, threshold=15, threshold_left=0, threshold_right=255)                     read a batch of data (批次读取数据)
        
        tfrecord_write(files_path_list, file_type, tfrecord_file_name)                                          write the data info into tfrecord (制备tfrecord文件)
        tfrecord_read(tfrecord_file_name)                                                                       read the data info from tfrecord (读取tfrecord文件)
        qmdct_extractor(mp3_file_path, height=200, width=576, frame_num=50, coeff_num=576)                      qmdct coefficients extraction (提取音频的QMDCT)
"""


def fullfile(file_dir, file_name):
    """
    fullfile as matlab
    :param file_dir: file dir
    :param file_name: file name
    :return:
        full_file_path: the full file path
    """
    full_file_path = os.path.join(file_dir, file_name)
    full_file_path = full_file_path.replace("\\", "/")

    return full_file_path


def get_files_list(file_dir, file_type="txt", start_idx=None, end_idx=None):
    """
    get the files list
    :param file_dir: file directory
    :param file_type: type of files, "*" is to get all files in this folder
    :param start_idx: start index
    :param end_idx: end index
    :return:
        file_list: a list containing full file path
    """
    pattern = "/*." + file_type
    file_list = sorted(glob(file_dir + pattern))
    total_num = len(file_list)
    if type(start_idx) is int and start_idx > total_num:
        start_idx = None
    if type(end_idx) is int and end_idx > total_num:
        end_idx = None
    file_list = file_list[start_idx:end_idx]

    return file_list


def get_time(unix_time_stamp=None):
    """
    unix time stamp -> time in "%Y-%m-%d %H:%M:%S" format
    e.g. 1522048036 -> 2018-03-26 15:07:16
    :param unix_time_stamp: unix time stamp
    :return:
        time_string: time in "%Y-%m-%d %H:%M:%S" format
    """
    time_format = "%Y-%m-%d %H:%M:%S"
    if unix_time_stamp is None:
        value = time.localtime()
    else:
        value = time.localtime(unix_time_stamp)
    time_string = time.strftime(time_format, value)

    return time_string


def get_unix_stamp(time_string="1970-01-01 08:01:51", time_format="%Y-%m-%d %H:%M:%S"):
    """
    time expression with "%Y-%m-%d %H:%M:%S" format -> unix time stamp
    :param time_string: time expression with "%Y-%m-%d %H:%M:%S" format
    :param time_format: time format to be exchanged
    :return:
        unix time stamp: unix time stamp
    """
    unix_time_stamp = time.mktime(time.strptime(time_string, time_format))

    return int(unix_time_stamp)


def read_data(cover_files_path, stego_files_path, start_idx=None, end_idx=None, is_shuffle=True):
    """
    read file names from the storage
    :param cover_files_path: the folder name of cover files
    :param stego_files_path: the folder name of stego files
    :param start_idx: the start index
    :param end_idx: the end index
    :param is_shuffle: whether shuffle or not (default is True)
    :return:
        cover_data_list: list of cover data
        cover_label_list: list of cover label
        stego_data_list: list of stego data
        stego_label_list: list of stego label
    """
    cover_files_list = get_files_list(cover_files_path)         # data list of cover files
    stego_files_list = get_files_list(stego_files_path)         # data list of stego files
    sample_num = len(cover_files_list)                          # total pairs of samples
    cover_label_list = np.zeros(sample_num, np.int32)           # label list of cover files
    stego_label_list = np.ones(sample_num, np.int32)            # label list of stego files

    temp = np.array([cover_files_list, cover_label_list, stego_files_list, stego_label_list])
    temp_t = temp.transpose()

    if is_shuffle is True:
        np.random.shuffle(temp_t)

    if start_idx is not None and start_idx > sample_num:
        start_idx = 0
    if end_idx is not None and end_idx > sample_num:
        end_idx = sample_num + 1

    cover_data_list = list(temp_t[start_idx:end_idx, 0])
    cover_label_list = list(temp_t[start_idx:end_idx, 1])

    stego_data_list = list(temp_t[start_idx:end_idx, 2])
    stego_label_list = list(temp_t[start_idx:end_idx, 3])

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


def get_data(file_path, height, width, channel, carrier="qmdct"):
    """
    read data batch by batch
    :param file_path: path of file
    :param height: the height of the data matrix
    :param width: the width of the data matrix
    :param channel: the channel of the data matrix
    :param carrier: the type of carrier (qmdct | audio | image)

    :return:
    """

    if carrier == "audio":
        data = audio_read(audio_file_path=file_path)
    elif carrier == "image":
        data = image_read(image_file_path=file_path, height=height, width=width, channel=channel)
    else:
        data = text_read(text_file_path=file_path, height=height, width=width, channel=channel)

    return data


def get_data_batch(files_list, height, width, channel, carrier="qmdct"):
    """
    read data batch by batch
    :param files_list: files list (audio | image | text)
    :param height: the height of the data matrix
    :param width: the width of the data matrix
    :param channel: the channel of the data matrix
    :param carrier: the type of carrier (qmdct | audio | image)

    :return:
    """
    if carrier == "audio":
        data = audio_read_batch(audio_files_list=files_list)
    elif carrier == "image":
        data = image_read_batch(image_files_list=files_list, height=height, width=width, channel=channel)
    else:
        data = text_read_batch(text_files_list=files_list, height=height, width=width, channel=channel)

    return data


def evaluation(logits, labels):
    """
    calculate the false positive rate, false negative rate, accuracy rate, precision rate and recall rate
    :param logits: prediction
    :param labels: label
    :return:
        false_positive_rate: false positive rate
        false_negative_rate: false negative rate
        accuracy_rate: accuracy rate
        precision_rate: precision rate
        recall_rate: recall rate
    """
    # format exchange
    if isinstance(logits, list):
        logits = np.array(logits)
        labels = np.array(labels)

    if isinstance(logits, type(tf.constant([0]))):
        sess = tf.Session()
        logits = logits.eval(session=sess)
        labels = labels.eval(session=sess)

    # calculate
    correct_false_num, correct_true_num, false_positive_num, false_negative_num = 0, 0, 0, 0
    for logit, label in zip(logits, labels):
        if logit == 0 and label == 0:
            correct_false_num += 1
        if logit == 1 and label == 0:
            false_positive_num += 1
        if logit == 0 and label == 1:
            false_negative_num += 1
        if logit == 1 and label == 1:
            correct_true_num += 1

    false_positive_rate = false_positive_num / str(labels.tolist()).count("1")
    false_negative_rate = false_negative_num / str(labels.tolist()).count("0")
    accuracy_rate = 1 - (false_positive_rate + false_negative_rate) / 2
    precision_rate = correct_true_num / str(logits.tolist()).count("1")
    recall_rate = correct_true_num / str(labels.tolist()).count("1")

    return false_positive_rate, false_negative_rate, accuracy_rate, precision_rate, recall_rate


def qmdct_extractor(mp3_file_path, width=576, frame_num=50, coeff_num=576):
    """
    qmdct coefficients extraction
    :param mp3_file_path: mp3 file path
    :param width: the width of QMDCT coefficients matrix, default: 576
    :param frame_num: the frame num of QMDCT coefficients extraction, default: 50
    :param coeff_num: the num of coefficients in a channel
    :return:
        QMDCT coefficients matrix, size: (4 * frame_num) * coeff_num -> 200 * 576
    """
    wav_file_path = mp3_file_path.replace(".mp3", ".wav")
    txt_file_path = mp3_file_path.replace(".mp3", ".txt")

    command = "lame_qmdct.exe " + mp3_file_path + " -framenum " + str(frame_num) + " -startind 0 " + " -coeffnum " + str(coeff_num) + " --decode"
    os.system(command)
    os.remove(wav_file_path)

    height = frame_num * 4
    content = text_read(text_file_path=txt_file_path, height=height, width=width)

    os.remove(txt_file_path)

    return content


def get_model_file_path(path):
    """
    get the path of trained tensorflow model
    :param path: input path, file path or folder path
    :return:
        the path of trained tensorflow model
    """
    if get_path_type(path) == "file":
        return path.split(".")[0]
    elif get_path_type(path) == "folder":
        return tf.train.latest_checkpoint(path)
    else:
        return None
