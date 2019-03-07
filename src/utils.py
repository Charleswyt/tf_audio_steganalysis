#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2017.11.20
Finished on 2017.11.20
Modified on 2018.08.29

@author: Yuntao Wang
"""

import os
import time
import numpy as np
from glob import glob
import tensorflow as tf
from text_preprocess import *
from image_preprocess import *
from audio_preprocess import *
from file_preprocess import get_path_type
from tensorflow.python import pywrap_tensorflow


def folder_make(path):
    """
    create a folder
    :param path: the path to be created
    :return:
    """
    if not os.path.exists(path) and not os.path.isfile(path):
        os.mkdir(path)
    else:
        pass


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


def read_data(cover_files_path, stego_files_path, start_idx=None, end_idx=None, file_type="txt", is_shuffle=True):
    """
    read file names from the storage
    :param cover_files_path: the folder name of cover files
    :param stego_files_path: the folder name of stego files
    :param file_type: file type, default is "txt"
    :param start_idx: the start index, default is None
    :param end_idx: the end index, default is None
    :param is_shuffle: whether shuffle or not (default is True)
    :return:
        cover_data_list: list of cover data
        cover_label_list: list of cover label
        stego_data_list: list of stego data
        stego_label_list: list of stego label
    """
    cover_files_list = get_files_list(file_dir=cover_files_path, file_type=file_type,
                                      start_idx=start_idx, end_idx=end_idx)         # data list of cover files
    stego_files_list = get_files_list(file_dir=stego_files_path, file_type=file_type,
                                      start_idx=start_idx, end_idx=end_idx)         # data list of stego files
    sample_num_cover = len(cover_files_list)                                        # total pairs of samples (cover)
    sample_num_stego = len(stego_files_list)                                        # total pairs of samples (stego)
    sample_num = min(sample_num_cover, sample_num_stego)                            # deal with the quantity inequality of cover and stego

    cover_files_list = cover_files_list[:sample_num]                                # data list of cover files
    stego_files_list = stego_files_list[:sample_num]                                # data list of stego files
    cover_label_list = np.zeros(sample_num, np.int32)                               # label list of cover files
    stego_label_list = np.ones(sample_num, np.int32)                                # label list of stego files

    temp = np.array([cover_files_list, cover_label_list, stego_files_list, stego_label_list])
    temp_t = temp.transpose()

    if is_shuffle is True:
        np.random.shuffle(temp_t)

    cover_data_list = list(temp_t[:, 0])
    cover_label_list = list(temp_t[:, 1])

    stego_data_list = list(temp_t[:, 2])
    stego_label_list = list(temp_t[:, 3])

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

        temp = np.array([datas, labels])
        temp_t = temp.transpose()

        np.random.shuffle(temp_t)

        datas = list(temp_t[:, 0])
        labels = list(temp_t[:, 1])

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
    elif carrier == "mfcc":
        data = get_mfcc(audio_file_path=file_path)
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
    elif carrier == "mfcc":
        data = get_mfcc_batch(audio_files_list=files_list)
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


def get_sub_directory(directory_path):
    """
    get subdirectory in a directory
    :param directory_path: directory path to be retrieved
    :return:
        sub_directory_list
    """
    sub_directory_list = []
    file_path_list = os.listdir(directory_path)
    file_path_list.sort()
    for file_path in file_path_list:
        full_path = fullfile(directory_path, file_path)
        if os.path.isdir(full_path):
            sub_directory_list.append(full_path)

    return sub_directory_list


def write_and_encode(files_path_list, tf_record_file_path="train.tfrecords", files_type="txt",
                     carrier="qmdct", data_height=200, data_width=576, data_channel=1, start_idx=None, end_idx=None):
    """
    write the data into tfrecord file
    :param files_path_list: a files list, e.g. cover and stego
    :param tf_record_file_path: path of tfrecord file, default: ./train.tfrecords
    :param files_type: type of data file, default: "txt"
    :param carrier: type of carrier (qmdct, image, audio), default: qmdct
    :param data_height: height of input data matrix, default: 200
    :param data_width: width of input data matrix, default: 576
    :param data_channel: channel of input data matrix, default: 1
    :param start_idx: start index of files list, default: None
    :param end_idx: end index of files list, default: None
    :return:
        NULL
    """

    if os.path.exists(tf_record_file_path):
        pass
    else:
        writer = tf.python_io.TFRecordWriter(tf_record_file_path)

        for index, files_path in enumerate(files_path_list):
            files_list = get_files_list(files_path, files_type, start_idx, end_idx)
            files_list = files_list[start_idx:end_idx]

            for file in files_list:
                data = get_data(file, height=data_height, width=data_width, channel=data_channel, carrier=carrier)
                data_raw = data.tobytes()
                content = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "data_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
                }))
                writer.write(content.SerializeToString())
        writer.close()


def read_and_decode(filename, n_epoch=3, is_shuffle=True, data_height=200, data_width=576, data_channel=1):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], shuffle=is_shuffle, num_epochs=n_epoch)
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'data_raw': tf.FixedLenFeature([], tf.string)
                  })

    data = tf.decode_raw(features['data_raw'], tf.float32)
    data = tf.reshape(data, [data_height, data_width, data_channel])
    labels = tf.cast(features['label'], tf.int32)

    return data, labels


def create_batch(filename, batch_size):
    data, label = read_and_decode(filename)

    min_after_dequeue = 10 * batch_size
    capacity = 2 * min_after_dequeue
    data_batch, label_batch = tf.train.shuffle_batch([data, label],
                                                     batch_size=batchsize,
                                                     capacity=capacity,
                                                     num_threads=8,
                                                     seed=1,
                                                     min_after_dequeue=min_after_dequeue
                                                     )

    return data_batch, label_batch


def get_variables_number(trainable_variables):
    """
    calculate the number of trainable variables in the current network
    :param trainable_variables: trainable variables
    :return:
        total_parameters: the total number of trainable variables
    """
    total_parameters = 0
    for variable in trainable_variables:
        # shape is an array of tf.Dimension
        shapes = variable.get_shape()
        variable_parameters = 1
        for shape in shapes:
            variable_parameters *= shape.value
        total_parameters += variable_parameters

    return total_parameters
