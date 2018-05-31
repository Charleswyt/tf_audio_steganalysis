#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2017.11.20
Finished on 2017.11.20
@author: Wang Yuntao
"""

import csv
import tensorflow as tf
from text_preprocess import *
from image_preprocess import *
from audio_preprocess import *
from matplotlib.pylab import plt

"""
    function:
        fullfile(file_dir, file_name)                                                                           concatenate the file path (实现路径连接，功能类似于matlab中的fullfile)
        get_time(_unix_time_stamp=None)                                                                         get calender time via unix time stamp (根据Unix时间戳获取日历时间)
        get_unix_stamp(_time_string="1970-01-01 08:01:51", _format="%Y-%m-%d %H:%M:%S")                         get unix time stamp via calender time (根据日历时间获取Unix时间戳)
        read_data(cover_files_path, stego_files_path, start_idx=0, end_idx=10000, is_shuffle=True)              get file list and  corresponding label list (获取文件列表与标签列表)
        minibatches(cover_datas=None, cover_labels=None, stego_datas=None, stego_labels=None, batchsize=None)   get minibatch for training (批次读取数据, 此处的数据仍为文件列表)
        get_data_batch(files_list, height, width, carrier="audio", is_abs=False, is_diff=False, is_diff_abs=False, order=2, direction=0,
                       is_trunc=False, threshold=15, threshold_left=0, threshold_right=255)                     read a batch of data (批次读取数据)
        
        tfrecord_write(files_path_list, file_type, tfrecord_file_name)                                          write the data info into tfrecord (制备tfrecord文件)
        tfrecord_read(tfrecord_file_name)                                                                       read the data info from tfrecord (读取tfrecord文件)
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


def read_data(cover_files_path, stego_files_path, start_idx=0, end_idx=1000000, is_shuffle=True):
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


def get_data_batch(files_list, height, width, carrier="audio", is_abs=False, is_diff=False, is_diff_abs=False, order=2, direction=0,
                   is_trunc=False, threshold=15, threshold_left=0, threshold_right=255):
    """
    read data batch by batch
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
        data = text_read_batch(text_files_list=files_list, height=height, width=width, is_abs=is_abs, is_diff=is_diff, order=order, direction=direction,
                               is_diff_abs=is_diff_abs, is_trunc=is_trunc, threshold=threshold)
    elif carrier == "image":
        data = image_read_batch(image_files_list=files_list, height=height, width=width, is_diff=is_diff, order=order, direction=direction,
                                is_trunc=is_trunc, threshold=threshold, threshold_left=threshold_left, threshold_right=threshold_right)
    else:
        data = read_text_batch(text_files_list=files_list, height=height, width=width, is_abs=is_abs, is_diff=is_diff, order=order, direction=direction,
                               is_trunc=is_trunc, threshold=threshold)

    return data


def media_read(media_file_path, media_file_type, height, width, as_grey=False, sampling_rate=44100, to_mono=False,
               is_abs=False, is_diff=False, is_diff_abs=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    read media according to the file path and file type
    :param media_file_path: the path of media file
    :param media_file_type: the type of media file
    :param height: the output height of media file
    :param width: the output width of media file
    :return:
        media in ndarry format
    """
    if media_file_type == "image":
        media = io.imread(media_file_path, as_grey=as_grey)
    elif media_file_type == "audio":
        media = audio_read(media_file_path, sampling_rate=sampling_rate, to_mono=to_mono)
    elif media_file_path == "text":
        media = text_read(media_file_path)
    else:
        media = None

    return media


def tfrecord_write(files_path_list, file_type, tfrecord_file_name):
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)
    for index, files_path in enumerate(files_path_list):
        files_list = get_files_list(files_path)
        for file in files_list:
            image = io.imread(file)
            image_raw = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "media": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


def tfrecord_read(tfrecord_file_name):
    filename_queue = tf.train.string_input_producer([tfrecord_file_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "label": tf.FixedLenFeature([], tf.int64),
                                           "media": tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['media'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features["label"], tf.int32)  # 在流中抛出label张量

    return img, label


if __name__ == "__main__":
    write_tfrecord("123", "4456", "hhh")

