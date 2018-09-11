#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.06.26
Finished on
@author: Wang Yuntao
"""

import tensorflow as tf
import image_preprocess as ip
import audio_preprocess as ap
import text_preprocess as tp


def tfrecord_write(files_path_list, tfrecord_file_name):
    """
    read files from disk and write the data into tfrecord
    :param files_path_list: data
    :param tfrecord_file_name:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)
    for index, files_path in enumerate(files_path_list):
        files_list = get_files_list(files_path)
        for file in files_list:
            data = media_read(media_file_path, media_file_type="text", height=200, width=576, as_grey=False, sampling_rate=44100, mono=False, offset=0, duration=None)
            image = io.imread(file)
            image_raw = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
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


def media_read(media_file_path, media_file_type="text", height=200, width=576,
               as_grey=False, sampling_rate=44100, mono=False, offset=0, duration=None):
    """
    read media according to the file path and file type
    :param media_file_path: the path of media file
    :param media_file_type: the type of media file, default is "text"
    :param height: the output height of media file, default is 200
    :param width: the output width of media file, default is 576
    :param as_grey: whether covert the image to gray image or not, default is False
    :param sampling_rate: the sampling rate of audio file, default is 44100
    :param mono: whether convert the audio file to mono, default is False
    :param offset: start reading after this time (in seconds), default is 0
    :param duration: only load up to this much audio (in seconds), default is None

    :return:
        media in ndarry format
    """
    if media_file_type == "image":
        media = io.imread(media_file_path, as_grey=as_grey)
    elif media_file_type == "audio":
        media = ap.audio_read(media_file_path, sampling_rate=sampling_rate, mono=mono, offset=offset, duration=duration)
    elif media_file_path == "text":
        media = tp.text_read(media_file_path, height=height, width=width)
    else:
        media = None

    return media
