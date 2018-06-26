#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.06.26
Finished on
@author: Wang Yuntao
"""

import tensorflow as tf


def tfrecord_write(files_path_list, tfrecord_file_name):
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

