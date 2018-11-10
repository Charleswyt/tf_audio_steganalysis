#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.10.31
Finished on 2018.11.01
Modified on

@author: Yuntao Wang
"""

import tensorflow as tf


def rnn_lstm(input_data, class_num, is_bn):
    lstm_size = 1024                                                                        # hidden units
    weights = tf.get_variable(name="weights",
                              shape=[lstm_size, class_num],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable(name="biases",
                             shape=[class_num],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

    # 定义LSTM的基本单元
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                             inputs=input_data,
                                             dtype=tf.float32)

    logits = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)

    return logits


def rnn_gru(input_data, class_num, is_bn):
    lstm_size = 1024                                                                        # hidden units
    weights = tf.get_variable(name="weights",
                              shape=[lstm_size, class_num],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable(name="biases",
                             shape=[class_num],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

    cells = list()
    for _ in range(3):
        cell = tf.nn.rnn_cell.GRUCell(num_units=1024)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1.0)
        cells.append(cell)
    network = tf.nn.rnn_cell.MultiRNNCell(cells=cells)

    input_data = tf.transpose(input_data, [0, 2, 1])

    outputs, last_state = tf.nn.dynamic_rnn(cell=network, inputs=input_data, dtype=tf.float32)
    print(outputs.get_shape())

    # get last output
    outputs = tf.transpose(outputs, (1, 0, 2))
    last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    logits = tf.add(tf.matmul(last_output, weights), biases)

    return logits


def rnn_bi_lstm(input_data, class_num, is_bn):
    lstm_size = 1024                                                                        # hidden units
    weights = tf.get_variable(name="weights",
                              shape=[2*lstm_size, class_num],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable(name="biases",
                             shape=[class_num],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

    input_data = tf.transpose(input_data, [1, 0, 2])
    input_data = tf.reshape(input_data, [-1, 200])
    input_data = tf.split(input_data, 200)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                            lstm_bw_cell,
                                                            input_data,
                                                            dtype=tf.float32)

    logits = tf.add(tf.matmul(outputs[-1], weights), biases)

    return logits
