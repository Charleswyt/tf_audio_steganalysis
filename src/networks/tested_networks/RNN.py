#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.10.31
Finished on 2018.11.01
Modified on

@author: Yuntao Wang
"""

import tensorflow as tf


def rnn_lstm(input_data, class_num):
    """
    LSTM for MP3 Audio Steganalysis (input data: 200 * 200 QMDCT coefficients matrix)
    """
    print("RNN_LSTM: LSTM for MP3 Audio Steganalysis")

    lstm_size = 1024                                                                        # hidden units
    input_data_shape = input_data.get_shape()
    n_steps, n_inputs = input_data_shape[1], input_data_shape[2]

    weights_in = tf.get_variable(name="weights_in",
                                 shape=[n_steps, lstm_size],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

    biases_in = tf.get_variable(name="biases_in",
                                shape=[lstm_size],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

    weights_out = tf.get_variable(name="weights_out",
                                  shape=[lstm_size, class_num],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())

    biases_out = tf.get_variable(name="biases_out",
                                 shape=[class_num],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))

    # 定义LSTM的基本单元
    input_data = tf.transpose(input_data, [1, 0, 2])
    input_data = tf.reshape(input_data, [-1, n_inputs])
    input_data = tf.nn.tanh(tf.add(tf.matmul(input_data, weights_in), biases_in))
    input_data = tf.split(input_data, n_steps, 0)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size,
                                             forget_bias=1.0,
                                             state_is_tuple=True,
                                             activation=tf.tanh)
    init_state = lstm_cell.zero_state(16, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                             inputs=input_data,
                                             initial_state=init_state,
                                             dtype=tf.float32)

    logits = tf.add(tf.matmul(final_state[1], weights_out), biases_out)

    return logits


def rnn_gru(input_data, class_num):
    """
    GRU for MP3 Audio Steganalysis (input data: 200 * 200 QMDCT coefficients matrix)
    """
    print("RNN_GRU: GRU for MP3 audio steganalysis")

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
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
        cells.append(cell)
    network = tf.nn.rnn_cell.MultiRNNCell(cells=cells)

    outputs, last_state = tf.nn.dynamic_rnn(cell=network, inputs=input_data, dtype=tf.float32)

    # get last output
    outputs = tf.transpose(outputs, (1, 0, 2))
    last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    logits = tf.add(tf.matmul(last_output, weights), biases)

    return logits


def rnn_bi_lstm(input_data, class_num):
    """
    Bilateral LSTM for MP3 Audio Steganalysis (input data: 200 * 200 QMDCT coefficients matrix)
    """
    print("RNN_Bi_LSTM: Bilateral LSTM for MP3 Audio Steganalysis")

    lstm_size = 1024                                                                        # hidden units

    weights = tf.get_variable(name="weights",
                              shape=[lstm_size, class_num],
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
