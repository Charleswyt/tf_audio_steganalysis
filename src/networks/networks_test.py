#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.11.12
Finished on 2018.11.12
Modified on

@author: Yuntao Wang
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("data/", one_hot=True)
train_x, train_y, rest_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# hyper parameters
lr = 0.001
batch_size = 128
training_iters = 100000

# 为了使用RNN来分类图片，我们把每张图片的行看成是一个像素序列（sequence）。因为MNIST图片的大小是28×28像素，
# 所以我们把每一个图像样本看成一行行的序列。因此，共有（28个元素的序列）×（28行），然后每一步输入的序列长度是28，输入的步数是28步
# 神经网络的参数
n_inputs = 28           # 输入层的n
n_steps = 28            # 28长度
n_hidden_units = 128    # 隐藏层的神经元个数
n_classes = 10          # 输出的数量，即分类的类别，0～9个数字，共有10个

# placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 参数初始化
weights = {
    # (28, 128)
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# 定义RNN模型
def network(_input_data, _weights, _biases):
    # 把输入的X转换成X ==> (128 batch * 28 steps, 28 inputs)
    _input_data = tf.reshape(_input_data, [-1, n_inputs])

    # 进入隐藏层
    # X_in = (128 batch * 28 steps, 128 hidden)
    x_in = tf.matmul(_input_data, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])
    # 这里采用基本的LSTM循环网络单元：basic LSTM Cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 初始化为零值，lstm单元由两个部分组成：(c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn 接收张量(batch, steps, inputs)或者(steps, batch, inputs)x_in
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], _weights['out']) + _biases['out']

    return results


# 定义损失函数和优化器，优化器采用AdamOptimizer
pred = network(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 定义模型预测结果及准确率计算方法
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 在一个会话中启动图，开始训练，每20次输出1次准确率的大小
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print("Step-", step, sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1
