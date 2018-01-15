#!/usr/bin/env python3
# -*- coding: utf-8 -*-33

"""
Created on 2017.11.9
Finished on 2017.11.16
@author: Wang Yuntao
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成数据
train_size = 200
train_data = np.float32(np.random.rand(train_size, 2))
noise = np.random.lognormal(size=train_size) / 50
train_label = 0.2 * train_data[:, 0] + 0.8 * train_data[:, 1] + noise + 0.2
train_label = train_label.reshape(train_size, 1)

# 显示训练数据和测试数据
# ax = plt.figure().add_subplot(111, projection="3d")
# ax.scatter(train_data[:, 0], train_data[:, 1], train_label, c="r", marker="o")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()

# 建立模型
data = tf.placeholder(tf.float32, [None, 2])
value = tf.placeholder(tf.float32, [None, 1])

weights = tf.Variable(tf.zeros([2, 1]))
biases = tf.Variable(tf.zeros([1]))
predict = tf.matmul(data, weights) + biases

loss = tf.reduce_mean(tf.square(predict - value))
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/tf_learn/', sess.graph)

sess.run(init)

for i in range(1000):
    sess.run(train, {data: train_data, value: train_label})
    curr_w, curr_b, curr_loss, curr_y = sess.run([weights, biases, loss, value],
                                                 {data: train_data, value: train_label})
    print("iter: %s, weights: %s, biases: %s, loss: %s" % (i, curr_w, curr_b, curr_loss))
