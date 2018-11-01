#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on
Finished on
@author: Yuntao Wang
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

# hyper parameters
learning_rate = 1e-3                        # learning rate
n_steps = 380                               # length of sequence
n_inputs = 380                              # number of units in the input layer
n_hidden = 1024                             # number of units in the hidden layer
n_layers = 2                                # number of layers
class_num = 2                               # number of classification

# placeholder
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, n_steps * n_inputs], name="x_input")      # 输入
    y = tf.placeholder(tf.float32, [None, class_num], name="y_input")               # 输出
    keep_prob = tf.placeholder(tf.float32, name="keep_prob_input")                  # 保持多少不被 dropout
    batch_size = tf.placeholder(tf.int32, [], name="batch_size_input")              # 批大小

# weights and biases
with tf.name_scope("weights"):
    weights = tf.Variable(tf.truncated_normal([n_hiddens, class_num], stddev=0.1), dtype=tf.float32, name="weights")
    tf.summary.histogram("output_layer_weights", weights)
with tf.name_scope("biases"):
    biases = tf.Variable(tf.random_normal([class_num]), name="biases")
    tf.summary.histogram("output_layer_biases", biases)

with tf.name_scope("output_layer"):
    pred = RNN_LSTM(x, weights, biases)
    tf.summary.histogram("outputs", pred)

with tf.name_scope("loss"):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
    tf.summary.scalar("loss", cost)

with tf.name_scope("train"):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("accuracy"):
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1))[1]
    tf.summary.scalar("accuracy", accuracy)


def rnn_lstm(x, weights, biases):
    # RNN 输入 reshape
    print(x)
    x = tf.reshape(x, [-1, n_steps, n_inputs])
    print(x)
    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope("lstm_dropout"):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    enc_cells = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
    with tf.name_scope("lstm_cells_layers"):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)
    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn 运行网络
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
    # 输出
    return tf.nn.softmax(tf.matmul(outputs[:, -1, :], weights) + biases)


merged = tf.summary.merge_all()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# parameters
step_train, step_valid = 0, 0                                                           # train step and valid step
max_accuracy = 0
max_accuracy_epoch = 0
n_epoch = 10

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

    # training
    for epoch in range(n_epoch):
        start_time = time.time()

        # read files list
        cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list = read_data(cover_train_path,
                                                                                                                 stego_train_path,
                                                                                                                 start_index_train,
                                                                                                                 end_index_train)

        # read files list
        cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list = read_data(cover_valid_path,
                                                                                                                 stego_valid_path,
                                                                                                                 start_index_valid,
                                                                                                                 end_index_valid)
        # update the learning rate
        lr = sess.run(learning_rate)

        # train
        train_iterations, train_loss, train_accuracy = 0, 0, 0
        for x_train_batch, y_train_batch in \
                minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size_train):
            # data read and process
            x_train_data = get_data_batch(x_train_batch, height, width, carrier=carrier, is_diff=is_diff, order=order, direction=direction, is_diff_abs=is_diff_abs,
                                          is_trunc=is_trunc, threshold=threshold)

            # get the accuracy and loss
            _, err, ac, summary_str_train = sess.run([train_optimizer, loss, accuracy, summary_op],
                                                     feed_dict={data: x_train_data, labels: y_train_batch, is_bn: True})

            train_loss += err
            train_accuracy += ac
            step_train += 1
            train_iterations += 1
            train_writer_train.add_summary(summary_str_train, global_step=step_train)

            print("epoch: %003d, train iterations: %003d: train loss: %f, train accuracy: %f" % (epoch + 1, train_iterations, err, ac))

        print("==================================================================================")

        # validation
        valid_iterations, valid_loss, valid_accuracy = 0, 0, 0
        for x_valid_batch, y_valid_batch in \
                minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size_valid):
            # data read and process
            x_valid_data = get_data_batch(x_valid_batch, height, width, carrier=carrier, is_diff=is_diff, order=order, direction=direction, is_diff_abs=is_diff_abs,
                                          is_trunc=is_trunc, threshold=threshold)

            # get the accuracy and loss
            err, ac, summary_str_valid = sess.run([loss, accuracy, summary_op],
                                                  feed_dict={data: x_valid_data, labels: y_valid_batch, is_bn: True})
            valid_loss += err
            valid_accuracy += ac
            valid_iterations += 1
            step_valid += 1
            train_writer_valid.add_summary(summary_str_valid, global_step=step_valid)

            print("epoch: %003d, valid iterations: %003d, valid loss: %f, valid accuracy: %f" % (epoch + 1, valid_iterations, err, ac))

        # calculate the average in a batch
        train_loss_average = train_loss / train_iterations
        valid_loss_average = valid_loss / valid_iterations
        train_accuracy_average = train_accuracy / train_iterations
        valid_accuracy_average = valid_accuracy / valid_iterations

        # model save
        if valid_accuracy_average > max_accuracy:
            max_accuracy = valid_accuracy_average
            max_accuracy_epoch = epoch + 1
            saver.save(sess, os.path.join(model_file_path, model_file_name), global_step=global_step)
            print("The model is saved successfully.")

        print("epoch: %003d, learning rate: %f, train loss: %f, train accuracy: %f, valid loss: %f, valid accuracy: %f, "
              "max valid accuracy: %f, max valid acc epoch: %d" % (epoch + 1, lr, train_loss_average, train_accuracy_average, valid_loss_average,
                                                                   valid_accuracy_average, max_accuracy, max_accuracy_epoch))

        end_time = time.time()
        print("Runtime: %.2fs" % (end_time - start_time))

    train_writer_train.close()
    train_writer_valid.close()
    sess.close()
