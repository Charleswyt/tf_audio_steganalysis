#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2017.12.29
Finished on 2017.12.29
Modified on 2018.09.17

@author: Yuntao Wang
"""

from HPFs.filters import *
import tensorflow as tf
from math import floor
from tensorflow.contrib.layers.python.layers import batch_norm


def pool_layer(input_data, height, width, x_stride, y_stride, name, is_max_pool=True, padding="VALID"):
    """
    pooling layer
    :param input_data: the input data
    :param height: the height of the convolutional kernel, no pooling operation in the dimension of "batch_size" and "channel", so default value is 1
    :param width: the width of the convolutional kernel
    :param x_stride: stride in X axis
    :param y_stride: stride in Y axis
    :param name: the name of the pooling layer
    :param is_max_pool: if True, max pooling, else average pooling
    :param padding: padding="SAME"
    :return:
        output: a 4-D tensor [batch_size, height, width. channel]
    """

    if is_max_pool is True:
        output = tf.nn.max_pool(input_data,
                                ksize=[1, height, width, 1],
                                strides=[1, x_stride, y_stride, 1],
                                padding=padding,
                                name=name)
        pooling_type = "max_pooling"

    else:
        output = tf.nn.avg_pool(input_data,
                                ksize=[1, height, width, 1],
                                strides=[1, x_stride, y_stride, 1],
                                padding=padding,
                                name=name)
        pooling_type = "average_pooling"

    shape = output.get_shape()

    print("name: %s, shape: (%d, %d, %d), type: %s" % (name, shape[1], shape[2], shape[3], pooling_type))

    return output


def global_pool(input_data, name, is_max_pool=True):
    """
    global pooling layer
    :param input_data: the input data
    :param name: the name of the pooling layer
    :param is_max_pool: if True, max pooling, else average pooling
    :return:
        output: a 4-D tensor [batch_size, height, width. channel]
    """
    shape = input_data.get_shape()
    height, width = shape[1], shape[2]
    output = pool_layer(input_data, height, width, height, width, name=name, is_max_pool=is_max_pool)

    return output


def batch_normalization(input_data, name, activation_method="relu", is_train=True):
    """
    BN layer
    :param input_data: the input data
    :param name: name
    :param activation_method: the method of activation function
    :param is_train: if False, skip this layer, default is True
    :return:
        output: output after batch normalization
    """
    output = batch_norm(inputs=input_data, decay=0.9, center=True, scale=True, epsilon=1e-5, scope=name, updates_collections=None,
                        reuse=tf.AUTO_REUSE, is_training=is_train, zero_debias_moving_mean=True)
    output = activation_layer(input_data=output,
                              activation_method=activation_method)

    print("name: %s, activation: %s, is_train: %r" % (name, activation_method, is_train))

    return output


def batch_normalization_origin(input_data, name, offset=0.0, scale=1.0, variance_epsilon=1e-3):
    """
    batch normalization layer in IH & MMSec
    :param input_data: the input data
    :param name: name of the layer
    :param offset: beta
    :param scale: gamma
    :param variance_epsilon: avoid zero divide error
    :param variance_epsilon: variance_epsilon
    :return:
        output: a 4-D tensor [batch_size, height, width. channel]
    """

    batch_mean, batch_var = tf.nn.moments(input_data, [0])
    output_data = tf.nn.batch_normalization(x=input_data,
                                            mean=batch_mean,
                                            variance=batch_var,
                                            offset=offset,
                                            scale=scale,
                                            variance_epsilon=variance_epsilon,
                                            name=name)
    print("name: %s" % name)

    return output_data


def dropout(input_data, keep_pro=0.5, name="dropout", seed=None, is_train=True):
    """
    dropout layer
    :param input_data: the input data
    :param keep_pro: the probability that each element is kept
    :param name: name
    :param seed: int or None. An integer or None to create random seed
    :param is_train: if False, skip this layer, default is True
    :return:
        output: output with dropout
    drop率的选择：
        经过交叉验证, 隐含节点dropout率等于0.5的时候效果最好, 原因是0.5的时候dropout随机生成的网络结构最多
        dropout也可被用作一种添加噪声的方法, 直接对input进行操作. 输入层设为更接近1的数, 使得输入变化不会太大(0.8)
    """
    if is_train is True:
        output = tf.nn.dropout(x=input_data,
                               keep_prob=keep_pro,
                               name=name,
                               seed=seed)
        print("name: %s, keep_pro: %f" % (name, keep_pro))
    else:
        output = input_data

    return output


def fc_layer(input_data, output_dim, name, activation_method="relu", alpha=None, init_method="xavier", is_train=True):
    """
    fully-connected layer
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param output_dim: the dimension of the output data
    :param name: name of the layer
    :param activation_method: the type of activation function
    :param alpha: leakey relu alpha
    :param init_method: the method of weights initialization (default: xavier)
    :param is_train: if False, skip this layer, default is True
    :return:
        output: a 2-D tensor [batch_size, channel]
    """
    if is_train is True:
        shape = input_data.get_shape()
        if len(shape) == 4:
            input_dim = shape[1].value * shape[2].value * shape[3].value
        else:
            input_dim = shape[-1].value

        flat_input_data = tf.reshape(input_data, [-1, input_dim])
        if init_method is None:
            output = input_data
        else:
            with tf.variable_scope(name):
                # the method of weights initialization
                if init_method == "xavier":
                    initializer = tf.contrib.layers.xavier_initializer()
                elif init_method == "gaussian":
                    initializer = tf.random_normal_initializer(stddev=0.01)
                else:
                    initializer = tf.truncated_normal_initializer(stddev=0.01)

                weights = tf.get_variable(name="weight",
                                          shape=[input_dim, output_dim],
                                          dtype=tf.float32,
                                          initializer=initializer)

                biases = tf.get_variable(name="biases",
                                         shape=[output_dim],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))

                output = tf.nn.bias_add(value=tf.matmul(flat_input_data, weights),
                                        bias=biases,
                                        name="fc_bias_add")

                output = activation_layer(input_data=output,
                                          activation_method=activation_method,
                                          alpha=alpha)

                print("name: %s, shape: %d -> %d, activation:%s, alpha = %r"
                      % (name, input_dim, output_dim, activation_method, alpha))
    else:
        output = input_data

    return output


def fconv_layer(input_data, filter_num, name, is_train=True, padding="VALID", init_method="xavier", bias_term=True, is_pretrain=True):
    """
    fully conv layer
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_num: the number of the convolutional kernel
    :param name: name of the layer
    :param is_train: if False, skip this layer, default is True
    :param padding: the padding method, "SAME" | "VALID" (default: "VALID")
    :param init_method: the method of weights initialization (default: xavier)
    :param bias_term: whether the bias term exists or not (default: False)
    :param is_pretrain: whether the parameters are trainable (default: True)
    :return:
        output: a 4-D tensor [batch_size, height, width. channel]
    """
    if is_train is True:
        shape = input_data.get_shape()
        conv_height, conv_width, conv_channel = shape[1].value, shape[2].value, shape[3].value

        with tf.variable_scope(name):

            # the method of weights initialization
            if init_method == "xavier":
                initializer = tf.contrib.layers.xavier_initializer()
            elif init_method == "gaussian":
                initializer = tf.random_normal_initializer(stddev=0.01)
            else:
                initializer = tf.truncated_normal_initializer(stddev=0.01)

            weights = tf.get_variable(name="weights",
                                      shape=[conv_height, conv_width, conv_channel, filter_num],
                                      dtype=tf.float32,
                                      initializer=initializer,
                                      trainable=is_pretrain)
            biases = tf.get_variable(name="biases",
                                     shape=[filter_num],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=is_pretrain)
            feature_map = tf.nn.conv2d(input=input_data,
                                       filter=weights,
                                       strides=[1, 1, 1, 1],
                                       padding=padding,
                                       name="conv")
            # biases term
            if bias_term is True:
                output = tf.nn.bias_add(value=feature_map,
                                        bias=biases,
                                        name="biases_add")
            else:
                output = feature_map
    else:
        output = input_data

    # info show
    shape = output.get_shape()
    print("name: %s, shape: (%d, %d, %d)"
          % (name, shape[1], shape[2], shape[3]))

    return output


def activation_layer(input_data, activation_method="None", alpha=0.2):
    """
    activation function layer
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param activation_method: the type of activation function
    :param alpha: for leaky relu
        "relu": max(features, 0)
        "relu6": min(max(features, 0), 6)
        "tanh": tanh(features)
        "sigmoid": 1 / (1 + exp(-features))
        "softplus": log(exp(features) + 1)
        "elu": exp(features) - 1 if < 0, features otherwise
        "crelu": [relu(features), -relu(features)]
        "leakrelu": leak * features, if < 0, feature, otherwise
        "softsign": features / (abs(features) + 1)
    :return:
        output: a 4-D tensor [batch_size, height, width. channel]
    """
    if activation_method == "relu":
        output = tf.nn.relu(input_data, name="relu")
    elif activation_method == "relu6":
        output = tf.nn.relu6(input_data, name="relu6")
    elif activation_method == "tanh":
        output = tf.nn.tanh(input_data, name="tanh")
    elif activation_method == "sigmoid":
        output = tf.nn.sigmoid(input_data, name="sigmoid")
    elif activation_method == "softplus":
        output = tf.nn.softplus(input_data, name="softplus")
    elif activation_method == "crelu":
        output = tf.nn.crelu(input_data, "crelu")
    elif activation_method == "elu":
        output = tf.nn.elu(input_data, name="elu")
    elif activation_method == "softsign":
        output = tf.nn.softsign(input_data, "softsign")
    elif activation_method == "leakrelu":
        output = tf.where(input_data < 0.0, alpha * input_data, input_data)
    else:
        output = input_data

    return output


def conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name,
               activation_method="relu", alpha=0.2, padding="VALID", atrous=1,
               init_method="xavier", bias_term=True, is_pretrain=True):
    """
    convolutional layer
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param height: the height of the convolutional kernel
    :param width: the width of the convolutional kernel
    :param x_stride: stride in X axis
    :param y_stride: stride in Y axis
    :param filter_num: the number of the convolutional kernel
    :param name: the name of the layer
    :param activation_method: the type of activation function (default: relu)
    :param alpha: leaky relu alpha (default: 0.2)
    :param padding: the padding method, "SAME" | "VALID" (default: "VALID")
    :param atrous: the dilation rate, if atrous == 1, conv, if atrous > 1, dilated conv (default: 1)
    :param init_method: the method of weights initialization (default: xavier)
    :param bias_term: whether the bias term exists or not (default: False)
    :param is_pretrain: whether the parameters are trainable (default: True)

    :return:
        output: a 4-D tensor [number, height, width, channel]
    """
    channel = input_data.get_shape()[-1]

    # the method of weights initialization
    if init_method == "xavier":
        initializer = tf.contrib.layers.xavier_initializer()
    elif init_method == "gaussian":
        initializer = tf.random_normal_initializer(stddev=0.01)
    else:
        initializer = tf.truncated_normal_initializer(stddev=0.01)

    # the initialization of the weights and biases
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name="weights",
                                  shape=[height, width, channel, filter_num],
                                  dtype=tf.float32,
                                  initializer=initializer,
                                  trainable=is_pretrain)
        biases = tf.get_variable(name="biases",
                                 shape=[filter_num],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 trainable=is_pretrain)

        # the method of convolution
        if atrous == 1:
            feature_map = tf.nn.conv2d(input=input_data,
                                       filter=weights,
                                       strides=[1, x_stride, y_stride, 1],
                                       padding=padding,
                                       name="conv")
        else:
            feature_map = tf.nn.atrous_conv2d(value=input_data,
                                              filters=weights,
                                              rate=atrous,
                                              padding=padding,
                                              name="atrous_conv")
        # biases term
        if bias_term is True:
            output = tf.nn.bias_add(value=feature_map,
                                    bias=biases,
                                    name="biases_add")
        else:
            output = feature_map

        # info show
        shape = output.get_shape()
        print("name: %s, shape: (%d, %d, %d), activation: %s"
              % (name, shape[1], shape[2], shape[3], activation_method))

        # activation
        output = activation_layer(output, activation_method, alpha)

        return output


def static_conv_layer(input_data, kernel, x_stride, y_stride, name, padding="VALID"):
    """
    convolutional layer with static kernel which can be seen as a HPF
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param kernel: the filter kernel
    :param x_stride: stride in X axis
    :param y_stride: stride in Y axis
    :param name: the name of the layer
    :param padding: the padding method, "SAME" | "VALID" (default: "VALID")

    :return:
        feature_map: 4-D tensor [number, height, width, channel]
    """
    with tf.variable_scope(name):
        feature_map = tf.nn.conv2d(input=input_data,
                                   filter=kernel,
                                   strides=[1, x_stride, y_stride, 1],
                                   padding=padding,
                                   name="conv")
        shape = feature_map.get_shape()
        print("name: %s, shape: (%d, %d, %d)"
              % (name, shape[1], shape[2], shape[3]))

        return feature_map


def phase_split(input_data, block_size=8, name=None):
    """
    get m * m (2 or 4 in general) block_split layer
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param block_size: size of block, 2 or 4
    :param name: the name of the layer
    :return:
        feature_map: 4-D tensor [number, height, width, channel]
    """

    block_num = block_size * block_size
    init_block = block_num * [0]
    temp = init_block[:]
    temp[0] = 1
    phase_split_kernel = tf.constant(value=temp,
                                     dtype=tf.float32,
                                     shape=[block_size, block_size, 1, 1],
                                     name="phase_aware_" + str(block_size) + "x" + str(block_size) + "_0")

    output = tf.nn.conv2d(input=input_data,
                          filter=phase_split_kernel,
                          strides=[1, block_size, block_size, 1],
                          padding="VALID",
                          name="phase_split_" + str(block_size) + "x" + str(block_size) + "_0")

    for i in range(block_num - 1):
        temp = init_block[:]
        temp[i + 1] = 1
        phase_split_kernel = tf.constant(value=temp,
                                         dtype=tf.float32,
                                         shape=[block_size, block_size, 1, 1],
                                         name="phase_aware_" + str(block_size) + "x" + str(block_size) + "_" + str(i + 1))

        result = tf.nn.conv2d(input=input_data,
                              filter=phase_split_kernel,
                              strides=[1, block_size, block_size, 1],
                              padding="VALID",
                              name="phase_split_" + str(block_size) + "x" + str(block_size) + "_" + str(i + 1))

        output = tf.concat([output, result], 3, "phase_split_concat")

    shape = output.get_shape()
    print("name: %s, shape: (%d, %d, %d)"
          % (name, shape[1], shape[2], shape[3]))

    return output


def diff_layer(input_data, is_diff, is_diff_abs, is_abs_diff, order, direction, name, padding="SAME"):
    """
    the layer which is used for difference
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param is_diff: whether make difference or not
    :param is_diff_abs: whether make difference and abs or not
    :param is_abs_diff: whether make abs and difference or not
    :param order: the order of difference
    :param direction: the direction of difference, "inter"(between row) or "intra"(between col)
    :param name: the name of the layer
    :param padding: the method of padding, default is "SAME"

    :return:
        feature_map: 4-D tensor [number, height, width, channel]
    """

    print("name: %s, is_diff: %r, is_diff_abs: %r, is_abs_diff: %r, order: %d, direction: %s"
          % (name, is_diff, is_diff_abs, is_abs_diff, order, direction))

    if order == 0:
        return input_data
    else:
        if order == 1 and direction == "inter":
            filter_diff = tf.constant(value=[1, -1],
                                      dtype=tf.float32,
                                      shape=[2, 1, 1, 1],
                                      name="diff_inter_1")
        elif order == 1 and direction == "intra":
            filter_diff = tf.constant(value=[1, -1],
                                      dtype=tf.float32,
                                      shape=[1, 2, 1, 1],
                                      name="diff_intra_1")
        elif order == 2 and direction == "inter":
            filter_diff = tf.constant(value=[1, -2, 1],
                                      dtype=tf.float32,
                                      shape=[3, 1, 1, 1],
                                      name="diff_inter_2")
        elif order == 2 and direction == "intra":
            filter_diff = tf.constant(value=[1, -2, 1],
                                      dtype=tf.float32,
                                      shape=[1, 3, 1, 1],
                                      name="diff_intra_2")
        else:
            filter_diff = tf.constant(value=[1],
                                      dtype=tf.float32,
                                      shape=[1, 1, 1, 1],
                                      name="None")

        if is_diff is True:
            output = tf.nn.conv2d(input=input_data,
                                  filter=filter_diff,
                                  strides=[1, 1, 1, 1],
                                  padding=padding)

            return output

        elif is_diff_abs is True:
            output = tf.nn.conv2d(input=input_data,
                                  filter=filter_diff,
                                  strides=[1, 1, 1, 1],
                                  padding=padding)
            output = tf.abs(output)

            return output

        elif is_abs_diff is True:
            input_data = tf.abs(input_data)
            output = tf.nn.conv2d(input=input_data,
                                  filter=filter_diff,
                                  strides=[1, 1, 1, 1],
                                  padding=padding)

            return output

        else:
            return input_data


def rich_hpf_layer(input_data, name):
    """
    multiple HPF processing
    diff_layer(input_data, is_diff, is_diff_abs, is_abs_diff, order, direction, name, padding="SAME")

    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param name: the name of the layer
    :return:
        feature_map: 4-D tensor [number, height, width, channel]
    """
    dif_inter_1 = diff_layer(input_data, True, False, False, 1, "inter", "dif_inter_1", padding="SAME")
    dif_inter_2 = diff_layer(input_data, True, False, False, 2, "inter", "dif_inter_2", padding="SAME")
    dif_intra_1 = diff_layer(input_data, True, False, False, 1, "intra", "dif_intra_1", padding="SAME")
    dif_intra_2 = diff_layer(input_data, True, False, False, 2, "intra", "dif_intra_2", padding="SAME")

    dif_abs_inter_1 = diff_layer(input_data, False, False, True, 1, "inter", "abs_dif_inter_1", padding="SAME")
    dif_abs_inter_2 = diff_layer(input_data, False, False, True, 2, "inter", "abs_dif_inter_2", padding="SAME")
    dif_abs_intra_1 = diff_layer(input_data, False, False, True, 1, "intra", "abs_dif_intra_1", padding="SAME")
    dif_abs_intra_2 = diff_layer(input_data, False, False, True, 2, "intra", "abs_dif_intra_2", padding="SAME")

    output = tf.concat([dif_inter_1, dif_inter_2, dif_intra_1, dif_intra_2, dif_abs_inter_1, dif_abs_inter_2, dif_abs_intra_1, dif_abs_intra_2], 3, name=name)

    return output


def loss_layer(logits, labels, logits_siamese=None, is_regulation=False, coeff=1e-3, method="sparse_softmax_cross_entropy"):
    """
    calculate the loss
    :param logits: logits [batch_size, class_num]
    :param labels: labels [batch_size, class_num]
    :param logits_siamese: logits of siamese network [batch_size, class_num]
    :param is_regulation: whether regulation or not
    :param coeff: the coefficients of the regulation
    :param method: loss method
    :return:
        loss_total: loss with regularization
    """
    with tf.variable_scope("loss"):
        if method == "sparse_softmax_cross_entropy":
            loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        elif method == "siamese_loss":
            loss = siamese_loss(logits1=logits, logits2=logits_siamese, labels=labels)
        else:
            loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

        loss = tf.reduce_mean(loss)

        if is_regulation is True:
            tv = tf.trainable_variables()
            regularization_cost = coeff * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            loss_total = loss + regularization_cost
        else:
            loss_total = loss

        return loss_total


def accuracy_layer(logits, labels):
    """
    calculate the accuracy
    :param logits: logits
    :param labels: label
    :return: accuracy
    """
    with tf.variable_scope("accuracy"):
        predictions = tf.nn.in_top_k(logits, labels, 1, name="predictions")
        results = tf.cast(predictions, tf.float16)
        accuracy = tf.reduce_mean(results)

        return accuracy


def evaluation(logits, labels):
    """
    calculate the accuracy, fpr and fnr
    :param logits: logits
    :param labels: label
    :return: accuracy, fpr, fnr
    """
    predictions = tf.nn.softmax(logits)
    predictions = tf.argmax(predictions, 1)

    ones_like_labels = tf.ones_like(labels)
    zeros_like_labels = tf.zeros_like(labels)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    true_positive = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(labels, ones_like_labels),
                tf.equal(predictions, ones_like_predictions)), tf.float32))

    true_negative = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(labels, zeros_like_labels),
                tf.equal(predictions, zeros_like_predictions)), tf.float32))

    false_positive = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(labels, zeros_like_labels),
                tf.equal(predictions, ones_like_predictions)), tf.float32))

    false_negative = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(labels, ones_like_labels),
                tf.equal(predictions, zeros_like_predictions)), tf.float32))

    # true_positive_rate = true_positive / (true_positive + false_negative)
    # true_negative_rate = true_negative / (true_negative + false_positive)
    false_positive_rate = false_positive / (tf.add(false_positive, true_negative))
    false_negative_rate = false_negative / (tf.add(false_negative, true_positive))

    accuracy = 1 - (false_positive_rate + false_negative_rate) / 2

    return accuracy, false_positive_rate, false_negative_rate


def error_layer(logits, labels):
    """
    calculate the error
    :param logits: logits
    :param labels: label
    :return: error rate
    """
    with tf.variable_scope("accuracy"):
        logits = tf.nn.softmax(logits)
        results = tf.cast(tf.argmax(logits, 1), tf.int32)
        wrong_prediction = tf.not_equal(results, labels)
        accuracy = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))

        return accuracy


def siamese_loss(logits1, logits2, labels):
    """
    loss calculation for siamese network
    :param logits1: logits1 of network
    :param logits2: logits2 of network
    :param labels: new label indicating whether logit1 and logit2 belong to one class, different - 0, same - 1
    :return:
    """
    constant = 5
    constant = tf.constant(constant, name="constant", dtype=tf.float32)
    distance = tf.sqrt(tf.reduce_sum(tf.square(logits1 - logits2), 1))
    pos = tf.multiply(tf.multiply(labels, 2 / constant), tf.square(distance))
    neg = tf.multiply(tf.multiply(1 - labels, 2 * constant), tf.exp(-2.77 / constant * distance))
    loss = tf.add(pos, neg)
    loss = tf.reduce_mean(loss)
    
    return loss


def optimizer(losses, learning_rate, global_step, optimizer_type="Adam", beta1=0.9, beta2=0.999,
              epsilon=1e-8, initial_accumulator_value=0.1, momentum=0.9, decay=0.9):
    """
    optimizer
    :param losses: loss
    :param learning_rate: lr
    :param global_step: global step
    :param optimizer_type: the type of optimizer
    可选类型:
    GradientDescent                                               -> W += learning_rate * dW
    Adagrad         catch += dW ** 2                              -> W += -learning_rate * dW / (sqrt(catch) + epsilon)
    Adam            m = beta1 * m + (1 - beta1) * dW
                    v = beta2 * v + (1 - beta2) * (dW ** 2)       -> W += -learning_rate * m / (sqrt(v) + epsilon)
    Momentum        v = (momentum * v - learning * dW)            -> W += v
    RMSProp         catch += decay * catch + (1 - decay) * dW ** 2-> W += -learning_rate * dW /  (sqrt(catch) + epsilon)
    Note:
        Adam通常会取得比较好的结果，同时收敛非常快相比SGD
        L-BFGS适用于全batch做优化的情况
        有时候可以多种优化方法同时使用，比如使用SGD进行warm up，然后Adam
        对于比较奇怪的需求，deepbit两个loss的收敛需要进行控制的情况，比较慢的SGD比较适用

    :param beta1: Adam, default 0.9
    :param beta2: Adam, default 0.999
    :param epsilon: Adam | RMSProp, default 1e-8
    :param initial_accumulator_value: Adagrad, default 0.1
    :param momentum: Momentum | RMSProp, default 0.9
    :param decay: Momentum | RMSProp, default 0.9
    :return:
        train_op: optimizer
    """
    with tf.name_scope("optimizer"):
        if optimizer_type == "GradientDescent":
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
                                                    name=optimizer_type)
        elif optimizer_type == "Adagrad":
            opt = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                            initial_accumulator_value=initial_accumulator_value,
                                            name=optimizer_type)

        elif optimizer_type == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         beta1=beta1,
                                         beta2=beta2,
                                         epsilon=epsilon,
                                         name=optimizer_type)

        elif optimizer_type == "Momentum":
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=momentum,
                                             name=optimizer_type)

        elif optimizer_type == "RMSProp":
            opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                            decay=decay,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name=optimizer_type)
        else:
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
                                                    name=optimizer_type)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss=losses,
                                    global_step=global_step,
                                    name="optimizer")

        return train_op


def learning_rate_decay(init_learning_rate, global_step, decay_steps, decay_rate, decay_method="exponential", staircase=False,
                        end_learning_rate=0.0001, power=1.0, cycle=False):
    """
    function: learning rate decay -> constant |  | inverse_time | natural_exp | polynomial

    :param init_learning_rate: A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
    :param decay_method: The method of learning rate decay
    :param global_step: A scalar int32 or int64 Tensor or a Python number. Global step to use for the decay computation. Must not be negative.
    :param decay_steps: A scalar int32 or int64 Tensor or a Python number. Must be positive. See the decay computation above.
    :param decay_rate: A scalar float32 or float64 Tensor or a Python number. The decay rate.
    :param staircase: Boolean. If True decay the learning rate at discrete intervals.
    :param end_learning_rate: A scalar float32 or float64 Tensor or a Python number. The minimal end learning rate.
    :param power: A scalar float32 or float64 Tensor or a Python number. The power of the polynomial. Defaults to linear, 1.0.
    :param cycle: A boolean, whether or not it should cycle beyond decay_steps.
    :return:
        decayed_learning_rate
        type:
            fixed               -> decayed_learning_rate = learning_rate
            step                -> decayed_learning_rate = learning_rate ^ (floor(global_step / decay_steps))
            exponential_decay   -> decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            inverse_time_decay  -> decayed_learning_rate = learning_rate / (1 + decay_rate * t)
            natural_exp_decay   -> decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
            polynomial_decay    -> decayed_learning_rate =
                                    (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
    """
    if decay_method == "fixed":
        decayed_learning_rate = init_learning_rate
    elif decay_method == "exponential":
        decayed_learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase)
    elif decay_method == "inverse_time":
        decayed_learning_rate = tf.train.inverse_time_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase)
    elif decay_method == "natural_exp":
        decayed_learning_rate = tf.train.natural_exp_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase)
    elif decay_method == "polynomial":
        decayed_learning_rate = tf.train.polynomial_decay(init_learning_rate, global_step, decay_steps, decay_rate, end_learning_rate, power, cycle)
    elif decay_method == "step":
        decayed_learning_rate = tf.pow(init_learning_rate * decay_rate, floor(global_step / decay_steps))
    else:
        decayed_learning_rate = init_learning_rate

    return decayed_learning_rate
