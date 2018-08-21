#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import floor
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm, layer_norm

"""
Created on 2017.12.29
Finished on 2017.12.29
@author: Wang Yuntao
"""

"""
function:
    build the classified network

    1.  pool_layer(input_data, height, width, x_stride, y_stride, name, is_max_pool=True, padding="SAME")
    2.  batch_normalization(input_data, name, offset=0.0, scale=1.0, variance_epsilon=1e-3, activation_method="relu", is_train=True)
    3.  dropout(input_data, keep_pro=0.5, name="dropout", seed=None, is_train=True)
    4.  fc_layer(input_data, output_dim, name, activation_method="relu", alpha=0.1, is_train=True)
    5.  activation_layer(input_data, activation_method="None", alpha=0.2)
    6.  conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name, activation_method="relu", alpha=0.2, padding="SAME", atrous=1,
               init_method="xavier", bias_term=False, is_pretrain=True)
    7.  static_conv_layer(input_data, kernel, x_stride, y_stride, name, padding="SAME")
    8.  loss_layer(logits, label)
    9.  accuracy(logits, label)
    10. optimizer(losses, learning_rate, global_step, optimizer_type="Adam", beta1=0.9, beta2=0.999,
              epsilon=1e-8, initial_accumulator_value=0.1, momentum=0.9, decay=0.9)
    11. learning_rate_decay(init_learning_rate, global_step, decay_steps, decay_rate, decay_method="exponential", staircase=False,
                        end_learning_rate=0.0001, power=1.0, cycle=False)
    12. size_tune(input_data)
"""


def pool_layer(input_data, height, width, x_stride, y_stride, name, is_max_pool=True, padding="SAME"):
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
                        reuse=False, is_training=is_train, zero_debias_moving_mean=True)
    output = activation_layer(input_data=output,
                              activation_method=activation_method)
    print("name: %s, activation: %s, is_training: %r" % (name, activation_method, is_train))

    return output


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


def fc_layer(input_data, output_dim, name, activation_method="relu", alpha=0.1, is_train=True):
    """
    fully-connected layer
    :param input_data: the input data
    :param output_dim: the dimension of the output data
    :param name: name
    :param activation_method: the type of activation function
    :param alpha: leakey relu alpha
    :param is_train: if False, skip this layer, default is True
    :return:
        output: a 4-D tensor [batch_size, height, width. channel]
    """
    if is_train is True:
        shape = input_data.get_shape()
        if len(shape) == 4:
            input_dim = shape[1].value * shape[2].value * shape[3].value
        else:
            input_dim = shape[-1].value

        flat_input_data = tf.reshape(input_data, [-1, input_dim])

        with tf.variable_scope(name):
            weights = tf.get_variable(name="weight",
                                      shape=[input_dim, output_dim],
                                      dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
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

            if activation_method == "leakrelu":
                print("name: %s, shape: %d -> %d, activation:%s, alpha = %f"
                      % (name, input_dim, output_dim, activation_method, alpha))
            else:
                print("name: %s, shape: %d -> %d, activation:%s"
                      % (name, input_dim, output_dim, activation_method))
    else:
        output = input_data

    return output


def activation_layer(input_data, activation_method="None", alpha=0.2):
    """
    activation function layer
    :param input_data: the input data
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
               activation_method="relu", alpha=0.2, padding="SAME", atrous=1,
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
    :param padding: the padding method, "SAME" | "VALID" (default: "SAME")
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
    with tf.variable_scope(name):
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


def static_conv_layer(input_data, kernel, x_stride, y_stride, name, padding="SAME"):
    """
        convolutional layer with static kernel which can be seen as a HPF
        :param input_data: the input data tensor [batch_size, height, width, channels]
        :param kernel: the filter kernel
        :param x_stride: stride in X axis
        :param y_stride: stride in Y axis
        :param name: the name of the layer
        :param padding: the padding method, "SAME" | "VALID"
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


def loss_layer(logits, labels, is_regulation=False, coeff=1e-3, method="sparse_softmax_cross_entropy"):
    """
    calculate the loss
    :param logits: logits [batch_size, class_num]
    :param labels: labels [batch_size, class_num]
    :param is_regulation: whether regulation or not
    :param coeff: the coefficients of the regulation
    :param method: loss method
    :return:
        loss: loss with regularization
    """
    with tf.variable_scope("loss"):
        if method == "sigmoid_cross_entropy":
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        elif method == "softmax_cross_entropy":
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        elif method == "sparse_softmax_cross_entropy":
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        else:
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

        loss_cross_entropy = tf.reduce_sum(cross_entropy)

        if is_regulation is True:
            tv = tf.trainable_variables()
            regularization_cost = coeff * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            loss = loss_cross_entropy + regularization_cost
        else:
            loss = loss_cross_entropy + 0

        return loss


def accuracy_layer(logits, labels):
    """
    calculate the accuracy
    :param logits: logits
    :param labels: label
    :return: accuracy
    """
    with tf.variable_scope("accuracy"):
        results = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct_prediction = tf.equal(results, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy


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


def size_tune(input_data):
    """
    this function is used to solve the variant size, calculate the mean, variance, max, min and other statistical characteristics of each feature map
    if the shape of input data is [batch_size, height, width, channel], the shape of output is [batch_size, 1, dim, channel].
        Herein, the dim rests with the number of statistics.
    :param input_data: input data, a 4-D tensor [batch_size, height, width, channel]
    :return:
        a 4-D tensor [batch_size, 1, dim, channel]
    """
    data_max = tf.reduce_max(input_tensor=input_data, axis=[1, 2], keep_dims=True, name="max")          # calculate the maximum of the feature map
    data_min = tf.reduce_min(input_tensor=input_data, axis=[1, 2], keep_dims=True, name="min")          # calculate the minimum of the feature map
    data_mean, data_variance = tf.nn.moments(x=input_data, axes=[1, 2], keep_dims=True)                 # calculate the mean and variance of the feature map

    output = tf.concat([data_mean, data_max, data_min, data_variance], 2)

    return output


def inception_v1(input_data, filter_num, name, activation_method="relu", alpha=0.2, padding="SAME", atrous=1,
                 init_method="xavier", bias_term=True, is_pretrain=True):
    """
    the structure of inception V1
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_num: the number of the convolutional kernel
    :param name: the name of the layer
    :param activation_method: the type of activation function (default: relu)
    :param alpha: leaky relu alpha (default: 0.2)
    :param padding: the padding method, "SAME" | "VALID" (default: "SAME")
    :param atrous: the dilation rate, if atrous == 1, conv, if atrous > 1, dilated conv (default: 1)
    :param init_method: the method of weights initialization (default: xavier)
    :param bias_term: whether the bias term exists or not (default: False)
    :param is_pretrain: whether the parameters are trainable (default: True)

    :return: 4-D tensor
    """
    branch1 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name+"_branch_1",
                         activation_method, alpha, padding, atrous, init_method, bias_term, is_pretrain)

    branch2 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name+"branch_2_1_1",
                         activation_method, alpha, padding, atrous, init_method, bias_term, is_pretrain)
    branch2 = conv_layer(branch2, 3, 3, 1, 1, filter_num, name+"branch_2_3_3",
                         activation_method, alpha, padding, atrous, init_method, bias_term, is_pretrain)

    branch3 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "branch_3_1_1",
                         activation_method, alpha, padding, atrous, init_method, bias_term, is_pretrain)
    branch3 = conv_layer(branch3, 5, 5, 1, 1, filter_num, name + "branch_3_5_5",
                         activation_method, alpha, padding, atrous, init_method, bias_term, is_pretrain)

    branch4 = pool_layer(input_data, 3, 3, 1, 1, name+"branch_4_pool", False, padding)
    branch4 = conv_layer(branch4, 1, 1, 1, 1, filter_num, name + "branch_4_1_1",
                         activation_method, alpha, padding, atrous, init_method, bias_term, is_pretrain)

    output = tf.concat([branch1, branch2, branch3, branch4], 3)

    return output
