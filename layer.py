#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

"""
Created on 2017.12.29
Finished on 2017.12.29
@author: Wang Yuntao
"""

"""
function:
    build the classified network

    pool_layer(input_data, height, width, x_stride, y_stride, name, is_max_pool=True, padding="SAME")
    batch_normalization(input_data, name, offset=0.0, scale=1.0, variance_epsilon=1e-3)
    dropout(input_data, keep_pro=0.5, name="dropout")
    fc_layer(input_data, output_dim, name, relu_flag=True)
    conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name, activation="relu", 
        padding="SAME", is_pretrain=True)
    loss(logits, label)
    accuracy(logits, label)
    optimizer(losses, learning_rate, global_step, optimizer_type="Adam", beta1=0.9, beta2=0.999,
              epsilon=1e-8, initial_accumulator_value=0.1, momentum=0.9, decay=0.9)
"""


def pool_layer(input_data, height, width, x_stride, y_stride, name, is_max_pool=True, padding="SAME"):
    """
    pooling layer
    :param input_data: the input data
    :param height: the height of the convolutional kernel
    :param width: the width of the convolutional kernel
    :param x_stride: stride in X axis
    :param y_stride: stride in Y axis
    :param name: the name of the layer
    :param is_max_pool: if True, max pooling, else average pooling
    :param padding: padding="SAME"
    :return: 4-D tensor
    """

    if is_max_pool is True:
        pooling = tf.nn.max_pool(input_data,
                                 ksize=[1, height, width, 1],
                                 strides=[1, x_stride, y_stride, 1],
                                 padding=padding,
                                 name=name)
        pooling_type = "max_pooling"

    else:
        pooling = tf.nn.avg_pool(input_data,
                                 ksize=[1, height, width, 1],
                                 strides=[1, x_stride, y_stride, 1],
                                 padding=padding,
                                 name=name)
        pooling_type = "average_pooling"

    shape = pooling.get_shape()
    print("name: %s, shape: (%d, %d, %d, %d), type: %s" % (name, shape[0], shape[1], shape[2], shape[3], pooling_type))

    return pooling


def normalization(input_data, depth_radius, name, bias=1.0, alpha=0.001 / 9.0, beta=0.75):
    """
    :param input_data: the input data
    :param depth_radius: depth radius
    :param name: name
    :param bias: bias
    :param alpha: alpha
    :param beta: beta
    :return: tf.nn.lrn
        sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
        output = input / (bias + alpha * sqr_sum) ** beta
    """
    return tf.nn.lrn(input=input_data,
                     depth_radius=depth_radius,
                     bias=bias,
                     alpha=alpha,
                     beta=beta,
                     name=name)


def batch_normalization(input_data, name, offset=0.0, scale=1.0, variance_epsilon=1e-3):
    """
    BN layer
    :param input_data: the input data
    :param name: name
    :param offset: beta
    :param scale: gamma
    :param variance_epsilon: variance_epsilon
    :return: 4-D tensor
    """

    batch_mean, batch_var = tf.nn.moments(input_data, [0])
    output_data = tf.nn.batch_normalization(input_data,
                                            mean=batch_mean,
                                            variance=batch_var,
                                            offset=offset,
                                            scale=scale,
                                            variance_epsilon=variance_epsilon,
                                            name=name)
    print("name: %s" % name)

    return output_data


def dropout(input_data, keep_pro=0.5, name="dropout"):
    """
    dropout layer
    :param input_data: the input data
    :param keep_pro: the probability that each element is kept
    :param name: name
    :return:

    drop率的选择：
        经过交叉验证, 隐含节点dropout率等于0.5的时候效果最好, 原因是0.5的时候dropout随机生成的网络结构最多
        dropout也可被用作一种添加噪声的方法, 直接对input进行操作. 输入层设为更接近1的数, 使得输入变化不会太大(0.8)
    """
    output = tf.nn.dropout(input_data,
                           keep_prob=keep_pro,
                           name=name)

    print("name: %s, keep_pro: %f" % (name, keep_pro))

    return output


def fc_layer(input_data, output_dim, name, is_bn=True, activation_method="relu", alpha=0.1):
    """
    fully-connected layer
    :param input_data: the input data
    :param output_dim: the dimension of the output data
    :param name: name
    :param is_bn: whether the BN layer is used
    :param activation_method: the type of activation function
    :param alpha: leakey relu alpha
    :return: output
    """

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
                                name="bias_add")

        if is_bn is True:
            output = batch_normalization(output, name+"_BN")

        if activation_method == "leakrelu":
            print("name: %s, shape: %d -> %d, activation:%s, alpha = %f"
                  % (name, input_dim, output_dim, activation_method, alpha))
        else:
            print("name: %s, shape: %d -> %d, activation:%s"
                  % (name, input_dim, output_dim, activation_method))

        output = activation(output, activation_method, alpha)

        return output


def activation(input_data, activation_method=None, alpha=0.2):
    """
    activation function
    :param input_data: the input data
    :param activation_method: the type of activation function
    :param alpha: for leaky relu
        "relu": max(features, 0)
        "relu6": min(max(features, 0), 6)
        "tanh": tanh(features)
        "sigmoid": 1 / (1 + exp(-features))
        "softplus": log(exp(features) + 1)
        "elu": exp(features) - 1 if < 0, features otherwise
        "leakrelu": max(features, leak * features)
    :return:
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
    elif activation_method == "elu":
        output = tf.nn.elu(input_data, name="elu")
    elif activation_method == "leakrelu":
        output = tf.maximum(alpha * input_data, input_data)
    else:
        output = input_data

    return output


def conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name,
               is_bn=True, activation_method="relu", alpha=0.2, padding="SAME", is_pretrain=True):
    """
    convolutional layer
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param height: the height of the convolutional kernel
    :param width: the width of the convolutional kernel
    :param x_stride: stride in X axis
    :param y_stride: stride in Y axis
    :param filter_num: the number of the convolutional kernel
    :param name: the name of the layer
    :param is_bn: whether the BN layer is used
    :param activation_method: the type of activation function
    :param alpha: leaky relu alpha
    :param padding: the padding method, "SAME" | "VALID"
    :param is_pretrain: whether the parameters are trainable

    :return: 4-D tensor
    """
    channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights",
                                  shape=[height, width, channel, filter_num],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=is_pretrain)
        biases = tf.get_variable(name="biases",
                                 shape=[filter_num],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 trainable=is_pretrain)
        feature_map = tf.nn.conv2d(input=input_data,
                                   filter=weights,
                                   strides=[1, x_stride, y_stride, 1],
                                   padding=padding,
                                   name="conv")
        output = tf.nn.bias_add(value=feature_map,
                                bias=biases,
                                name="biases_add")
        shape = output.get_shape()
        print("name: %s, shape: (%d, %d, %d, %d), activation: %s"
              % (name, shape[0], shape[1], shape[2], shape[3], activation_method))

        if is_bn is True:
            output = batch_normalization(output, name+"_BN")
        output = activation(output, activation_method, alpha)

        return output


def loss(logits, label, method="sparse_softmax_cross_entropy"):
    """
    calculate the loss
    :param logits: logits [batch_size, class_num]
    :param label: one hot label
    :param method: loss method
    :return: loss
    """
    with tf.variable_scope("loss") as scope:
        if method == "sigmoid_cross_entropy":
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                    labels=label,
                                                                    name="cross_entropy")

        elif method == "softmax_cross_entropy":
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=label,
                                                                    name="cross_entropy")
        elif method == "sparse_softmax_cross_entropy":
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=label,
                                                                           name="loss")
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=label,
                                                                           name="loss")

        losses = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "/loss", losses)

        return losses


def accuracy(logits, label):
    """
    calculate the accuracy
    :param logits: logits
    :param label: label
    :return: accuracy
    """
    with tf.variable_scope("accuracy") as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
        # correct = tf.nn.in_top_k(logits,abel, 1)
        correct = tf.cast(correct, tf.float32)
        acc = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope.name + "/accuracy", acc)

        return acc


def optimizer(losses, learning_rate, optimizer_type="Adam", beta1=0.9, beta2=0.999,
              epsilon=1e-8, initial_accumulator_value=0.1, momentum=0.9, decay=0.9):
    """
    optimizer
    :param losses: loss
    :param learning_rate: lr
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

        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = opt.minimize(losses,
                                global_step=global_step,
                                name="optimizer")

        return train_op


def learning_rate_decay(init_learning_rate, global_step, decay_steps, decay_rate, decay_method="exponential", staircase=False,
                        end_learning_rate=0.0001, power=1.0, cycle=False,):
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
    :return: decayed_learning_rate
    type:
        exponential_decay   -> decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        inverse_time_decay  -> decayed_learning_rate = learning_rate / (1 + decay_rate * t)
        natural_exp_decay   -> decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        polynomial_decay    -> decayed_learning_rate =
                                (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
    """
    if decay_method == "constant":
        decayed_learning_rate = init_learning_rate
    elif decay_method == "exponential":
        decayed_learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase)
    elif decay_method == "inverse_time":
        decayed_learning_rate = tf.train.inverse_time_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase)
    elif decay_method == "natural_exp":
        decayed_learning_rate = tf.train.natural_exp_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase)
    elif decay_method == "polynomial":
        decayed_learning_rate = tf.train.polynomial_decay(init_learning_rate, global_step, decay_steps, decay_rate, end_learning_rate, power, cycle)
    else:
        decayed_learning_rate = init_learning_rate

    return decayed_learning_rate
