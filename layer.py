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
    池化层
    :param input_data: 输入数据
    :param height: 池化核高度
    :param width: 池化核宽度
    :param x_stride: x方向步长
    :param y_stride: y方向步长
    :param name: name
    :param is_max_pool: 是否选用最大化池化
    :param padding: padding="SAME"
    :return: 4-D tensor
    """
    shape = input_data.get_shape()
    print("name: %s, shape: (%d, %d, %d, %d)" % (name, shape[0], shape[1], shape[2], shape[3]))
    if is_max_pool is True:
        return tf.nn.max_pool(input_data,
                              ksize=[1, height, width, 1],
                              strides=[1, x_stride, y_stride, 1],
                              padding=padding,
                              name=name)
    else:
        return tf.nn.avg_pool(input_data,
                              ksize=[1, height, width, 1],
                              strides=[1, x_stride, y_stride, 1],
                              padding=padding,
                              name=name)


def normalization(input_data, depth_radius, name, bias=1.0, alpha=0.001 / 9.0, beta=0.75):
    """
    :param input_data: 输入数据
    :param name: name
    :param depth_radius: depth radius
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
    批正态化
    :param input_data: 输入数据
    :param name: name
    :param offset: beta
    :param scale: gamma
    :param variance_epsilon: 避免除零错
    :return: 4-D tensor

    tf.nn.moments(x, axes, name=None, keep_dim=False)
        x: 输入数据，[batchsize, height, width, kernels]
        axes: 求解的维度，是个list，例如 [0, 1, 2]
        name: 名字
        keep_dims: 是否保持维度
        batch_mean: 当前批次的均值
        batch_var: 当前批次的方差

    tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)
        x: 输入数据
        mean: 当前批次的均值
        variance: 当前批次的方差
        offset: 一般初始化为0
        scale: 一般初始化为1
        variance_epsilon: 设为一个很小的数即可, 如0.001

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

    :param input_data: 输入数据
    :param keep_pro: 保持概率
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


def fc_layer(input_data, output_dim, name, activation_method="relu", alpha=0.1):
    """
    全连接层
    :param input_data: 输入数据
    :param output_dim: 输出维度
    :param name: name
    :param activation_method: 激活函数类型
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
    激活函数
    :param input_data: 输入数据
    :param activation_method: 激活函数类型
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
        output = tf.maximum(alpha * input_data, input_data, name="leakrelu")
    else:
        output = input_data

    return output


def conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name,
               activation_method="relu", alpha=0.1, padding="SAME", is_pretrain=True):
    """
    卷积层
    :param input_data: 输入数据 tensor [batch_size, height, width, channels]
    :param height: 卷积核高度
    :param width: 卷积核宽度
    :param x_stride: x方向的步长
    :param y_stride: y方向的步长
    :param filter_num: 卷积核个数
    :param name: 卷积层名
    :param activation_method: 激活函数类型
    :param alpha: leaky relu alpha
    :param padding: 边缘补全方式
    :param is_pretrain: 该层参数是否需要训练(默认开启)

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

        output = activation(output, activation_method, alpha)

        return output


def loss(logits, label, method="sparse_softmax_cross_entropy"):
    """
    损失函数层
    :param logits: logits [batch_size, class_num]
    :param label: one hot label
    :param method: loss method()
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
    计算准确率
    :param logits: logits
    :param label: label
    :return: auuracy
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
    优化器
    :param losses: 损失函数
    :param learning_rate: 学习率
    :param optimizer_type: 优化器种类
    可选类型:
    GradientDescent                                               -> W += learning_rate * dW
    Adagrad         catch += dW ** 2                              -> W += -learning_rate * dW / (sqrt(catch) + epsilon)
    Adam            m = beta1 * m + (1 - beta1) * dW
                    v = beta2 * v + (1 - beta2) * (dW ** 2)       -> W += -learning_rate * m / (sqrt(v) + epsilon)
    Momentum        v = (momentum * v - learning * dW)            -> W += v
    RMSProp         catch += decay * catch + (1 - decay) * dW ** 2-> W += -learning_rate * dW /  (sqrt(catch) + epsilon)
    Note:
        ADAM通常会取得比较好的结果，同时收敛非常快相比SGD
        L-BFGS适用于全batch做优化的情况
        有时候可以多种优化方法同时使用，比如使用SGD进行warm up，然后ADAM
        对于比较奇怪的需求，deepbit两个loss的收敛需要进行控制的情况，比较慢的SGD比较适用

    :param beta1: Adam优化器, 默认为0.9
    :param beta2: Adam优化器, 默认为0.999
    :param epsilon: Adam | RMSProp优化器, 默认为1e-8
    :param initial_accumulator_value: Adagrad优化器, 默认为0.1
    :param momentum: Momentum | RMSProp优化器, 默认为0.9
    :param decay: Momentum | RMSProp优化器, 默认为0.9
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


def get_learning_rate(base_learning_rate, iters, max_iter, method="fixed", gamma=0.01, step=100, power=1.0):
    """
    :param base_learning_rate: 初始学习率
    :param iters: 当前迭代次数
    :param method: 学习率更新策略(fixed, step, inv, poly)
    :param gamma: gamma
    :param step: step(for step method)
    :param power: power
    :return: learning_rate
    """

    if method == "fixed":
        learning_rate = base_learning_rate
    elif method == "step":
        learning_rate = base_learning_rate * gamma ** (int(iters / step))
    elif method == "exp":
        learning_rate = base_learning_rate * gamma ** iters
    elif method == "inv":
        learning_rate = base_learning_rate * (1 + gamma * iters) ** (-power)
    elif method == "poly":
        learning_rate = base_learning_rate * (1 - iters / max_iter) ** (-power)
    else:
        learning_rate = base_learning_rate

    return learning_rate
