#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019.03.05
Finished on 2019.03.05
Modified on

@author: Yuntao Wang
"""

from layer import *


def res_conv_block(input_data, height, width, x_stride, y_stride, filter_num, name,
                   activation_method="relu", alpha=0.2, padding="SAME", atrous=1,
                   init_method="xavier", bias_term=True, is_pretrain=True):
    """
    residual convolutional layer
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
    conv1 = conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name=name+"_conv1", activation_method=activation_method, alpha=alpha,
                       padding=padding, atrous=atrous, init_method=init_method, bias_term=bias_term, is_pretrain=is_pretrain)
    conv2 = conv_layer(conv1, height, width, x_stride, y_stride, filter_num, name=name + "_conv2", activation_method="None", alpha=alpha,
                       padding=padding, atrous=atrous, init_method=init_method, bias_term=bias_term, is_pretrain=is_pretrain)
    output = tf.add(input_data, conv2, name=name+"add")

    output = activation_layer(input_data=output,
                              activation_method=activation_method,
                              alpha=alpha)

    return output


def res_conv_block_beta(input_data, height, width, x_stride, y_stride, filter_num, name,
                        activation_method="relu", alpha=0.2, padding="SAME", atrous=1,
                        init_method="xavier", bias_term=True, is_pretrain=True):
    """
    residual convolutional layer
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
    conv1 = conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name=name+"_conv1", activation_method=activation_method, alpha=alpha,
                       padding=padding, atrous=atrous, init_method=init_method, bias_term=bias_term, is_pretrain=is_pretrain)
    conv2 = conv_layer(conv1, height, width, x_stride, y_stride, filter_num, name=name + "_conv2", activation_method="None", alpha=alpha,
                       padding=padding, atrous=atrous, init_method=init_method, bias_term=bias_term, is_pretrain=is_pretrain)
    output = tf.add(input_data, conv2, name=name+"add")

    output = activation_layer(input_data=output,
                              activation_method=activation_method,
                              alpha=alpha)

    return output
