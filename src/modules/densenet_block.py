#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019.03.05
Finished on 2019.03.05
Modified on

@author: Yuntao Wang
"""

from layer import *


def dense_block(input_data, filter_nums, layers, name, is_bn=True):
    """
    basic block of dense net
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_nums: the number of convolutional kernels
    :param layers: the number of convolutional layers in a dense block
    :param name: the name of the dense block
    :param is_bn: if False, skip this layer, default is True
    :return:
        output: a 4-D tensor [number, height, width, channel]
    """
    layers_concat = list()
    layers_concat.append(input_data)

    output = basic_block(input_data, filter_nums=filter_nums, name=name+"_basic_block1", is_bn=is_bn)
    layers_concat.append(output)

    for layer in range(layers - 1):
        output = tf.concat(layers_concat, axis=3)
        output = basic_block(output, filter_nums=filter_nums, name=name+"_basic_block"+str(layer+2), is_bn=is_bn)
        layers_concat.append(output)

    output = tf.concat(layers_concat, axis=3)

    return output


def basic_block(input_data, filter_nums, name, is_bn=True):
    """
    basic convolutional block in dense net block
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_nums: the number of convolutional kernels
    :param name: the name of the basic block
    :param is_bn: if False, skip this layer, default is True
    :return:
        output: a 4-D tensor [number, height, width, channel]
    """
    output = batch_normalization(input_data, name=name + "_BN1_1", activation_method="tanh", is_train=is_bn)
    output = conv_layer(output, 1, 1, 1, 1, 4 * filter_nums, name=name + "_conv1_2", activation_method="None", padding="SAME")
    output = batch_normalization(output, name=name + "_BN1_3", activation_method="tanh", is_train=is_train)
    output = conv_layer(output, 3, 3, 1, 1, filter_nums, name=name + "_conv1_4", activation_method="None", padding="SAME")

    return output


def transition_layer(input_data, filter_nums, name, is_bn=True):
    """
    transition layer between two dense blocks
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_nums: the number of convolutional kernels
    :param name: the name of the layer
    :param is_bn: if False, skip this layer, default is True
    :return:
        output: a 4-D tensor [number, height, width, channel]
    """
    output = batch_normalization(input_data, name=name + "_BN1_1", activation_method="tanh", is_train=is_bn)
    output = conv_layer(output, 1, 1, 1, 1, filter_nums, name=name + "_conv1_2", activation_method="None", padding="SAME")
    output = pool_layer(output, 2, 2, 2, 2, name=name+"pool1_1", is_max_pool=True)

    return output
