#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019.02.28
Finished on 2019.02.28
Modified on

@author: Yuntao Wang
"""

from layer import *


def inception_v1(input_data, filter_num, name, is_bn=True, is_max_pool=True):
    """
    the structure of inception V1
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_num: the number of the convolutional kernel
    :param name: the name of the layer
    :param is_bn: if False, skip this layer, default is True
    :param is_max_pool: whether max pooling or not (default: True)

    :return: 4-D tensor
    """
    branch1 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_1_conv_1",
                         activation_method="None", padding="SAME")
    branch1 = batch_normalization(branch1, name=name + "_branch_2_BN", activation_method="None", is_train=is_bn)

    branch2 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_2_conv_1",
                         activation_method="tanh", padding="SAME")
    branch2 = conv_layer(branch2, 3, 3, 1, 1, filter_num, name + "_branch_2_conv_3",
                         activation_method="None", padding="SAME")
    branch2 = batch_normalization(branch2, name=name + "_branch_2_BN", activation_method="None", is_train=is_bn)

    branch3 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_3_conv_1",
                         activation_method="tanh", padding="SAME")
    branch3 = conv_layer(branch3, 5, 5, 1, 1, filter_num, name + "_branch_3_conv_5",
                         activation_method="None", padding="SAME")
    branch3 = batch_normalization(branch3, name=name + "_branch_3_BN", activation_method="None", is_train=is_bn)

    branch4 = pool_layer(input_data, 3, 3, 1, 1, name + "branch_4_pool", is_max_pool=is_max_pool, padding="SAME")
    branch4 = conv_layer(branch4, 1, 1, 1, 1, filter_num, name + "branch_4_conv_1", activation_method="None", padding="SAME")
    branch4 = batch_normalization(branch4, name=name + "_branch_4_2", activation_method="None", is_train=is_bn)

    output = tf.concat([branch1, branch2, branch3, branch4], 3)

    return output


def inception_v2(input_data, filter_num, name,  is_max_pool=True, is_bn=True):
    """
    the structure of inception V2
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_num: the number of the convolutional kernel
    :param name: the name of the layer
    :param is_max_pool: whether max pooling or not (default: True)
    :param is_bn: if False, skip this layer, default is True

    :return: 4-D tensor
    """
    branch1 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_1_conv_1", activation_method="None", padding="SAME")
    branch1 = batch_normalization(branch1, name=name + "_branch_2_BN", activation_method="None", is_train=is_bn)

    branch2 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_2_conv_1", activation_method="tanh", padding="SAME")
    branch2 = conv_layer(branch2, 3, 3, 1, 1, filter_num, name + "_branch_2_conv_3", activation_method="None", padding="SAME")
    branch2 = batch_normalization(branch2, name=name+"_branch_2_BN", activation_method="None", is_train=is_bn)

    branch3 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_3_conv_1", activation_method="tanh", padding="SAME")
    branch3 = conv_layer(branch3, 5, 5, 1, 1, filter_num, name + "_branch_3_conv_5", activation_method="None", padding="SAME")
    branch3 = batch_normalization(branch3, name=name+"_branch_3_BN", activation_method="None", is_train=is_bn)

    branch4 = pool_layer(input_data, 3, 3, 1, 1, name+"branch_4_pool", is_max_pool=is_max_pool, padding="SAME")
    branch4 = conv_layer(branch4, 1, 1, 1, 1, filter_num, name+"branch_4_conv_1", activation_method="None", padding="SAME")
    branch4 = batch_normalization(branch4, name=name + "_branch_4_BN", activation_method="None", is_train=is_bn)

    output = tf.concat([branch1, branch2, branch3, branch4], 3)

    input_data_branch = conv_layer(input_data, 1, 1, 1, 1, 4*filter_num, name+"_input_data_branch_conv_1", activation_method="None", padding="SAME")
    input_data_branch = batch_normalization(input_data_branch, name + "_input_data_branch_BN", activation_method="None", is_train=is_bn)

    output = output + input_data_branch
    output = activation_layer(output, activation_method=activation_method)

    return output


def inception_v3(input_data, filter_num, name, is_max_pool=True, is_bn=True):
    """
    the structure of inception V3
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_num: the number of the convolutional kernel
    :param name: the name of the layer
    :param is_max_pool: whether max pooling or not (default: True)
    :param is_bn: if False, skip this layer, default is True

    :return: 4-D tensor
    """
    branch1 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name+"_branch_1_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch1 = batch_normalization(branch1, name=name + "_branch_1_BN", activation_method="None", is_train=is_bn)

    branch2 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_2_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch2 = conv_layer(branch2, 1, 3, 1, 1, filter_num, name + "_branch_2_conv_2",
                         activation_method="None", alpha=alpha, padding=padding)
    branch2 = conv_layer(branch2, 3, 1, 1, 1, filter_num, name + "_branch_2_conv_3",
                         activation_method="None", alpha=alpha, padding=padding)
    branch2 = batch_normalization(branch2, name=name + "_branch_2_BN", activation_method="None", is_train=is_bn)

    branch3 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_3_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch3 = conv_layer(branch3, 1, 5, 1, 1, filter_num, name + "_branch_3_conv_2",
                         activation_method="None", alpha=alpha, padding=padding)
    branch3 = conv_layer(branch3, 5, 1, 1, 1, filter_num, name + "_branch_3_conv_3",
                         activation_method="None", alpha=alpha, padding=padding)
    branch3 = batch_normalization(branch3, name=name + "_branch_3_BN", activation_method="None", is_train=is_bn)

    branch4 = pool_layer(input_data, 3, 3, 1, 1, name + "branch_4_pool", is_max_pool, padding=padding)
    branch4 = conv_layer(branch4, 1, 1, 1, 1, filter_num, name + "branch_4_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch4 = batch_normalization(branch4, name=name + "_branch_4_BN", activation_method="None", is_train=is_bn)

    output = tf.concat([branch1, branch2, branch3, branch4], 3)

    input_data_branch = conv_layer(input_data, 1, 1, 1, 1, 4 * filter_num, name + "_input_data_branch_conv_1",
                                   activation_method="None", alpha=alpha, padding=padding)
    input_data_branch = batch_normalization(input_data_branch, name=name + "_input_data_branch_BN", activation_method="None", is_train=is_bn)

    output = output + input_data_branch
    output = activation_layer(output, activation_method=activation_method)

    return output


def inception_v4(input_data, filter_num, name, activation_method="relu", alpha=0.2, padding="VALID", is_max_pool=True, is_bn=True):
    """
    the structure of inception V4
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param filter_num: the number of the convolutional kernel
    :param name: the name of the layer
    :param activation_method: the type of activation function (default: relu)
    :param alpha: leaky relu alpha (default: 0.2)
    :param padding: the padding method, "SAME" | "VALID" (default: "SAME")
    :param is_max_pool: whether max pooling or not (default: True)
    :param is_bn: if False, skip this layer, default is True

    :return: 4-D tensor
    """
    branch1 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_1_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch1 = batch_normalization(branch1, name=name + "_branch_2_BN", activation_method="None", is_train=is_bn)

    branch2 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_2_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch2 = conv_layer(branch2, 3, 3, 1, 1, filter_num, name + "_branch_2_conv_3",
                         activation_method="None", alpha=alpha, padding=padding)
    branch2 = batch_normalization(branch2, name=name + "_branch_2_BN", activation_method="None", is_train=is_bn)

    branch3 = conv_layer(input_data, 1, 1, 1, 1, filter_num, name + "_branch_3_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch3 = conv_layer(branch3, 5, 5, 1, 1, filter_num, name + "_branch_3_conv_5",
                         activation_method="None", alpha=alpha, padding=padding)
    branch3 = batch_normalization(branch3, name=name + "_branch_3_BN", activation_method="None", is_train=is_bn)

    branch4 = pool_layer(input_data, 3, 3, 1, 1, name + "branch_4_pool", is_max_pool, padding=padding)
    branch4 = conv_layer(branch4, 1, 1, 1, 1, filter_num, name + "branch_4_conv_1",
                         activation_method="None", alpha=alpha, padding=padding)
    branch4 = batch_normalization(branch4, name=name + "_branch_4_BN", activation_method="None", is_train=is_bn)

    output = tf.concat([branch1, branch2, branch3, branch4], 3)

    input_data_branch = conv_layer(input_data, 1, 1, 1, 1, 4 * filter_num, name + "_input_data_branch_conv_1",
                                   activation_method="None", alpha=alpha, padding=padding)
    input_data_branch = batch_normalization(input_data_branch, name=name + "_input_data_branch_BN", activation_method="None", is_train=is_bn)

    output = output + input_data_branch
    output = activation_layer(output, activation_method=activation_method)

    return output