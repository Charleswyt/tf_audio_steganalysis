#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019.01.08
Finished on 2019.01.08
Modified on

@author: Yuntao Wang
"""

from layer import *
from modules.densenet_block import *


def dense_net_mp3(input_data, class_num=2, is_bn=True):
    """
    Dense net for MP3 audio
    """
    print("dense_net_mp3: Dense net for MP3 audio")
    print("Network Structure: ")

    conv_1 = conv_layer(input_data, 3, 3, 1, 1, 48, name="conv1", activation_method="None", padding="SAME")

    block1 = dense_block(conv_1, filter_nums=24, layers=6, name="block1")
    trans1 = transition_layer(block1, filter_nums=24, name="trans1")

    block2 = dense_block(trans1, filter_nums=24, layers=12, name="block2")
    trans2 = transition_layer(block2, filter_nums=24, name="trans2")

    block3 = dense_block(trans2, filter_nums=24, layers=24, name="block3")
    trans3 = transition_layer(block3, filter_nums=24, name="trans3")

    block4 = dense_block(trans3, filter_nums=24, layers=48, name="block4")

    bn = batch_normalization(block4, name="BN", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn_shape = bn.get_shape()
    bn_height, bn_width = bn_shape[1], bn_shape[2]
    pool = pool_layer(bn, bn_height, bn_width, bn_height, bn_width, name="global_max_pool", is_max_pool=True)

    logits = fc_layer(pool, class_num, name="out", activation_method=None)

    return logits


def dense_net_mp3_42(input_data, class_num=2, is_bn=True):
    """
    Dense net for MP3 audio
    """
    print("dense_net_mp3: Dense net for MP3 audio")
    print("Network Structure: ")

    conv_1 = conv_layer(input_data, 3, 3, 1, 1, 48, name="conv1", activation_method="None", padding="SAME")

    block1 = dense_block(conv_1, filter_nums=24, layers=6, name="block1")
    trans1 = transition_layer(block1, filter_nums=24, name="trans1")

    block2 = dense_block(trans1, filter_nums=24, layers=12, name="block2")
    trans2 = transition_layer(block2, filter_nums=24, name="trans2")

    block3 = dense_block(trans2, filter_nums=24, layers=24, name="block3")

    bn = batch_normalization(block3, name="BN", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn_shape = bn.get_shape()
    bn_height, bn_width = bn_shape[1], bn_shape[2]
    pool = pool_layer(bn, bn_height, bn_width, bn_height, bn_width, name="global_max_pool", is_max_pool=True)

    logits = fc_layer(pool, class_num, name="out", activation_method=None)

    return logits


def dense_net_mp3_18(input_data, class_num=2, is_bn=True):
    """
    Dense net for MP3 audio
    """
    print("dense_net_mp3: Dense net for MP3 audio")
    print("Network Structure: ")
    # # High Pass Filtering
    # conv0 = rich_hpf_layer(input_data, name="HPFs")
    #
    # # HPF and input data concat
    # conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    # concat_shape = conv0_input_merge.get_shape()
    # print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    conv_1 = conv_layer(input_data, 3, 3, 1, 1, 48, name="conv1", activation_method="None", padding="SAME")

    block1 = dense_block(conv_1, filter_nums=24, layers=6, name="block1")
    trans1 = transition_layer(block1, filter_nums=24, name="trans1")

    block2 = dense_block(trans1, filter_nums=24, layers=8, name="block2")

    bn = batch_normalization(block2, name="BN", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn_shape = bn.get_shape()
    bn_height, bn_width = bn_shape[1], bn_shape[2]
    pool = pool_layer(bn, bn_height, bn_width, bn_height, bn_width, name="global_max_pool", is_max_pool=True)

    logits = fc_layer(pool, class_num, name="out", activation_method=None)

    return logits


def dense_net_mp3_6(input_data, class_num=2, is_bn=True):
    """
    Dense net for MP3 audio
    """
    print("dense_net_mp3: Dense net for MP3 audio")
    print("Network Structure: ")

    conv_1 = conv_layer(input_data, 3, 3, 1, 1, 48, name="conv1", activation_method="None", padding="SAME")

    block1 = dense_block(conv_1, filter_nums=12, layers=6, name="block1")
    trans1 = transition_layer(block1, filter_nums=24, name="trans1")

    bn = batch_normalization(trans1, name="BN", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn_shape = bn.get_shape()
    bn_height, bn_width = bn_shape[1], bn_shape[2]
    pool = pool_layer(bn, bn_height, bn_width, bn_height, bn_width, name="global_max_pool", is_max_pool=True)

    logits = fc_layer(pool, class_num, name="out", activation_method=None)

    return logits
