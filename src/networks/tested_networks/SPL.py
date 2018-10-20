#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.09.25
Finished on 2018.09.25
Modified on 2018.10.10

@author: Wang Yuntao
"""

from layer import *
from filters import *

"""
    function:
        all versions are modified on the basis of tpan
        
        tbafcn1_1 : Remove the truncation module
        tbafcn1_2 : Remove the phase-split module
        tbafcn1_3 : Remove the left-crop module
        tbafcn1_4 : Quit replacing fully connected layers with fully convolutional layer
"""


def tbafcn1_1(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove the truncation module
    """

    print("TPA-Net1_1: Remove the truncation module")
    print("Network Structure: ")

    # downsampling
    block_sampling = block_sampling_layer(input_data, block_size=2, name="block_sampling")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    conv1_block_sampling = tf.concat([pool1_4, block_sampling], 3, "block_sampling")
    concat_shape = conv1_block_sampling.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group2
    conv2_1 = conv_layer(conv1_block_sampling, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # fully conv layer
    fcn6 = fconv_layer(pool5_4, 4096, name="fc6")
    bn7 = batch_normalization(fcn6, name="BN7", activation_method=activation_method, is_train=is_bn)
    fcn8 = fconv_layer(bn7, 512, name="fc8")
    bn9 = batch_normalization(fcn8, name="BN9", activation_method=activation_method, is_train=is_bn)
    fcn10 = fconv_layer(bn9, class_num, name="fcn11")

    logits = tf.reshape(fcn10, [-1, class_num])

    return logits


def tbafcn1_2(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove the block-split shortcut module
    """

    print("TPA-Net1_2: Remove the block-split shortcut module")
    print("Network Structure: ")

    # preprocessing
    data_trunc = truncation_layer(input_data, min_value=-8, max_value=8, name="truncation")

    # Group1
    conv1_1 = conv_layer(data_trunc, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # fully conv layer
    fcn6 = fconv_layer(pool5_4, 4096, name="fc6")
    bn7 = batch_normalization(fcn6, name="BN7", activation_method=activation_method, is_train=is_bn)
    fcn8 = fconv_layer(bn7, 512, name="fc8")
    bn9 = batch_normalization(fcn8, name="BN9", activation_method=activation_method, is_train=is_bn)
    fcn10 = fconv_layer(bn9, class_num, name="fcn11")

    logits = tf.reshape(fcn10, [-1, class_num])

    return logits


def tbafcn1_3(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Quit replacing fully connected layers with fully convolutional layer
    """
    print("TPA-Net1_3: Quit replacing fully connected layers with fully convolutional layer")
    print("Network Structure: ")

    # preprocessing
    data_trunc = truncation_layer(input_data, min_value=-8, max_value=8, name="truncation")

    # downsampling
    block_sampling = block_sampling_layer(data_trunc, block_size=2, name="block_sampling")

    # Group1
    conv1_1 = conv_layer(data_trunc, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    conv1_block_sampling = tf.concat([pool1_4, block_sampling], 3, "block_sampling")
    concat_shape = conv1_block_sampling.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group2
    conv2_1 = conv_layer(conv1_block_sampling, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc6 = fc_layer(pool5_4, 4096, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits
