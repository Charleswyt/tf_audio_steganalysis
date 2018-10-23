#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.10.08
Finished on 2018.10.08
Modified on 2018.10.22

@author: Wang Yuntao
"""

from layer import *
from filters import *


def rhmban(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Network with Rich HPF and Multiscale Block-Aware Split module
    """

    print("RHMBA-Net: Network with Rich HPF and Multiscale Block-Aware Split module")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # down sampling
    block_sampling_2 = block_sampling_layer(input_data, block_size=2, name="block_split2")
    block_sampling_4 = block_sampling_layer(input_data, block_size=4, name="block_split4")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    pool1_4_block_2_merge = tf.concat([pool1_4, block_sampling_2], 3, name="conv1_block_downsampling_merge")
    concat_shape = pool1_4_block_2_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group2
    conv2_1 = conv_layer(pool1_4_block_2_merge, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    pool2_4_block_2_merge = tf.concat([pool2_4, block_sampling_4], 3, name="conv2_block_downsampling_merge")
    concat_shape = pool2_4_block_2_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group3
    conv3_1 = conv_layer(pool2_4_block_2_merge, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
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
    fc6 = fc_layer(pool5_4, 4096, name="fcn6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method=activation_method, is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fcn8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method=activation_method, is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def rhmban1_1(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove high-pass filtering module
    """

    print("RHBA-Net1_1: Remove high-pass filtering module")
    print("Network Structure: ")

    # down sampling
    block_sampling_2 = block_sampling_layer(input_data, block_size=2, name="block_split2")
    block_sampling_4 = block_sampling_layer(input_data, block_size=4, name="block_split4")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    pool1_4_block_2_merge = tf.concat([pool1_4, block_sampling_2], 3, name="conv1_block_downsampling_merge")
    concat_shape = pool1_4_block_2_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group2
    conv2_1 = conv_layer(pool1_4_block_2_merge, 3, 3, 1, 1, 64, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 128, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    pool2_4_block_4_merge = tf.concat([pool2_4, block_sampling_4], 3, name="conv2_block_downsampling_merge")
    concat_shape = pool2_4_block_4_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group3
    conv3_1 = conv_layer(pool2_4_block_4_merge, 3, 3, 1, 1, 128, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 256, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 256, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 512, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 512, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 1024, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # fully conv layer
    fc6 = fc_layer(pool5_4, 4096, name="fcn6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method=activation_method, is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fcn8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method=activation_method, is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def rhmban1_2(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove multiscale block-aware split shortcut module
    """

    print("RHBA-Net1_2: Remove multiscale block-aware split module")
    print("Network Structure: ")

    # High Pass Filter
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
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
    fc6 = fc_layer(pool5_4, 4096, name="fcn6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method=activation_method, is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fcn8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method=activation_method, is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def rhmban1_3(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove 2 x 2 block-aware split shortcut module
    """

    print("RHBA-Net1_3: Remove 2 x 2 block-aware split shortcut module")
    print("Network Structure: ")

    # High Pass Filter
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # down sampling
    block_sampling_4 = block_sampling_layer(input_data, block_size=4, name="block_split4")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    pool2_4_block_4_merge = tf.concat([pool2_4, block_sampling_4], 3, name="conv2_block_downsampling_merge")
    concat_shape = pool2_4_block_4_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group3
    conv3_1 = conv_layer(pool2_4_block_4_merge, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
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
    fc6 = fc_layer(pool5_4, 4096, name="fcn6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method=activation_method, is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fcn8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method=activation_method, is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def rhmban1_4(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove 4 x 4 block-aware split shortcut module
    """

    print("RHBA-Net1_4: Remove 4 x 4 block-aware split shortcut module")
    print("Network Structure: ")

    # High Pass Filter
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # down sampling
    block_sampling_2 = block_sampling_layer(input_data, block_size=2, name="block_split2")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # conv block and block-aware split concat
    pool2_4_block_4_merge = tf.concat([pool1_4, block_sampling_2], 3, name="conv1_block_downsampling_merge")
    concat_shape = merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("block_sampling", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group2
    conv2_1 = conv_layer(pool2_4_block_4_merge, 3, 3, 1, 1, 64, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 128, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 128, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 256, name="conv3_2", activation_method=None, padding=padding)
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
    fc6 = fc_layer(pool5_4, 4096, name="fcn6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method=activation_method, is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fcn8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method=activation_method, is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits
