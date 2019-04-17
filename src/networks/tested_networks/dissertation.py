#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019.02.21
Finished on 2019.02.21
Modified on

@author: Yuntao Wang
"""

from modules.inception import *


def chap4(input_data, class_num=2, is_bn=True):
    """
    Network in Chapter 4
    """

    print("Network in Chapter 4")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 3, 3, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 3, 3, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 3, 3, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # fully conv layer
    fconv6 = conv_layer(pool5_4, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="tanh", is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="tanh", is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_1(input_data, class_num=2, is_bn=True):
    """
    Remove HPFs
    """

    print("Remove Rich High Pass Filters")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # fully conv layer
    fconv6 = conv_layer(pool5_4, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="tanh", is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="tanh", is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_2(input_data, class_num=2, is_bn=True):
    """
    Remove the shortcut of QMDCT coefficients
    """

    print("Remove the shortcut of QMDCT coefficients")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # Group1
    conv1_1 = conv_layer(conv0, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # fully conv layer
    fconv6 = conv_layer(pool5_4, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="tanh", is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="tanh", is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_3(input_data, class_num=2):
    """
    Remove BN layers
    """

    print("Remove BN layers")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    pool3_3 = pool_layer(conv3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    pool4_3 = pool_layer(conv4_2, 2, 2, 2, 2, name="pool4_3", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_3, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    pool5_3 = pool_layer(conv5_2, 2, 2, 2, 2, name="pool5_3", is_max_pool=True)

    # fully conv layer
    fconv6 = conv_layer(pool5_3, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    fconv7 = conv_layer(fconv6, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    fconv8 = conv_layer(fconv7, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")

    # Global Max Pooling
    fconv8_shape = fconv8.get_shape()
    fconv8_height, fconv8_width = fconv8_shape[1], fconv8_shape[2]
    pool9 = pool_layer(fconv8, fconv8_height, fconv8_width, fconv8_height, fconv8_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool9, [-1, class_num])

    return logits


def chap4_4(input_data, class_num=2, is_bn=True):
    """
    Remove 1x1 Convolutional kernel
    """

    print("Remove 1x1 Convolutional kernel")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    bn1_2 = batch_normalization(conv1_1, name="BN1_2", activation_method="tanh", is_train=is_bn)
    pool1_3 = pool_layer(bn1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method="tanh", is_train=is_bn)
    pool2_3 = pool_layer(bn2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    bn3_2 = batch_normalization(conv3_1, name="BN3_2", activation_method="tanh", is_train=is_bn)
    pool3_3 = pool_layer(bn3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    bn4_2 = batch_normalization(conv4_1, name="BN4_2", activation_method="tanh", is_train=is_bn)
    pool4_3 = pool_layer(bn4_2, 2, 2, 2, 2, name="pool4_3", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_3, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    bn5_2 = batch_normalization(conv5_1, name="BN5_2", activation_method="tanh", is_train=is_bn)
    pool5_3 = pool_layer(bn5_2, 2, 2, 2, 2, name="pool5_3", is_max_pool=True)

    # fully conv layer
    fconv6 = conv_layer(pool5_3, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="tanh", is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="tanh", is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_5(input_data, class_num=2, is_bn=True):
    """
    Average pooling
    """

    print("Average pooling")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=False)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=False)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=False)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=False)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=False)

    # fully conv layer
    fconv6 = conv_layer(pool5_4, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="tanh", is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="tanh", is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_6(input_data, class_num=2, is_bn=True):
    """
    5 x 5 convolutional kernel is used
    """

    print("5 x 5 convolutional kernel is used")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 5, 5, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=False)

    # Group2
    conv2_1 = conv_layer(pool1_4, 5, 5, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=False)

    # Group3
    conv3_1 = conv_layer(pool2_4, 5, 5, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=False)

    # Group4
    conv4_1 = conv_layer(pool3_4, 5, 5, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=False)

    # Group5
    conv5_1 = conv_layer(pool4_4, 5, 5, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=False)

    # fully conv layer
    fconv6 = conv_layer(pool5_4, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="tanh", is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="tanh", is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_7(input_data, class_num=2, is_bn=True):
    """
    Type of activation function: ReLu
    """

    print("Network of Chapter4_5")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="relu", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="relu", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=False)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="relu", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="relu", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=False)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="relu", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="relu", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=False)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="relu", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="relu", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=False)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="relu", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="relu", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=False)

    # fully conv layer
    fconv6 = conv_layer(pool5_4, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="relu", is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="relu", is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="relu", is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_8(input_data, class_num=2, is_bn=True):
    """
    Type of activation function: leaky relu
    """

    print("Type of activation function: leaky relu")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="leakrelu", alpha=0.2, padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="leakrelu", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="leakrelu", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="leakrelu", alpha=0.2, padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="leakrelu", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="leakrelu", alpha=0.2, padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="leakrelu", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="leakrelu", alpha=0.2, padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="leakrelu", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # fully conv layer
    fconv6 = conv_layer(pool5_4, 6, 12, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method="leakrelu", is_train=is_bn)
    bn7 = activation_layer(bn7, activation_method="leakrelu", alpha=0.2)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method="leakrelu", is_train=is_bn)
    bn9 = activation_layer(bn9, activation_method="leakrelu", alpha=0.2)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method="leakrelu", is_train=is_bn)
    bn11 = activation_layer(bn11, activation_method="leakrelu", alpha=0.2)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def chap4_9(input_data, class_num=2, is_bn=True):
    """
    Quit replacing fc layers with conv layer
    """

    print("Quit replacing fc layers with conv layer")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # fc layers
    fc6 = fc_layer(pool5_4, 4096, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def chap4_10(input_data, class_num=2, is_bn=True):
    """
    Deepen the network
    """

    print("Deepen the network")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 256, name="conv6_1", activation_method="tanh", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding="SAME")
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method="tanh", is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=True)

    # fully conv layer
    fconv7 = conv_layer(pool6_4, 3, 6, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn8 = batch_normalization(fconv7, name="BN7", activation_method="tanh", is_train=is_bn)

    fconv9 = conv_layer(bn8, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn10 = batch_normalization(fconv9, name="BN9", activation_method="tanh", is_train=is_bn)

    fconv11 = conv_layer(bn10, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn12 = batch_normalization(fconv11, name="BN11", activation_method="tanh", is_train=is_bn)

    # Global Max Pooling
    bn12_shape = bn12.get_shape()
    bn12_height, bn12_width = bn12_shape[1], bn12_shape[2]
    pool13 = pool_layer(bn12, bn12_height, bn12_width, bn12_height, bn12_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool13, [-1, class_num])

    return logits


def mfcc_net(input_data, class_num=2, is_bn=True):
    """
    Network with the input data of mfcc
    """

    print("Network with the input data of mfcc")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method="relu", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="relu", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="relu", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="relu", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="relu", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="relu", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="relu", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="relu", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # fc layers
    fc6 = fc_layer(pool4_4, 1024, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def mfcc_net1(input_data, class_num=2, is_bn=True):
    """
    Network with the input data of mfcc
    """

    print("Network with the input data of mfcc")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding="SAME")
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method="tanh", is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding="SAME")
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method="tanh", is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # fc layers
    fc6 = fc_layer(pool4_4, 1024, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def mfcc_net2(input_data, class_num=2, is_bn=True):
    """
    Network with the input data of mfcc
    """

    print("Network with the input data of mfcc")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method=None, padding="SAME")
    bn1_2 = batch_normalization(conv1_1, name="BN1_2", activation_method="tanh", is_train=is_bn)
    conv1_2 = conv_layer(bn1_2, 1, 1, 1, 1, 32, name="conv1_3", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_4", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_5", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=None, padding="SAME")
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method="tanh", is_train=is_bn)
    conv2_3 = conv_layer(bn2_2, 1, 1, 1, 1, 64, name="conv2_3", activation_method=None, padding="SAME")
    bn2_4 = batch_normalization(conv2_3, name="BN2_4", activation_method="tanh", is_train=is_bn)
    pool2_5 = pool_layer(bn2_4, 2, 2, 2, 2, name="pool2_5", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_5, 3, 3, 1, 1, 64, name="conv3_1", activation_method=None, padding="SAME")
    bn3_2 = batch_normalization(conv3_1, name="BN3_2", activation_method="tanh", is_train=is_bn)
    conv3_3 = conv_layer(bn3_2, 1, 1, 1, 1, 128, name="conv3_3", activation_method=None, padding="SAME")
    bn3_4 = batch_normalization(conv3_3, name="BN3_4", activation_method="tanh", is_train=is_bn)
    pool3_5 = pool_layer(bn3_4, 2, 2, 2, 2, name="pool3_5", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_5, 3, 3, 1, 1, 128, name="conv4_1", activation_method=None, padding="SAME")
    bn4_2 = batch_normalization(conv4_1, name="BN4_2", activation_method="tanh", is_train=is_bn)
    conv4_3 = conv_layer(bn4_2, 1, 1, 1, 1, 256, name="conv4_3", activation_method=None, padding="SAME")
    bn4_4 = batch_normalization(conv4_3, name="BN4_4", activation_method="tanh", is_train=is_bn)
    pool4_5 = pool_layer(bn4_4, 2, 2, 2, 2, name="pool4_5", is_max_pool=True)

    # fc layers
    fc6 = fc_layer(pool4_5, 1024, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def mfcc_net3(input_data, class_num=2, is_bn=True):
    """
    Network with the input data of mfcc
    """

    print("Network with the input data of mfcc")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method=None, padding="SAME")
    bn1_2 = batch_normalization(conv1_1, name="BN1_2", activation_method="relu", is_train=is_bn)
    conv1_2 = conv_layer(bn1_2, 1, 1, 1, 1, 32, name="conv1_3", activation_method=None, padding="SAME")
    bn1_3 = batch_normalization(conv1_2, name="BN1_4", activation_method="relu", is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_5", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=None, padding="SAME")
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method="relu", is_train=is_bn)
    conv2_3 = conv_layer(bn2_2, 1, 1, 1, 1, 64, name="conv2_3", activation_method=None, padding="SAME")
    bn2_4 = batch_normalization(conv2_3, name="BN2_4", activation_method="relu", is_train=is_bn)
    pool2_5 = pool_layer(bn2_4, 2, 2, 2, 2, name="pool2_5", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_5, 3, 3, 1, 1, 64, name="conv3_1", activation_method=None, padding="SAME")
    bn3_2 = batch_normalization(conv3_1, name="BN3_2", activation_method="relu", is_train=is_bn)
    conv3_3 = conv_layer(bn3_2, 1, 1, 1, 1, 128, name="conv3_3", activation_method=None, padding="SAME")
    bn3_4 = batch_normalization(conv3_3, name="BN3_4", activation_method="relu", is_train=is_bn)
    pool3_5 = pool_layer(bn3_4, 2, 2, 2, 2, name="pool3_5", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_5, 3, 3, 1, 1, 128, name="conv4_1", activation_method=None, padding="SAME")
    bn4_2 = batch_normalization(conv4_1, name="BN4_2", activation_method="relu", is_train=is_bn)
    conv4_3 = conv_layer(bn4_2, 1, 1, 1, 1, 256, name="conv4_3", activation_method=None, padding="SAME")
    bn4_4 = batch_normalization(conv4_3, name="BN4_4", activation_method="relu", is_train=is_bn)
    pool4_5 = pool_layer(bn4_4, 2, 2, 2, 2, name="pool4_5", is_max_pool=True)

    # fc layers
    fc6 = fc_layer(pool4_5, 1024, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="relu", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="relu", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def google_net1(input_data, class_num=2, is_bn=True):
    """
    Google Net with inception V1
    """

    print("Google Net with inception V1")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group 1
    conv1_1 = inception_v1(conv0_input_merge, filter_num=8, name="conv1_1", is_max_pool=True, is_bn=is_bn)
    pool1_2 = pool_layer(conv1_1, 3, 3, 2, 2, name="pool1_2")

    # Group 2
    conv2_1 = inception_v1(pool1_2, filter_num=16, name="conv2_1", is_max_pool=True, is_bn=is_bn)
    pool2_2 = pool_layer(conv2_1, 3, 3, 2, 2, name="pool2_2")

    # Group 3
    conv3_1 = inception_v1(pool2_2, filter_num=32, name="conv3_1", is_max_pool=True, is_bn=is_bn)
    pool3_2 = pool_layer(conv3_1, 3, 3, 2, 2, name="pool3_2")

    # Group 4
    conv4_1 = inception_v1(pool3_2, filter_num=64, name="conv4_1", is_max_pool=True, is_bn=is_bn)
    pool4_2 = pool_layer(conv4_1, 3, 3, 2, 2, name="pool4_2")

    # Group 5
    conv5_1 = inception_v1(pool4_2, filter_num=128, name="conv5_1", is_max_pool=True, is_bn=is_bn)
    pool5_2 = pool_layer(conv5_1, 3, 3, 2, 2, name="pool5_2")

    # Group 6
    conv6_1 = inception_v1(pool5_2, filter_num=256, name="conv6_1", is_max_pool=True, is_bn=is_bn)
    pool6_2 = global_pool(conv6_1, name="global_pool6_2")

    # fc layers
    fc6 = fc_layer(pool6_2, 128, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn7, class_num, name="fc8", activation_method=None)

    return logits


def google_net2(input_data, class_num=2, is_bn=True):
    """
    Google Net with inception V2
    """

    print("Google Net with inception V2")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group 1
    conv1_1 = inception_v2(conv0_input_merge, 8, "conv1_1", is_max_pool=True, is_bn=is_bn)
    pool1_2 = pool_layer(conv1_1, 3, 3, 2, 2, name="pool1_2")

    # Group 2
    conv2_1 = inception_v2(pool1_2, 16, "conv2_1", is_max_pool=True, is_bn=is_bn)
    pool2_2 = pool_layer(conv2_1, 3, 3, 2, 2, name="pool2_2")

    # Group 3
    conv3_1 = inception_v2(pool2_2, 32, "conv3_1", is_max_pool=True, is_bn=is_bn)
    pool3_2 = pool_layer(conv3_1, 3, 3, 2, 2, name="pool3_2")

    # Group 4
    conv4_1 = inception_v2(pool3_2, 64, "conv4_1", is_max_pool=True, is_bn=is_bn)
    pool4_2 = pool_layer(conv4_1, 3, 3, 2, 2, name="pool4_2")

    # Group 5
    conv5_1 = inception_v2(pool4_2, 128, "conv5_1", is_max_pool=True, is_bn=is_bn)
    pool5_2 = pool_layer(conv5_1, 3, 3, 2, 2, name="pool5_2")

    # Group 6
    conv6_1 = inception_v2(pool5_2, 256, "conv6_1", is_max_pool=True, is_bn=is_bn)
    pool6_2 = global_pool(conv6_1, name="global_pool6_2")

    # fc layers
    fc6 = fc_layer(pool6_2, 128, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn7, class_num, name="fc8", activation_method=None)

    return logits


def google_net3(input_data, class_num=2, is_bn=True):
    """
    Google Net with inception V3
    """

    print("Google Net with inception V3")
    print("Network Structure: ")

    # High Pass Filtering
    conv0 = rich_hpf_layer(input_data, name="HPFs")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group 1
    conv1_1 = inception_v3(conv0_input_merge, 8, "conv1_1", is_max_pool=True, is_bn=is_bn)
    pool1_2 = pool_layer(conv1_1, 3, 3, 2, 2, name="pool1_2")

    # Group 2
    conv2_1 = inception_v3(pool1_2, 16, "conv2_1", is_max_pool=True, is_bn=is_bn)
    pool2_2 = pool_layer(conv2_1, 3, 3, 2, 2, name="pool2_2")

    # Group 3
    conv3_1 = inception_v3(pool2_2, 32, "conv3_1", is_max_pool=True, is_bn=is_bn)
    pool3_2 = pool_layer(conv3_1, 3, 3, 2, 2, name="pool3_2")

    # Group 4
    conv4_1 = inception_v3(pool3_2, 64, "conv4_1", is_max_pool=True, is_bn=is_bn)
    pool4_2 = pool_layer(conv4_1, 3, 3, 2, 2, name="pool4_2")

    # Group 5
    conv5_1 = inception_v3(pool4_2, 128, "conv5_1", is_max_pool=True, is_bn=is_bn)
    pool5_2 = pool_layer(conv5_1, 3, 3, 2, 2, name="pool5_2")

    # Group 6
    conv6_1 = inception_v3(pool5_2, 256, "conv6_1", is_max_pool=True, is_bn=is_bn)
    pool6_2 = global_pool(conv6_1, name="global_pool6_2")

    # fc layers
    fc6 = fc_layer(pool6_2, 128, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn7, class_num, name="fc8", activation_method=None)

    return logits
