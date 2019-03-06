#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.10.08
Finished on 2018.10.08
Modified on 2019.01.08

@author: Yuntao Wang
"""

from layer import *
from HPFs.filters import *


def rhfcn1_1(input_data, class_num=2, is_bn=True):
    """
    Remove High-Pass Filtering Module
    """

    print("RHFCN1_1: Remove High-Pass Filtering Module")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
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
    fconv6 = conv_layer(pool5_4, 6, 14, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
    bn7 = batch_normalization(fconv6, name="BN7", activation_method=activation_method, is_train=is_bn)

    fconv8 = conv_layer(bn7, 1, 1, 1, 1, 512, name="fconv8", activation_method=None, padding="VALID")
    bn9 = batch_normalization(fconv8, name="BN9", activation_method=activation_method, is_train=is_bn)

    fconv10 = conv_layer(bn9, 1, 1, 1, 1, 2, name="fconv10", activation_method=None, padding="VALID")
    bn11 = batch_normalization(fconv10, name="BN11", activation_method=None, is_train=is_bn)

    # Global Max Pooling
    bn11_shape = bn11.get_shape()
    bn11_height, bn11_width = bn11_shape[1], bn11_shape[2]
    pool11 = pool_layer(bn11, bn11_height, bn11_width, bn11_height, bn11_width, name="global_max", is_max_pool=True)

    logits = tf.reshape(pool11, [-1, class_num])

    return logits


def rhfcn1_2(input_data, class_num=2, is_bn=True):
    """
    Quit replacing fully connected layers with fully convolutional layer
    """

    print("RHFCN1_2: Quit replacing fully connected layers with fully convolutional layer")
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

    # Fully connected layer
    fc6 = fc_layer(pool5_4, 4096, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits


def rhfcn1_3(input_data, class_num=2, is_bn=True):
    """
    Remove rich HPF module and quit removing fc layers at the same time
    """

    print("RHFCN1_3: Remove rich HPF module and quit removing fc layers at the same time")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
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

    # Fully connected layer
    fc6 = fc_layer(pool5_4, 4096, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits
