#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.08.29
Finished on  2018.08.29
Modified on

@author: Wang Yuntao
"""

from layer import *

"""
    function:
        stegshi: Xu-Net for image steganalysis
"""


def stegshi(input_data, class_num=2, is_bn=True, is_max_pool=False):
    """
        Xu-Net for image steganalysis (图像隐写分析, Xu-Net)
    """
    print("stegshi: Remove the 1x1 conv layers.")
    print("Network Structure: ")

    # Group 0
    conv0 = static_conv_layer(input_data, kv_kernel, 1, 1, "conv0")

    # Group 1
    conv1_1 = conv_layer(conv0, 5, 5, 1, 1, 8, "conv1_1", activation_method=None, init_method="gaussian", bias_term=False)
    conv1_2 = tf.abs(conv1_1, "conv1_abs")
    conv1_3 = batch_normalization(conv1_2, name="conv1_BN", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(conv1_3, 5, 5, 2, 2, "pool1_4", is_max_pool=is_max_pool)

    # Group 2
    conv2_1 = conv_layer(pool1_4, 5, 5, 1, 1, 16, "conv2_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method="tanh", is_train=is_bn)
    pool2_3 = pool_layer(bn2_2, 5, 5, 2, 2, name="pool2_3", is_max_pool=is_max_pool)

    # Group 3
    conv3_1 = conv_layer(pool2_3, 1, 1, 1, 1, 32, "conv3_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn3_2 = batch_normalization(conv3_1, activation_method="relu", name="BN3_2", is_train=is_bn)
    pool3_3 = pool_layer(bn3_2, 5, 5, 2, 2, "pool3_3", is_max_pool=is_max_pool)

    # Group 4
    conv4_1 = conv_layer(pool3_3, 1, 1, 1, 1, 64, "conv4_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn4_2 = batch_normalization(conv4_1, activation_method="relu", name="BN4_2", is_train=is_bn)
    pool4_3 = pool_layer(bn4_2, 7, 7, 2, 2, "pool4_3", is_max_pool=is_max_pool)

    # Group 5
    conv5_1 = conv_layer(pool4_3, 1, 1, 1, 1, 128, "conv5_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn5_2 = batch_normalization(conv5_1, activation_method="relu", name="BN5_2", is_train=is_bn)
    pool5_3 = pool_layer(bn5_2, 7, 7, 2, 2, "pool5_3", is_max_pool=is_max_pool)

    # Group 6
    conv6_1 = conv_layer(pool5_3, 1, 1, 1, 1, 256, "conv6_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn6_2 = batch_normalization(conv6_1, activation_method="relu", name="BN6_2", is_train=is_bn)
    pool6_3 = pool_layer(bn6_2, 16, 16, 16, 16, "pool6_3", is_max_pool=is_max_pool)

    # Fully connected layer
    logits = fc_layer(pool6_3, class_num, "fc7", activation_method=None)

    return logits