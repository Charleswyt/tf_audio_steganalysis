#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.08.29
Finished on 2018.08.29
Modified on 2018.11.12

@author: Yuntao Wang
"""

from layer import *

"""
    function:
        s_xu_net: Xu-Net for spatial image steganalysis
        ye_net: Ye-Net for spatial image steganalysis
        yedroudj_net: Yedroudj-Net for spatial image steganalysis
        j_xu_net: Xu-Net for JPEG image steganalysis
"""


def s_xu_net(input_data, class_num=2, is_bn=True):
    """
    Xu-Net for spatial image steganalysis
    """
    print("S_Xu-Net: Spatial Image Steganalytic Network.")
    print("Network Structure: ")

    # Group 0
    conv0 = static_conv_layer(input_data, kv_kernel_generator(), 1, 1, "HPF")

    # Group 1
    conv1_1 = conv_layer(conv0, 5, 5, 1, 1, 8, "conv1_1", activation_method=None, init_method="gaussian", bias_term=False)
    conv1_2 = tf.abs(conv1_1, "conv1_abs")
    conv1_3 = batch_normalization(conv1_2, name="conv1_BN", activation_method="tanh", is_train=is_bn)
    pool1_4 = pool_layer(conv1_3, 5, 5, 2, 2, "pool1_4", is_max_pool=False)

    # Group 2
    conv2_1 = conv_layer(pool1_4, 5, 5, 1, 1, 16, "conv2_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method="tanh", is_train=is_bn)
    pool2_3 = pool_layer(bn2_2, 5, 5, 2, 2, name="pool2_3", is_max_pool=False)

    # Group 3
    conv3_1 = conv_layer(pool2_3, 1, 1, 1, 1, 32, "conv3_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn3_2 = batch_normalization(conv3_1, activation_method="relu", name="BN3_2", is_train=is_bn)
    pool3_3 = pool_layer(bn3_2, 5, 5, 2, 2, "pool3_3", is_max_pool=False)

    # Group 4
    conv4_1 = conv_layer(pool3_3, 1, 1, 1, 1, 64, "conv4_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn4_2 = batch_normalization(conv4_1, activation_method="relu", name="BN4_2", is_train=is_bn)
    pool4_3 = pool_layer(bn4_2, 5, 5, 2, 2, "pool4_3", is_max_pool=False)

    # Group 5
    conv5_1 = conv_layer(pool4_3, 1, 1, 1, 1, 128, "conv5_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn5_2 = batch_normalization(conv5_1, activation_method="relu", name="BN5_2", is_train=is_bn)
    pool5_3 = pool_layer(bn5_2, 32, 32, 2, 2, "pool5_3", is_max_pool=False)

    # Fully connected layer
    logits = fc_layer(pool5_3, class_num, "fc6", activation_method=None)

    return logits


def ye_net(input_data, class_num=2):
    """
    Ye-Net for spatial image steganalysis
    """
    print("Ye-Net: Spatial Image Steganalytic Network.")
    print("Network Structure: ")

    # Group 1
    conv1_1 = conv_layer(input_data, 5, 5, 1, 1, 30, name="conv1_1", activation_method=None, padding="SAME")
    conv1_1 = tf.clip_by_value(conv1_1, clip_value_min=-3, clip_value_max=3, name="TLU")

    # Group 2
    conv2_1 = conv_layer(conv1_1, 3, 3, 1, 1, 30, name="conv2_1", activation_method="relu")

    # Group 3
    conv3_1 = conv_layer(conv2_1, 3, 3, 1, 1, 30, name="conv3_1", activation_method="relu")

    # Group 4
    conv4_1 = conv_layer(conv3_1, 3, 3, 1, 1, 30, name="conv4_1", activation_method="relu")
    pool4_2 = pool_layer(conv4_1, 2, 2, 2, 2, name="pool4_2", is_max_pool=False)

    # Group 5
    conv5_1 = conv_layer(pool4_2, 5, 5, 1, 1, 32, name="conv5_1", activation_method="relu")
    pool5_2 = pool_layer(conv5_1, 3, 3, 2, 2, name="pool5_2", is_max_pool=False)

    # Group 6
    conv6_1 = conv_layer(pool5_2, 5, 5, 1, 1, 32, name="conv6_1", activation_method="relu")
    pool6_2 = pool_layer(conv6_1, 3, 3, 2, 2, name="pool6_2", is_max_pool=False)

    # Group 7
    conv7_1 = conv_layer(pool6_2, 5, 5, 1, 1, 32, name="conv7_1", activation_method="relu")
    pool7_2 = pool_layer(conv7_1, 3, 3, 2, 2, name="pool7_2", is_max_pool=False)

    # Group 8
    conv8_1 = conv_layer(pool7_2, 3, 3, 1, 1, 16, name="conv8_1", activation_method="relu")

    # Group 9
    conv9_1 = conv_layer(conv8_1, 3, 3, 3, 3, 16, name="conv9_1", activation_method="relu")

    # Fully connected layer
    logits = fc_layer(conv9_1, class_num, name="fc10", activation_method=None)

    return logits


def yedroudj_net(input_data, class_num=2, is_bn=True):
    """
    Yedroudj-Net for spatial image steganalysis
    """
    print("Yedroudj-Net: Spatial Image Steganalytic Network.")
    print("Network Structure: ")

    # Group 0
    conv0 = static_conv_layer(input_data, srm_kernels_generator(), 1, 1, "SRMs", padding="SAME")

    # Group 1
    conv1_1 = conv_layer(conv0, 5, 5, 1, 1, 30, "conv1_1", padding="SAME", activation_method=None)
    conv1_1_abs = tf.abs(conv1_1, name="conv1_1_abs")
    bn1_2 = batch_normalization(conv1_1_abs, name="BN1_2", is_train=is_bn)
    trunc1_3 = tf.clip_by_value(bn1_2, clip_value_min=-3, clip_value_max=3, name="Trunc1_3")

    # Group 2
    conv2_1 = conv_layer(trunc1_3, 5, 5, 1, 1, 30, "conv2_1", padding="SAME", activation_method=None)
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", is_train=is_bn)
    trunc2_3 = tf.clip_by_value(bn2_2, clip_value_min=-2, clip_value_max=2, name="Trunc2_3")
    pool2_4 = pool_layer(trunc2_3, 5, 5, 2, 2, name="pool2_4")

    # Group 3
    conv3_1 = conv_layer(pool2_4, 5, 5, 1, 1, 32, "conv3_1", padding="SAME", activation_method=None)
    bn3_2 = batch_normalization(conv3_1, name="BN3_2", activation_method="relu", is_train=is_bn)
    pool3_3 = pool_layer(bn3_2, 5, 5, 2, 2, name="pool3_3", is_max_pool=False)

    # Group 4
    conv4_1 = conv_layer(pool3_3, 5, 5, 1, 1, 64, "conv5_1", padding="SAME", activation_method=None)
    bn4_2 = batch_normalization(conv4_1, name="BN5_2", activation_method="relu", is_train=is_bn)
    pool4_3 = pool_layer(bn4_2, 5, 5, 2, 2, name="pool5_3", is_max_pool=False)

    # Group 5
    conv5_1 = conv_layer(pool4_3, 5, 5, 1, 1, 128, "conv6_1", padding="SAME", activation_method=None)
    bn5_2 = batch_normalization(conv5_1, name="BN6_2", activation_method="relu", is_train=is_bn)
    pool5_3 = pool_layer(bn5_2, 5, 5, 2, 2, name="pool6_3", is_max_pool=False)

    # Fully connected layer
    fc6 = fc_layer(pool5_3, 256, name="fc6", activation_method="relu")
    fc7 = fc_layer(fc6, 1024, name="fc7", activation_method="relu")
    logits = fc_layer(fc7, class_num, name="fc8", activation_method=None)

    return logits


def j_xu_net(input_data, class_num=2, is_bn=True):
    """
    Xu-Net for jpeg image steganalysis
    """
    print("J_Xu-Net: JPEG Image Steganalytic Network.")
    print("Network Structure: ")

    # Group
    dct_to_spatial = static_conv_layer(input_data, dct_kernel_generator(4), 1, 1, name="dct_4", padding="SAME")

