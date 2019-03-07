#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.08.29
Finished on 2018.08.29
Modified on 2018.11.12

@author: Yuntao Wang
"""

from layer import *
from HPFs.filters import *


def chen_net(input_data, class_num=2):
    """
    Chen-Net for wav audio steganalysis (designer: Bolin Chen)
    """
    print("Chen-Net: Steganalytic Network for wav audio (LSBM embedding)")
    print("Network Structure: ")

    # preprocessing
    conv0 = diff_layer(input_data, is_diff=True, is_abs_diff=False, is_diff_abs=False,
                       order=2, direction="intra", name="diff_intra_2", padding="SAME")

    # Group 1
    conv1_1 = conv_layer(conv0, 1, 5, 1, 1, 1, name="conv1_1", activation_method=None, padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 8, name="conv1_2", activation_method=None, padding="SAME")
    pool1_3 = conv_layer(conv1_2, 1, 3, 1, 2, 8, name="pool1_3", activation_method=None, padding="VALID")

    # Group 2
    conv2_1 = conv_layer(pool1_3, 1, 5, 1, 1, 8, name="conv2_1", activation_method=None, padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 16, name="conv2_2", activation_method=None, padding="SAME")
    pool2_3 = conv_layer(conv2_2, 1, 3, 1, 2, 16, name="pool2_3", activation_method=None, padding="VALID")

    # Group 3
    conv3_1 = conv_layer(pool2_3, 1, 5, 1, 1, 16, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 32, name="conv3_2", activation_method="tanh", padding="SAME")
    pool3_3 = pool_layer(conv3_2, 1, 3, 1, 2, name="pool3_3", is_max_pool=True)

    # Group 4
    conv4_1 = conv_layer(pool3_3, 1, 5, 1, 1, 32, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 64, name="conv4_2", activation_method="tanh", padding="SAME")
    pool4_3 = pool_layer(conv4_2, 1, 3, 1, 2, name="pool4_3", is_max_pool=True)

    # Group 5
    conv5_1 = conv_layer(pool4_3, 1, 5, 1, 1, 64, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 128, name="conv5_2", activation_method="tanh", padding="SAME")
    pool5_3 = pool_layer(conv5_2, 1, 3, 1, 2, name="pool5_3", is_max_pool=True)

    # Group 6
    conv6_1 = conv_layer(pool5_3, 1, 5, 1, 1, 128, name="conv6_1", activation_method="tanh", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 256, name="conv6_2", activation_method="tanh", padding="SAME")
    pool6_3 = pool_layer(conv6_2, 1, 3, 1, 2, name="pool6_3", is_max_pool=True)

    # Group 7
    conv7_1 = conv_layer(pool6_3, 1, 5, 1, 1, 256, name="conv7_1", activation_method="tanh", padding="SAME")
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 512, name="conv7_2", activation_method="tanh", padding="SAME")
    pool7_3 = pool_layer(conv7_2, 1, 250, 1, 250, name="global_average_pooling", is_max_pool=False)

    # Fully connected layer
    logits = fc_layer(pool7_3, class_num, name="fc8", activation_method=None)

    return logits


def wasdn(input_data, class_num=2, is_bn=True):
    """
    The proposed network
    """
    print("WASDN: Wang Audio Steganalysis Deep Network")
    print("Network Structure: ")

    # preprocessing
    conv0 = diff_layer(input_data=input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False,
                       order=2, direction="inter", name="difference", padding="SAME")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method="tanh", padding="SAME")
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method="tanh", padding="SAME")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method="tanh", padding="SAME")
    pool3_3 = pool_layer(conv3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method="tanh", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding="SAME")
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method="tanh", is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=True)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def rhfcn(input_data, class_num=2, is_bn=True):
    """
    Fully CNN for MP3 Steganalysis with Rich High-pass Filtering
    """

    print("RHFCN: Fully CNN for MP3 Steganalysis with Rich High-pass Filtering")
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
    fconv6 = conv_layer(pool5_4, 6, 14, 1, 1, 4096, name="fconv6", activation_method=None, padding="VALID")
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
