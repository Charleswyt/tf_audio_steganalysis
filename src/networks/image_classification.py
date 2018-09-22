#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.08.19
Finished on 2018.08.19
Modified on 

@author: Wang Yuntao
"""

from layer import *


"""
    function:
        le_net: Le-Net for image classification
        vgg16 : vgg16 for image classification
        vgg19 : vgg19 for image classification
"""


def le_net(input_data, class_num=10):
    """
    Le-Net for image classification
    """
    print("le_net: Remove the 1x1 conv layers.")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 5, 5, 1, 1, 6, "conv1", "VALID")
    pool1_2 = pool_layer(conv1_1, 2, 2, 2, 2, "pool1_2")

    # Group2
    conv2_1 = conv_layer(pool1_2, 5, 5, 1, 1, 16, "conv2_1", "VALID")
    pool2_2 = pool_layer(conv2_1, 2, 2, 2, 2, "pool2_2")

    # Fully connected layer
    fc4 = fc_layer(pool2_2, 120, "fc4", "relu")
    fc5 = fc_layer(fc4, 84, "fc5", False)
    fc5_drop = dropout(fc5, keep_pro=0.5, name="fc5_drop")
    logits = fc_layer(fc5_drop, class_num, "fc6", False)

    return logits


def vgg16(input_data, class_num=4096):
    """
    vgg16 for image classification
    """
    print("vgg16: Remove the 1x1 conv layers.")
    print("Network Structure: ")

    # vgg16
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 64, "conv1_1")
    conv1_2 = conv_layer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, "pool1_3")

    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 128, "conv2_1")
    conv2_2 = conv_layer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, "pool2_3")

    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 256, "conv3_1")
    conv3_2 = conv_layer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
    conv3_3 = conv_layer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
    pool3_4 = pool_layer(conv3_3, 2, 2, 2, 2, "pool3_4")

    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 512, "conv4_1")
    conv4_2 = conv_layer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
    conv4_3 = conv_layer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
    pool4_4 = pool_layer(conv4_3, 2, 2, 2, 2, "pool4_4")

    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 512, "conv5_1")
    conv5_2 = conv_layer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
    conv5_3 = conv_layer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
    pool5_4 = pool_layer(conv5_3, 2, 2, 2, 2, "pool5_4")

    fc6 = fc_layer(pool5_4, 4096, "fc6")
    fc6_drop = dropout(fc6, keep_pro=0.5, name="fc6_drop")
    fc7 = fc_layer(fc6_drop, 4096, "fc7")
    fc7_drop = dropout(fc7, keep_pro=0.5, name="fc7_drop")
    logits = fc_layer(fc7_drop, class_num, "fc8")

    return logits


def vgg19(input_data, class_num=4096):
    """
    vgg19 for image classification
    """
    print("vgg19: Remove the 1x1 conv layers.")
    print("Network Structure: ")

    # vgg19
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 64, "conv1_1")
    conv1_2 = conv_layer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, "pool1_3")

    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 128, "conv2_1")
    conv2_2 = conv_layer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, "pool2_3")

    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 256, "conv3_1")
    conv3_2 = conv_layer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
    conv3_3 = conv_layer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
    conv3_4 = conv_layer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
    pool3_5 = pool_layer(conv3_4, 2, 2, 2, 2, "pool3_5")

    conv4_1 = conv_layer(pool3_5, 3, 3, 1, 1, 512, "conv4_1")
    conv4_2 = conv_layer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
    conv4_3 = conv_layer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
    conv4_4 = conv_layer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
    pool4_5 = pool_layer(conv4_4, 2, 2, 2, 2, "pool4_5")

    conv5_1 = conv_layer(pool4_5, 3, 3, 1, 1, 512, "conv5_1")
    conv5_2 = conv_layer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
    conv5_3 = conv_layer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
    conv5_4 = conv_layer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
    pool5_5 = pool_layer(conv5_4, 2, 2, 2, 2, "pool5_5")

    fc6 = fc_layer(pool5_5, 4096, "fc6")
    fc6_drop = dropout(fc6, keep_pro=0.5, name="fc6_drop")
    fc7 = fc_layer(fc6_drop, 4096, "fc7")
    fc7_drop = dropout(fc7, keep_pro=0.5, name="fc7_drop")
    logits = fc_layer(fc7_drop, class_num, "fc8")

    return logits
