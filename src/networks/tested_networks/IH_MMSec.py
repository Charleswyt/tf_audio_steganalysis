#!/usr/bin/env python3
# -*- coding: utf-8 -*-33

"""
Created on 2018.09.25
Finished on 2018.09.25
Modified on 2019.01.07

@author: Yuntao Wang
"""

from layer import *
from filters import *


def wasdn1_1(input_data, class_num=2):
    """
    Remove all batch normalization layers
    """
    print("WASDN1_1: Remove the BN layer")
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
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method="tanh", padding="SAME")
    pool4_3 = pool_layer(conv4_2, 2, 2, 2, 2, name="pool4_3", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_3, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method="tanh", padding="SAME")
    pool5_3 = pool_layer(conv5_2, 2, 2, 2, 2, name="pool5_3", is_max_pool=True)

    # Group6
    conv6_1 = conv_layer(pool5_3, 3, 3, 1, 1, 512, name="conv6_1", activation_method="tanh", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method="tanh", padding="SAME")
    pool6_3 = pool_layer(conv6_2, 2, 2, 2, 2, name="pool6_3", is_max_pool=True)

    # Fully connected layer
    fc7 = fc_layer(pool6_3, 4096, name="fc7", activation_method="tanh")
    fc8 = fc_layer(fc7, 512, name="fc8", activation_method="tanh")

    logits = fc_layer(fc8, class_num, name="fc9", activation_method=None)

    return logits


def wasdn1_2(input_data, class_num=2, is_bn=True):
    """
    Average pooling layer is used for subsampling
    """
    print("WASDN1_2: replace the max pooling layer with the average pooling layer")
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
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=False)

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method="tanh", padding="SAME")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=False)

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method="tanh", padding="SAME")
    pool3_3 = pool_layer(conv3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=False)

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=False)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=False)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method="tanh", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding="SAME")
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method="tanh", is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=False)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def wasdn1_3(input_data, class_num=2, is_bn=True):
    """
    Convolutional layer with stride 2 is used for subsampling
    """
    print("WASDN1_3: replace the max pooling layer with the convolutional pooling layer")
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
    conv_pool_1_3 = conv_layer(conv1_2, 3, 3, 2, 2, 32, name="conv_pool_1_3", activation_method="None", padding="VALID")

    # Group2
    conv2_1 = conv_layer(conv_pool_1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method="tanh", padding="SAME")
    conv_pool_2_3 = conv_layer(conv2_2, 3, 3, 2, 2, 64, name="conv_pool_2_3", activation_method="None", padding="VALID")

    # Group3
    conv3_1 = conv_layer(conv_pool_2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, "conv3_2", activation_method="tanh", padding="SAME")
    conv_pool_3_3 = conv_layer(conv3_2, 3, 3, 2, 2, 128, name="conv_pool_3_3", activation_method="None", padding="VALID")

    # Group4
    conv4_1 = conv_layer(conv_pool_3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    conv_pool_4_4 = conv_layer(bn4_3, 3, 3, 2, 2, 256, name="conv_pool_4_4", activation_method="None", padding="VALID")

    # Group5
    conv5_1 = conv_layer(conv_pool_4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    conv_pool_5_4 = conv_layer(bn5_3, 3, 3, 2, 2, 512, name="conv_pool_5_4", activation_method="None", padding="VALID")

    # Group6
    conv6_1 = conv_layer(conv_pool_5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method="tanh", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding="SAME")
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method="tanh", is_train=is_bn)
    conv_pool_6_4 = conv_layer(bn6_3, 3, 3, 2, 2, 1024, name="conv_pool_6_4", activation_method="None", padding="VALID")

    # Fully connected layer
    fc7 = fc_layer(conv_pool_6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def wasdn1_4(input_data, class_num=2, is_bn=True):
    """
    Replace the convolutional kernel with 5x5 kernel
    """
    print("WASDN1_4: replace the 3 x 3 kernel with the 5 x 5 kernel")
    print("Network Structure: ")

    # preprocessing
    conv0 = diff_layer(input_data=input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False,
                       order=2, direction="inter", name="difference", padding="SAME")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 5, 5, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method="tanh", padding="SAME")
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_3, 5, 5, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method="tanh", padding="SAME")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_3, 5, 5, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method="tanh", padding="SAME")
    pool3_3 = pool_layer(conv3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_3, 5, 5, 1, 1, 128, name="conv4_1", activation_method="tanh", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="tanh", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 5, 5, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="tanh", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # Group6
    conv6_1 = conv_layer(pool5_4, 5, 5, 1, 1, 512, name="conv6_1", activation_method="tanh", padding="SAME")
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


def wasdn1_5(input_data, class_num=2, is_bn=True):
    """
    ReLu is used as the activation function
    """
    print("WASDN1_5: use relu as the activation function")
    print("Network Structure: ")

    # preprocessing
    conv0 = diff_layer(input_data=input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False,
                       order=2, direction="inter", name="difference", padding="SAME")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="relu", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method="relu", padding="SAME")
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method="relu", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method="relu", padding="SAME")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method="relu", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method="relu", padding="SAME")
    pool3_3 = pool_layer(conv3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method="relu", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="relu", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="relu", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="relu", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method="relu", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding="SAME")
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method="relu", is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=True)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="relu", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="relu", is_train=is_bn)

    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def wasdn1_6(input_data, class_num=2, is_bn=True):
    """
    Leaky-ReLu is used as the activation function
    """
    print("WASDN1_6: Use leakrelu as the activation function")
    print("Network Structure: ")

    # preprocessing
    conv0 = diff_layer(input_data=input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False,
                       order=2, direction="inter", name="difference", padding="SAME")

    # HPF and input data concat
    conv0_input_merge = tf.concat([conv0, input_data], 3, name="conv0_input_merge")
    concat_shape = conv0_input_merge.get_shape()
    print("name: %s, shape: (%d, %d, %d)" % ("conv0_input_merge", concat_shape[1], concat_shape[2], concat_shape[3]))

    # Group1
    conv1_1 = conv_layer(conv0_input_merge, 3, 3, 1, 1, 16, name="conv1_1", activation_method="leakrelu", padding="SAME")
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method="leakrelu", padding="SAME")
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method="leakrelu", padding="SAME")
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method="leakrelu", padding="SAME")
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method="leakrelu", padding="SAME")
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method="leakrelu", padding="SAME")
    pool3_3 = pool_layer(conv3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method="leakrelu", padding="SAME")
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding="SAME")
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method="leakrelu", is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method="leakrelu", padding="SAME")
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding="SAME")
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method="leakrelu", is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=True)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method="leakrelu", padding="SAME")
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding="SAME")
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method="leakrelu", is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=True)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="leakrelu", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="leakrelu", is_train=is_bn)

    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def wasdn1_7(input_data, class_num=2, is_bn=True):
    """
    Deepen the network to block convolutional layers
    """
    print("WASDN1_7: Deepen the network to 7 groups.")
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

    # Group7
    conv7_1 = conv_layer(pool6_4, 3, 3, 1, 1, 1024, name="conv7_1", activation_method="tanh", padding="SAME")
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 2048, name="conv7_2", activation_method=None, padding="SAME")
    bn7_3 = batch_normalization(conv7_2, name="BN7_3", activation_method="tanh", is_train=is_bn)
    pool7_4 = pool_layer(bn7_3, 2, 2, 2, 2, name="pool7_4", is_max_pool=True)

    # Fully connected layer
    fc8 = fc_layer(pool7_4, 4096, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)
    fc10 = fc_layer(bn9, 512, name="fc10", activation_method=None)
    bn11 = batch_normalization(fc10, name="BN11", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn11, class_num, name="fc12", activation_method=None)

    return logits


def wasdn1_8(input_data, class_num=2, is_bn=True):
    """
    Remove the 1x1 convolutional layers
    """
    print("WASDN1_8: Remove the 1x1 conv layers.")
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
    pool1_2 = pool_layer(conv1_1, 2, 2, 2, 2, name="pool1_2", is_max_pool=True)

    # Group2
    conv2_1 = conv_layer(pool1_2, 3, 3, 1, 1, 32, name="conv2_1", activation_method="tanh", padding="SAME")
    pool2_2 = pool_layer(conv2_1, 2, 2, 2, 2, name="pool2_2", is_max_pool=True)

    # Group3
    conv3_1 = conv_layer(pool2_2, 3, 3, 1, 1, 64, name="conv3_1", activation_method="tanh", padding="SAME")
    pool3_2 = pool_layer(conv3_1, 2, 2, 2, 2, name="pool3_2", is_max_pool=True)

    # Group4
    conv4_1 = conv_layer(pool3_2, 3, 3, 1, 1, 128, name="conv4_1", activation_method=None, padding="SAME")
    bn4_2 = batch_normalization(conv4_1, name="BN4_2", activation_method="tanh", is_train=is_bn)
    pool4_3 = pool_layer(bn4_2, 2, 2, 2, 2, name="pool4_3", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_3, 3, 3, 1, 1, 256, name="conv5_1", activation_method=None, padding="SAME")
    bn5_2 = batch_normalization(conv5_1, name="BN5_2", activation_method="tanh", is_train=is_bn)
    pool5_3 = pool_layer(bn5_2, 2, 2, 2, 2, name="pool5_3", is_max_pool=True)

    # Group6
    conv6_1 = conv_layer(pool5_3, 3, 3, 1, 1, 512, name="conv6_1", activation_method=None, padding="SAME")
    bn6_2 = batch_normalization(conv6_1, name="BN6_2", activation_method="tanh", is_train=is_bn)
    pool6_3 = pool_layer(bn6_2, 2, 2, 2, 2, name="pool6_3", is_max_pool=True)

    # Fully connected layer
    fc7 = fc_layer(pool6_3, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)

    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def wasdn1_9(input_data, class_num=2, is_bn=True):
    """
    Remove the HPF layer
    """
    print("WASDN1_9: Remove the HPF layer")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method="tanh", padding="SAME")
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


def wasdn2_1(input_data, class_num=2, is_bn=True):
    """
    Remove the BN layer in the first group
    """
    print("WASDN2_1: Remove the BN layer in the first group")
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


def wasdn2_2(input_data, class_num=2, is_bn=True):
    """
    Remove the BN layer in the first two groups
    """
    print("WASDN2_2: Remove the BN layers in the first  groups")
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


def wasdn2_3(input_data, class_num=2, is_bn=True):
    """
    Remove the BN layer in the first four groups
    """
    print("WASDN2_3: Remove the BN layers in the first four groups")
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
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method="tanh", padding="SAME")
    pool4_3 = pool_layer(conv4_2, 2, 2, 2, 2, name="pool4_3", is_max_pool=True)

    # Group5
    conv5_1 = conv_layer(pool4_3, 3, 3, 1, 1, 256, name="conv5_1", activation_method="tanh", padding="SAME")
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
