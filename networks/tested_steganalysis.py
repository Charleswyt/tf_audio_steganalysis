#!/usr/bin/env python3
# -*- coding: utf-8 -*-33

from layer import *
from filters import *

"""
    function:
        network1_1 : Remove the BN layer
        network1_2 : Average pooling layer is used for subsampling
        network1_3 : Convolutional layer with stride 2 is used for subsampling
        network1_4 : Replace the convolutional kernel with 5x5 kernel
        network1_5 : ReLu is used as the activation function
        network1_6 : Leaky-ReLu is used as the activation function
        network1_7 : Deepen the network to block convolutional layers
        network1_8 : Remove the 1x1 convolutional layers
        network1_9 : Remove the HPF layer
        network1__1: Remove the first BN layer in the first group
        network1__2: Remove the first BN layers in the first two groups
        network1__3: Remove the first BN layers in the first four groups
        network1__4: Add BN layers at the top of 3x3 conv layer
"""


def network1_1(input_data, class_num=2, is_bn=False, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    remove the batch normalization (去掉BN层)
    """
    print("network1_1: Remove the BN layer")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_2(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=False):
    """
    Average pooling layer is used for subsampling (将所有的降采样方式改为平均池化方式)
    """
    print("network1_2 replace the max pooling layer with the average pooling layer")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_3(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=False):
    """
    Convolutional layer with stride 2 is used for subsampling (将所有的降采样方式改为卷积池化方式)
    """
    print("network1_3 replace the max pooling layer with the convolutional pooling layer")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    conv1_4 = conv_layer(bn1_3, 3, 3, 2, 2, 16, name="conv1_4", activation_method="None", padding="VALID")

    # Group2
    conv2_1 = conv_layer(conv1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    conv2_4 = conv_layer(bn2_3, 3, 3, 2, 2, 64, name="conv2_4", activation_method="None", padding="VALID")

    # Group3
    conv3_1 = conv_layer(conv2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, "conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
    conv3_3 = conv_layer(bn3_3, 3, 3, 2, 2, 128, name="conv3_4", activation_method="None", padding="VALID")

    # Group4
    conv4_1 = conv_layer(conv3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_4(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Replace the convolutional kernel with 5x5 kernel (将卷积核尺寸由3 x 3改为5 x 5)
    """
    print("network1_4 replace the 3 x 3 kernel with the 5 x 5 kernel")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 5, 5, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 5, 5, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 5, 5, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 5, 5, 1, 1, 128, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 5, 5, 1, 1, 256, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 5, 5, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc11", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN12", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc13", activation_method=None)

    return logits


def network1_5(input_data, class_num=2, is_bn=True, activation_method="relu", padding="SAME", is_max_pool=True):
    """
    ReLu is used as the activation function (将激活函数由Tanh改为ReLu)
    """
    print("network1_5 use relu as the activation function")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_6(input_data, class_num=2, is_bn=True, activation_method="leakrelu", padding="SAME", is_max_pool=True):
    """
    Leaky-ReLu is used as the activation function (将激活函数由tanh改为Leaky-ReLu)
    """
    print("network1_6 use leakrelu as the activation function")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_7(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Deepen the network to block convolutional layers (加深网络)
    """
    print("network1_7: Deepen the network to 7 groups.")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Group7
    conv7_1 = conv_layer(pool6_4, 3, 3, 1, 1, 1024, name="conv7_1", activation_method=activation_method, padding=padding)
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 2048, name="conv7_2", activation_method=None, padding=padding)
    bn7_3 = batch_normalization(conv7_2, name="BN7_3", activation_method=activation_method, is_train=is_bn)
    pool7_4 = pool_layer(bn7_3, 2, 2, 2, 2, name="pool7_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc8 = fc_layer(pool7_4, 4096, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method=activation_method, is_train=is_bn)
    fc10 = fc_layer(bn9, 512, name="fc10", activation_method=None)
    bn11 = batch_normalization(fc10, name="BN11", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn11, class_num, name="fc12", activation_method=None)

    return logits


def network1_8(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove the 1x1 convolutional layers (移除1x1卷积层)
    """
    print("network1_8: Remove the 1x1 conv layers.")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    bn1_2 = batch_normalization(conv1_1, name="BN1_2", activation_method=activation_method, is_train=False)
    pool1_3 = pool_layer(bn1_2, 2, 2, 2, 2, name="pool1_3", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 32, name="conv2_1", activation_method=None, padding=padding)
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method=activation_method, is_train=False)
    pool2_3 = pool_layer(bn2_2, 2, 2, 2, 2, name="pool2_3", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 64, name="conv3_1", activation_method=None, padding=padding)
    bn3_2 = batch_normalization(conv3_1, name="BN3_2", activation_method=activation_method, is_train=False)
    pool3_3 = pool_layer(bn3_2, 2, 2, 2, 2, name="pool3_3", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 128, name="conv4_1", activation_method=None, padding=padding)
    bn4_2 = batch_normalization(conv4_1, name="BN4_2", activation_method=activation_method, is_train=is_bn)
    pool4_3 = pool_layer(bn4_2, 2, 2, 2, 2, name="pool4_3", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_3, 3, 3, 1, 1, 256, name="conv5_1", activation_method=None, padding=padding)
    bn5_2 = batch_normalization(conv5_1, name="BN5_2", activation_method=activation_method, is_train=is_bn)
    pool5_3 = pool_layer(bn5_2, 2, 2, 2, 2, name="pool5_3", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_3, 3, 3, 1, 1, 512, name="conv6_1", activation_method=None, padding=padding)
    bn6_2 = batch_normalization(conv6_1, name="BN6_2", activation_method=activation_method, is_train=is_bn)
    pool6_3 = pool_layer(bn6_2, 2, 2, 2, 2, name="pool6_3", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_3, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_9(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove the HPF layer (移除高通滤波器)
    """
    print("network1_2 replace the max pooling layer with the average pooling layer")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1__1(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove the BN layer in the first group
    """
    print("network1__1: Remove the BN layer in the first group.")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1__2(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove the BN layer in the first two groups
    """
    print("network1__2: Remove the BN layers in the first  groups.")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
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

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1__3(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Remove the BN layer in the first four groups
    """
    print("network1__3: Remove the BN layers in the first four groups")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 32, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=False)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 32, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 64, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=False)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 64, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 128, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=False)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 128, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 256, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=False)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 256, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 512, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 512, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 1024, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1__4(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    Add BN layers at the top of 3x3 conv layer
    """
    print("network1: The proposed network")
    print("Network Structure: ")

    # preprocessing
    data_diff = diff_layer(input_data, is_diff=True, is_diff_abs=False, is_abs_diff=False, order=2, direction="inter", name="difference")

    # Group1
    conv1_1 = conv_layer(data_diff, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
    bn1_2 = batch_normalization(conv1_1, name="BN1_2", activation_method=activation_method, is_train=False)
    conv1_3 = conv_layer(bn1_2, 1, 1, 1, 1, 32, name="conv1_3", activation_method=None, padding=padding)
    bn1_4 = batch_normalization(conv1_3, name="BN1_4", activation_method=activation_method, is_train=False)
    pool1_5 = pool_layer(bn1_4, 2, 2, 2, 2, name="pool1_5", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_5, 3, 3, 1, 1, 32, name="conv2_1", activation_method=None, padding=padding)
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method=activation_method, is_train=False)
    conv2_3 = conv_layer(bn2_2, 1, 1, 1, 1, 64, name="conv2_3", activation_method=None, padding=padding)
    bn2_4 = batch_normalization(conv2_3, name="BN2_4", activation_method=activation_method, is_train=False)
    pool2_5 = pool_layer(bn2_4, 2, 2, 2, 2, name="pool2_5", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_5, 3, 3, 1, 1, 64, name="conv3_1", activation_method=None, padding=padding)
    bn3_2 = batch_normalization(conv3_1, name="BN3_2", activation_method=activation_method, is_train=False)
    conv3_3 = conv_layer(bn3_2, 1, 1, 1, 1, 128, name="conv3_3", activation_method=None, padding=padding)
    bn3_4 = batch_normalization(conv3_3, name="BN3_4", activation_method=activation_method, is_train=False)
    pool3_5 = pool_layer(bn3_4, 2, 2, 2, 2, name="pool3_5", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_5, 3, 3, 1, 1, 128, name="conv4_1", activation_method=None, padding=padding)
    bn4_2 = batch_normalization(conv4_1, name="BN4_2", activation_method=activation_method, is_train=is_bn)
    conv4_3 = conv_layer(bn4_2, 1, 1, 1, 1, 256, name="conv4_3", activation_method=None, padding=padding)
    bn4_4 = batch_normalization(conv4_3, name="BN4_4", activation_method=activation_method, is_train=is_bn)
    pool4_5 = pool_layer(bn4_4, 2, 2, 2, 2, name="pool4_5", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_5, 3, 3, 1, 1, 256, name="conv5_1", activation_method=None, padding=padding)
    bn5_2 = batch_normalization(conv5_1, name="BN5_2", activation_method=activation_method, is_train=is_bn)
    conv5_3 = conv_layer(bn5_2, 1, 1, 1, 1, 512, name="conv5_3", activation_method=None, padding=padding)
    bn5_4 = batch_normalization(conv5_3, name="BN5_4", activation_method=activation_method, is_train=is_bn)
    pool5_5 = pool_layer(bn5_4, 2, 2, 2, 2, name="pool5_5", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_5, 3, 3, 1, 1, 512, name="conv6_1", activation_method=None, padding=padding)
    bn6_2 = batch_normalization(conv6_1, name="BN6_2", activation_method=activation_method, is_train=is_bn)
    conv6_3 = conv_layer(bn6_2, 1, 1, 1, 1, 1024, name="conv6_3", activation_method=None, padding=padding)
    bn6_4 = batch_normalization(conv6_3, name="BN6_4", activation_method=activation_method, is_train=is_bn)
    pool6_5 = pool_layer(bn6_4, 2, 2, 2, 2, name="pool6_5", is_max_pool=is_max_pool)

    # Fully connected layer
    fc7 = fc_layer(pool6_5, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def siamese_network(input_data1, input_data2, network=None, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
        Siamese network for steganalysis
    """
    print("Siamese Network, Two input, Shared parameters")
    print("Network Structure: ")

    command1 = network + "(input_data1, classes_num, is_bn)"
    command2 = network + "(input_data2, classes_num, is_bn)"

    output1 = eval(command1)
    output2 = eval(command2)

    return output1, output2


def downsampling_network(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    downsampling network
    """
    print("network: downsampling network")
    print("Network Structure: ")

    # down sampling
    conv0 = down_sampling(input_data, 2, 2, "down_sampling")

    # Group1
    conv1_1 = conv_layer(conv0, 3, 3, 1, 1, 16, name="conv1_1", activation_method=activation_method, padding=padding)
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

    # Fully connected layer
    fc6 = fc_layer(pool5_4, 4096, name="fc6", activation_method=None)
    bn7 = batch_normalization(fc6, name="BN7", activation_method="tanh", is_train=is_bn)
    fc8 = fc_layer(bn7, 512, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn9, class_num, name="fc10", activation_method=None)

    return logits
