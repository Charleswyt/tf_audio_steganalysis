#!/usr/bin/env python3
# -*- coding: utf-8 -*-33

from layer import *

"""
    function:
        network1  : The proposed network (最终选定的网络)
        network1_1: Remove the BN layer (去掉BN层)
        network1_2: Average pooling layer is used for subsampling (将所有的降采样方式改为平均池化方式)
        network1_3: Convolutional layer with stride 2 is used for subsampling (将所有的降采样方式改为卷积池化方式)
        network1_4: Replace the convolutional kernel with 5x5 kernel (将卷积核尺寸由3 x 3改为5 x 5)
        network1_5: ReLu is used as the activation function (将激活函数由Tanh改为ReLu)
        network1_6: Leaky-ReLu is used as the activation function (将激活函数由tanh改为Leaky-ReLu)
        network1_7: Deepen the network to block convolution layers (加深网络)
        
        Note: HPF and ABS is applied at the pre-processing
"""


def network1(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    The proposed network
    """
    print("network1: The proposed network")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 8, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 16, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 16, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 32, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 32, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 64, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 64, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 128, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 128, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 256, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 256, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # 全连接层
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_1(input_data, class_num=2, is_bn=False, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    去掉BN层
    """
    print("network1_1: Remove the BN layer")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 8, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 16, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 16, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 32, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 32, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 64, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 64, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 128, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 128, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 256, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 256, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # 全连接层
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_2(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=False):
    """
    在network1的基础上, 将所有降采样方式均改为平均池化
    """
    print("network1_2 replace the max pooling layer with the average pooling layer")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 8, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 16, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 16, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 32, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 32, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 64, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 64, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 128, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 128, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 256, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 256, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # 全连接层
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_3(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME"):
    """
    在network1的基础上, 将所有降采样方式均改为卷积池化
    """
    print("network1_3 replace the max pooling layer with the convolutional pooling layer")
    print("Network Structure: ")
    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 1, "conv1_1", activation_method, padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 8, "conv1_2", activation_method, padding)

    conv1_3 = conv_layer(conv1_2, 3, 3, 2, 2, 8, "conv1_3", "None")

    # Group2
    conv2_1 = conv_layer(conv1_3, 3, 3, 1, 1, 8, "conv2_1", activation_method, padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 16, "conv2_2", activation_method, padding)
    conv2_3 = conv_layer(conv2_2, 3, 3, 2, 2, 16, "conv2_3", "None")

    # Group3
    conv3_1 = conv_layer(conv2_3, 3, 3, 1, 1, 16, "conv3_1", activation_method, padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 32, "conv3_2", activation_method, padding)
    conv3_3 = conv_layer(conv3_2, 3, 3, 2, 2, 32, "conv3_3", "None")

    # Group4
    conv4_1 = conv_layer(conv3_3, 3, 3, 1, 1, 32, "conv4_1", activation_method, padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 64, "conv4_2", activation_method, padding)
    pool4_3 = pool_layer(conv4_2, 2, 2, 2, 2, "pool4_3")

    # Group5
    conv5_1 = conv_layer(pool4_3, 3, 3, 1, 1, 64, "conv5_1", activation_method, padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 128, "conv5_2", activation_method, padding)
    pool5_3 = pool_layer(conv5_2, 2, 2, 2, 2, "pool5_3")

    # Group6
    conv6_1 = conv_layer(pool5_3, 3, 3, 1, 1, 128, "conv6_1", activation_method, padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 256, "conv6_2", activation_method, padding)
    pool6_3 = pool_layer(conv6_2, 2, 2, 2, 2, "pool6_3")

    # 全连接层
    fc7 = fc_layer(pool6_3, 2048, "fc7", None)
    bn8 = batch_normalization(fc7, name="bn8")
    bn8 = activation(bn8, "relu")

    fc9 = fc_layer(bn8, 512, "fc9", None)
    bn10 = batch_normalization(fc9, name="bn10")
    bn11 = activation(bn10, "relu")

    logits = fc_layer(bn11, class_num, "fc12", None)

    return logits


def network1_4(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    将卷积核尺寸由3 x 3改为5 x 5
    """
    print("network1_4 replace the 3x3 kernel with the 5x5 kernel")
    print("Network Structure: ")
    # Group1
    conv1_1 = conv_layer(input_data, 5, 5, 1, 1, 8, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 16, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 5, 5, 1, 1, 16, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 32, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 5, 5, 1, 1, 32, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 64, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 5, 5, 1, 1, 64, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 128, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 5, 5, 1, 1, 128, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 256, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 5, 5, 1, 1, 256, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # 全连接层
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc11", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN12", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc13", activation_method=None)

    return logits


def network1_5(input_data, class_num=2, is_bn=True, activation_method="relu", padding="SAME", is_max_pool=True):
    """
    在network1的基础上, 将激活函数换为relu
    """
    print("network1_5 use relu as the activation function")
    print("Network Structure: ")
    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 8, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 16, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 16, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 32, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 32, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 64, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 64, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 128, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 128, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 256, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 256, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # 全连接层
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_6(input_data, class_num=2, is_bn=True, activation_method="leakrelu", padding="SAME", is_max_pool=True):
    """
    在network1的基础上, 将激活函数换为leakrelu
    """
    print("network1_6 use leakrelu as the activation function")
    print("Network Structure: ")
    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 8, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 16, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 16, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 32, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 32, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 64, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 64, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 128, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 128, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 256, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 256, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # 全连接层
    fc7 = fc_layer(pool6_4, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method=activation_method, is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def network1_7(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    deepen the network
    """
    print("Network Structure: ")
    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 8, name="conv1_1", activation_method=activation_method, padding=padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 16, name="conv1_2", activation_method=None, padding=padding)
    bn1_3 = batch_normalization(conv1_2, name="BN1_3", activation_method=activation_method, is_train=is_bn)
    pool1_4 = pool_layer(bn1_3, 2, 2, 2, 2, name="pool1_4", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_4, 3, 3, 1, 1, 16, name="conv2_1", activation_method=activation_method, padding=padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 32, name="conv2_2", activation_method=None, padding=padding)
    bn2_3 = batch_normalization(conv2_2, name="BN2_3", activation_method=activation_method, is_train=is_bn)
    pool2_4 = pool_layer(bn2_3, 2, 2, 2, 2, name="pool2_4", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_4, 3, 3, 1, 1, 32, name="conv3_1", activation_method=activation_method, padding=padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 64, name="conv3_2", activation_method=None, padding=padding)
    bn3_3 = batch_normalization(conv3_2, name="BN3_3", activation_method=activation_method, is_train=is_bn)
    pool3_4 = pool_layer(bn3_3, 2, 2, 2, 2, name="pool3_4", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_4, 3, 3, 1, 1, 64, name="conv4_1", activation_method=activation_method, padding=padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 128, name="conv4_2", activation_method=None, padding=padding)
    bn4_3 = batch_normalization(conv4_2, name="BN4_3", activation_method=activation_method, is_train=is_bn)
    pool4_4 = pool_layer(bn4_3, 2, 2, 2, 2, name="pool4_4", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_4, 3, 3, 1, 1, 128, name="conv5_1", activation_method=activation_method, padding=padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 256, name="conv5_2", activation_method=None, padding=padding)
    bn5_3 = batch_normalization(conv5_2, name="BN5_3", activation_method=activation_method, is_train=is_bn)
    pool5_4 = pool_layer(bn5_3, 2, 2, 2, 2, name="pool5_4", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_4, 3, 3, 1, 1, 256, name="conv6_1", activation_method=activation_method, padding=padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 512, name="conv6_2", activation_method=None, padding=padding)
    bn6_3 = batch_normalization(conv6_2, name="BN6_3", activation_method=activation_method, is_train=is_bn)
    pool6_4 = pool_layer(bn6_3, 2, 2, 2, 2, name="pool6_4", is_max_pool=is_max_pool)

    # Group7
    conv7_1 = conv_layer(pool6_4, 3, 3, 1, 1, 512, name="conv7_1", activation_method=activation_method, padding=padding)
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 1024, name="conv7_2", activation_method=None, padding=padding)
    bn7_3 = batch_normalization(conv7_2, name="BN7_3", activation_method=activation_method, is_train=is_bn)
    pool7_4 = pool_layer(bn7_3, 2, 2, 2, 2, name="pool7_4", is_max_pool=is_max_pool)

    # 全连接层
    fc8 = fc_layer(pool7_4, 4096, name="fc8", activation_method=None)
    bn9 = batch_normalization(fc8, name="BN9", activation_method=activation_method, is_train=is_bn)
    fc10 = fc_layer(bn9, 512, name="fc10", activation_method=None)
    bn11 = batch_normalization(fc10, name="BN11", activation_method=activation_method, is_train=is_bn)
    logits = fc_layer(bn11, class_num, name="fc12", activation_method=None)

    return logits


def network1_8(input_data, class_num=2, is_bn=True, activation_method="tanh", padding="SAME", is_max_pool=True):
    """
    The proposed network
    """
    print("network1_8: ")
    print("Network Structure: ")

    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 8, name="conv1_1", activation_method=None, padding=padding)
    bn1_2 = batch_normalization(conv1_1, name="BN1_2", activation_method=activation_method, is_train=is_bn)
    conv1_3 = conv_layer(bn1_2, 1, 1, 1, 1, 16, name="conv1_3", activation_method=None, padding=padding)
    bn1_4 = batch_normalization(conv1_3, name="BN1_4", activation_method=activation_method, is_train=is_bn)
    pool1_5 = pool_layer(bn1_4, 2, 2, 2, 2, name="pool1_5", is_max_pool=is_max_pool)

    # Group2
    conv2_1 = conv_layer(pool1_5, 3, 3, 1, 1, 16, name="conv2_1", activation_method=None, padding=padding)
    bn2_2 = batch_normalization(conv2_1, name="BN2_2", activation_method=activation_method, is_train=is_bn)
    conv2_3 = conv_layer(bn2_2, 1, 1, 1, 1, 32, name="conv2_3", activation_method=None, padding=padding)
    bn2_4 = batch_normalization(conv2_3, name="BN2_4", activation_method=activation_method, is_train=is_bn)
    pool2_5 = pool_layer(bn2_4, 2, 2, 2, 2, name="pool2_5", is_max_pool=is_max_pool)

    # Group3
    conv3_1 = conv_layer(pool2_5, 3, 3, 1, 1, 32, name="conv3_1", activation_method=None, padding=padding)
    bn3_2 = batch_normalization(conv3_1, name="BN3_2", activation_method=activation_method, is_train=is_bn)
    conv3_3 = conv_layer(bn3_2, 1, 1, 1, 1, 64, name="conv3_3", activation_method=None, padding=padding)
    bn3_4 = batch_normalization(conv3_3, name="BN3_4", activation_method=activation_method, is_train=is_bn)
    pool3_5 = pool_layer(bn3_4, 2, 2, 2, 2, name="pool3_5", is_max_pool=is_max_pool)

    # Group4
    conv4_1 = conv_layer(pool3_5, 3, 3, 1, 1, 32, name="conv4_1", activation_method=None, padding=padding)
    bn4_2 = batch_normalization(conv4_1, name="BN4_2", activation_method=activation_method, is_train=is_bn)
    conv4_3 = conv_layer(bn4_2, 1, 1, 1, 1, 64, name="conv4_3", activation_method=None, padding=padding)
    bn4_4 = batch_normalization(conv4_3, name="BN4_4", activation_method=activation_method, is_train=is_bn)
    pool4_5 = pool_layer(bn4_4, 2, 2, 2, 2, name="pool4_5", is_max_pool=is_max_pool)

    # Group5
    conv5_1 = conv_layer(pool4_5, 3, 3, 1, 1, 32, name="conv5_1", activation_method=None, padding=padding)
    bn5_2 = batch_normalization(conv5_1, name="BN5_2", activation_method=activation_method, is_train=is_bn)
    conv5_3 = conv_layer(bn5_2, 1, 1, 1, 1, 64, name="conv5_3", activation_method=None, padding=padding)
    bn5_4 = batch_normalization(conv5_3, name="BN5_4", activation_method=activation_method, is_train=is_bn)
    pool5_5 = pool_layer(bn5_4, 2, 2, 2, 2, name="pool5_5", is_max_pool=is_max_pool)

    # Group6
    conv6_1 = conv_layer(pool5_5, 3, 3, 1, 1, 32, name="conv6_1", activation_method=None, padding=padding)
    bn6_2 = batch_normalization(conv6_1, name="BN6_2", activation_method=activation_method, is_train=is_bn)
    conv6_3 = conv_layer(bn6_2, 1, 1, 1, 1, 64, name="conv6_3", activation_method=None, padding=padding)
    bn6_4 = batch_normalization(conv6_3, name="BN6_4", activation_method=activation_method, is_train=is_bn)
    pool6_5 = pool_layer(bn6_4, 2, 2, 2, 2, name="pool6_5", is_max_pool=is_max_pool)

    # 全连接层
    fc7 = fc_layer(pool6_5, 4096, name="fc7", activation_method=None)
    bn8 = batch_normalization(fc7, name="BN8", activation_method="tanh", is_train=is_bn)
    fc9 = fc_layer(bn8, 512, name="fc9", activation_method=None)
    bn10 = batch_normalization(fc9, name="BN10", activation_method="tanh", is_train=is_bn)
    logits = fc_layer(bn10, class_num, name="fc11", activation_method=None)

    return logits


def stegshi(input_data, class_num=2, is_bn=True):
    # group 0
    conv0 = static_conv_layer(input_data, kv_kernel, 1, 1, "conv0")

    # group 1
    conv1_1 = conv_layer(conv0, 5, 5, 1, 1, 8, "conv1_1", activation_method=None, init_method="gaussian", bias_term=False)
    conv1_2 = tf.abs(conv1_1, "conv1_abs")
    if is_bn is True:
        conv1_3 = batch_normalization(conv1_2, name="conv1_BN")
    else:
        conv1_3 = conv1_2
    conv1_4 = activation(conv1_3, "tanh")
    pool1_5 = pool_layer(conv1_4, 5, 5, 2, 2, "pool1_4", is_max_pool=False)

    # group 2
    conv2_1 = conv_layer(pool1_5, 5, 5, 1, 1, 16, "conv2_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn2_2 = batch_normalization(conv2_1, activation_method="tanh", name="BN2_2", is_train=is_bn)
    pool2_3 = pool_layer(bn2_2, 5, 5, 2, 2, "pool2_3", is_max_pool=False)

    # group 3
    conv3_1 = conv_layer(pool2_3, 1, 1, 1, 1, 32, "conv3_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn3_2 = batch_normalization(conv3_1, activation_method="relu", name="BN3_2", is_train=is_bn)
    pool3_3 = pool_layer(bn3_2, 5, 5, 2, 2, "pool3_3", is_max_pool=False)

    # group 4
    conv4_1 = conv_layer(pool3_3, 1, 1, 1, 1, 64, "conv4_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn4_2 = batch_normalization(conv4_1, activation_method="relu", name="BN4_2", is_train=is_bn)
    pool4_3 = pool_layer(bn4_2, 7, 7, 2, 2, "pool4_3", is_max_pool=False)

    # group 5
    conv5_1 = conv_layer(pool4_3, 1, 1, 1, 1, 128, "conv5_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn5_2 = batch_normalization(conv5_1, activation_method="relu", name="BN5_2", is_train=is_bn)
    pool5_3 = pool_layer(bn5_2, 7, 7, 2, 2, "pool5_3", is_max_pool=False)

    # group 6
    conv6_1 = conv_layer(pool5_3, 1, 1, 1, 1, 256, "conv6_1", activation_method=None, init_method="gaussian", bias_term=False)
    bn6_2 = batch_normalization(conv6_1, activation_method="relu", name="BN6_2", is_train=is_bn)
    pool6_3 = pool_layer(bn6_2, 16, 16, 16, 16, "pool6_3", is_max_pool=False)

    # fc layer
    logits = fc_layer(pool6_3, class_num, "fc7", activation_method=None)

    return logits


def le_net(input_data, class_num=10):
    # Group1
    conv1_1 = conv_layer(input_data, 5, 5, 1, 1, 6, "conv1", "VALID")
    pool1_2 = pool_layer(conv1_1, 2, 2, 2, 2, "pool1_2")

    # Group2
    conv2_1 = conv_layer(pool1_2, 5, 5, 1, 1, 16, "conv2_1", "VALID")
    pool2_2 = pool_layer(conv2_1, 2, 2, 2, 2, "pool2_2")

    # 全连接层
    fc4 = fc_layer(pool2_2, 120, "fc4", "relu")
    fc5 = fc_layer(fc4, 84, "fc5", False)
    fc5_drop = dropout(fc5, keep_pro=0.5, name="fc5_drop")
    logits = fc_layer(fc5_drop, class_num, "fc6", False)

    return logits


def alex_net(input_data, class_num=4096):
    # Group 1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 64, "conv1_1")
    pool1_2 = pool_layer(conv1_1, 2, 2, 2, 2, "pool1_2")
    norm1_3 = normalization(pool1_2, 4, "norm1_3")
    dropout1_4 = dropout(norm1_3, keep_pro=0.8, name="dropout1_4")

    # Group 2
    conv2_1 = conv_layer(dropout1_4, 3, 3, 1, 1, 128, "conv2_1")
    pool2_2 = pool_layer(conv2_1, 2, 2, 2, 2, "pool2_2")
    norm2_3 = normalization(pool2_2, 4, "norm2_3")
    dropout2_4 = dropout(norm2_3, keep_pro=0.8, name="dropout2_4")

    # Group 3
    conv3_1 = conv_layer(dropout2_4, 3, 3, 1, 1, 256, "conv3_1")
    pool3_2 = pool_layer(conv3_1, 2, 2, 2, 2, "pool3_2")
    norm3_3 = normalization(pool3_2, 4, "norm3_3")
    dropout3_4 = dropout(norm3_3, keep_pro=0.8, name="dropout3_4")

    # 全连接层
    fc4 = fc_layer(dropout3_4, 1024, "fc4")
    fc5 = fc_layer(fc4, 1024, "fc5")
    logits = fc_layer(fc5, class_num, "fc6")

    return logits


def vgg16(input_data, class_num=4096):
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
