#!/usr/bin/env python3
# -*- coding: utf-8 -*-33

from layer import *

"""
    function:
        network1  : 最终选定的网络
        network1_1: 将激活函数由tanh改为relu
        network1_2: 将所有的降采样方式改为池化方式
        network1_3: 去掉BN层
        network1_4: 将卷积核尺寸由3 x 3改为5 x 5
        network1_5: 将group改为6
"""


def network1(input_data, class_num=2, activation_method="tanh", padding="SAME"):
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

    # Group7
    conv7_1 = conv_layer(pool6_3, 3, 3, 1, 1, 256, "conv7_1", activation_method, padding)
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 512, "conv7_2", activation_method, padding)
    pool7_3 = pool_layer(conv7_2, 2, 2, 2, 2, "pool7_3")

    # 全连接层
    fc8 = fc_layer(pool7_3, 2048, "fc8", "relu")
    bn9 = batch_normalization(fc8, name="bn9")
    fc10 = fc_layer(bn9, 512, "fc10", "relu")
    bn11 = batch_normalization(fc10, name="bn11")
    logits = fc_layer(bn11, class_num, "fc12", None)

    return logits


def network1_1(input_data, class_num=2, activation_method="relu", padding="SAME"):
    """
    在network1的基础上, 将激活函数换为relu
    """
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

    # Group7
    conv7_1 = conv_layer(pool6_3, 3, 3, 1, 1, 256, "conv7_1", activation_method, padding)
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 512, "conv7_2", activation_method, padding)
    pool7_3 = pool_layer(conv7_2, 2, 2, 2, 2, "pool7_3")

    # 全连接层
    fc8 = fc_layer(pool7_3, 2048, "fc8", "relu")
    bn9 = batch_normalization(fc8, name="bn9")
    fc10 = fc_layer(bn9, 512, "fc10", "relu")
    bn11 = batch_normalization(fc10, name="bn11")
    logits = fc_layer(bn11, class_num, "fc12", None)

    return logits


def network1_2(input_data, class_num=2, activation_method="tanh", padding="SAME"):
    """
    在network1的基础上, 将所有降采样方式均改为池化
    """
    print("Network Structure: ")
    # Group1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 1, "conv1_1", activation_method, padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 8, "conv1_2", activation_method, padding)
    pool1_3 = pool_layer(conv1_2, 2, 2, 2, 2, "pool1_3")

    # Group2
    conv2_1 = conv_layer(pool1_3, 3, 3, 1, 1, 8, "conv2_1", activation_method, padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 16, "conv2_2", activation_method, padding)
    pool2_3 = pool_layer(conv2_2, 2, 2, 2, 2, "pool2_3")

    # Group3
    conv3_1 = conv_layer(pool2_3, 3, 3, 1, 1, 16, "conv3_1", activation_method, padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 32, "conv3_2", activation_method, padding)
    pool3_3 = pool_layer(conv3_2, 2, 2, 2, 2, "pool3_3")

    # Group4
    conv4_1 = conv_layer(pool3_3, 3, 3, 1, 1, 32, "conv4_1", activation_method, padding)
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

    # Group7
    conv7_1 = conv_layer(pool6_3, 3, 3, 1, 1, 256, "conv7_1", activation_method, padding)
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 512, "conv7_2", activation_method, padding)
    pool7_3 = pool_layer(conv7_2, 2, 2, 2, 2, "pool7_3")

    # 全连接层
    fc8 = fc_layer(pool7_3, 2048, "fc8", "relu")
    bn9 = batch_normalization(fc8, name="bn9")
    fc10 = fc_layer(bn9, 512, "fc10", "relu")
    bn11 = batch_normalization(fc10, name="bn11")
    logits = fc_layer(bn11, class_num, "fc12", None)

    return logits


def network1_3(input_data, class_num=2, activation_method="tanh", padding="SAME"):
    """
    去掉BN层
    """
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

    # Group7
    conv7_1 = conv_layer(pool6_3, 3, 3, 1, 1, 256, "conv7_1", activation_method, padding)
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 512, "conv7_2", activation_method, padding)
    pool7_3 = pool_layer(conv7_2, 2, 2, 2, 2, "pool7_3")

    # 全连接层
    fc8 = fc_layer(pool7_3, 2048, "fc8", "relu")
    fc9 = fc_layer(fc8, 512, "fc9", "relu")
    logits = fc_layer(fc9, class_num, "fc12", None)

    return logits


def network1_4(input_data, class_num=2, activation_method="tanh", padding="SAME"):
    """
    讲卷积核尺寸由3 x 3改为5 x 5
    """
    print("Network Structure: ")
    # Group1
    conv1_1 = conv_layer(input_data, 5, 5, 1, 1, 1, "conv1_1", activation_method, padding)
    conv1_2 = conv_layer(conv1_1, 1, 1, 1, 1, 8, "conv1_2", activation_method, padding)
    conv1_3 = conv_layer(conv1_2, 5, 5, 2, 2, 8, "conv1_3", "None")

    # Group2
    conv2_1 = conv_layer(conv1_3, 5, 5, 1, 1, 8, "conv2_1", activation_method, padding)
    conv2_2 = conv_layer(conv2_1, 1, 1, 1, 1, 16, "conv2_2", activation_method, padding)
    conv2_3 = conv_layer(conv2_2, 5, 5, 2, 2, 16, "conv2_3", "None")

    # Group3
    conv3_1 = conv_layer(conv2_3, 5, 5, 1, 1, 16, "conv3_1", activation_method, padding)
    conv3_2 = conv_layer(conv3_1, 1, 1, 1, 1, 32, "conv3_2", activation_method, padding)
    conv3_3 = conv_layer(conv3_2, 5, 5, 2, 2, 32, "conv3_3", "None")

    # Group4
    conv4_1 = conv_layer(conv3_3, 5, 5, 1, 1, 32, "conv4_1", activation_method, padding)
    conv4_2 = conv_layer(conv4_1, 1, 1, 1, 1, 64, "conv4_2", activation_method, padding)
    pool4_3 = pool_layer(conv4_2, 2, 2, 2, 2, "pool4_3")

    # Group5
    conv5_1 = conv_layer(pool4_3, 5, 5, 1, 1, 64, "conv5_1", activation_method, padding)
    conv5_2 = conv_layer(conv5_1, 1, 1, 1, 1, 128, "conv5_2", activation_method, padding)
    pool5_3 = pool_layer(conv5_2, 2, 2, 2, 2, "pool5_3")

    # Group6
    conv6_1 = conv_layer(pool5_3, 5, 5, 1, 1, 128, "conv6_1", activation_method, padding)
    conv6_2 = conv_layer(conv6_1, 1, 1, 1, 1, 256, "conv6_2", activation_method, padding)
    pool6_3 = pool_layer(conv6_2, 2, 2, 2, 2, "pool6_3")

    # Group7
    conv7_1 = conv_layer(pool6_3, 5, 5, 1, 1, 256, "conv7_1", activation_method, padding)
    conv7_2 = conv_layer(conv7_1, 1, 1, 1, 1, 512, "conv7_2", activation_method, padding)
    pool7_3 = pool_layer(conv7_2, 2, 2, 2, 2, "pool7_3")

    # 全连接层
    fc9 = fc_layer(pool7_3, 2048, "fc9", "relu")
    bn10 = batch_normalization(fc9, name="bn10")
    fc11 = fc_layer(bn10, 512, "fc11", "relu")
    bn12 = batch_normalization(fc11, name="bn12")
    logits = fc_layer(bn12, class_num, "fc12", None)

    return logits


def network1_5(input_data, class_num=2, activation_method="tanh", padding="SAME"):
    """
    group = 6
    """
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
    fc7 = fc_layer(pool6_3, 2048, "fc7", "relu")
    bn8 = batch_normalization(fc7, name="bn8")
    fc9 = fc_layer(bn8, 512, "fc9", "relu")
    bn10 = batch_normalization(fc9, name="bn10")
    logits = fc_layer(bn10, class_num, "fc11", None)

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
    logits = fc_layer(fc5, class_num, "fc6", False)

    return logits


def alex_net(input_data, class_num=4096):
    # Group 1
    conv1_1 = conv_layer(input_data, 3, 3, 1, 1, 64, "conv1_1")
    pool1_2 = pool_layer(conv1_1, 2, 2, 2, 2, "pool1_2")
    norm1_3 = normalization(pool1_2, 4, "norm1_3")
    dropout1_4 = dropout(norm1_3, 0.8, "dropout1_4")

    # Group 2
    conv2_1 = conv_layer(dropout1_4, 3, 3, 1, 1, 128, "conv2_1")
    pool2_2 = pool_layer(conv2_1, 2, 2, 2, 2, "pool2_2")
    norm2_3 = normalization(pool2_2, 4, "norm2_3")
    dropout2_4 = dropout(norm2_3, 0.8, "dropout2_4")

    # Group 3
    conv3_1 = conv_layer(dropout2_4, 3, 3, 1, 1, 256, "conv3_1")
    pool3_2 = pool_layer(conv3_1, 2, 2, 2, 2, "pool3_2")
    norm3_3 = normalization(pool3_2, 4, "norm3_3")
    dropout3_4 = dropout(norm3_3, 0.8, "dropout3_4")

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
    fc6_drop = dropout(fc6, name="fc6_drop")
    fc7 = fc_layer(fc6_drop, 4096, "fc7")
    fc7_drop = dropout(fc7, name="fc7_drop")
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
    fc6_drop = dropout(fc6, name="fc6_drop")
    fc7 = fc_layer(fc6_drop, 4096, "fc7")
    fc7_drop = dropout(fc7, name="fc7_drop")
    logits = fc_layer(fc7_drop, class_num, "fc8")

    return logits

