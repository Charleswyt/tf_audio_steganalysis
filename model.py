#!/usr/bin/env python3
# -*- coding: utf-8 -*-33

from network import *
import numpy as np
import tensorflow as tf

"""
Created on 2017.11.9
Finished on 2017.11.16
@author: Wang Yuntao
"""

"""
function:
    build the classified network
    
    pool_layer(input_data, height, width, x_stride, y_stride, name, is_max_pool=True, padding="SAME")
    batch_normalization(input_data, name, offset=0.0, scale=1.0, variance_epsilon=1e-3)
    dropout(input_data, keep_pro=0.5, name="dropout")
    fc_layer(input_data, output_dim, name, relu_flag=True)
    conv_layer(input_data, height, width, x_stride, y_stride, filter_num, name, activation="relu", 
        padding="SAME", is_pretrain=True)
    loss(logits, label)
    accuracy(logits, label)
    optimizer(losses, learning_rate, global_step, optimizer_type="Adam", beta1=0.9, beta2=0.999,
              epsilon=1e-8, initial_accumulator_value=0.1, momentum=0.9, decay=0.9)
    class Network(object)
    
    cats_vs_dogs_network(images, batch_size, n_classes)
"""


class Network(object):

    # Pre-Process
    def __init__(self, input_data, batch_size, keep_pro, class_num, skip_layer=None, model_path="vgg19.npy"):
        # Network类初始化
        if skip_layer is None:
            skip_layer = []

        self.input_data = input_data                        # 输入数据
        self.batch_size = batch_size                        # 批次大小
        self.keep_pro = keep_pro                            # 保留概率
        self.class_num = class_num                          # 分类数
        self.skip_layer = skip_layer                        # 跳过的层数
        self.model_path = model_path                        # 模型文件路径
        self.vgg16()

    def model_load(self, session):
        # 加载模型文件
        data_dict = np.load(self.model_path, encoding="bytes").item()
        keys = sorted(data_dict.keys())
        print("Skip Layer: ", self.skip_layer)

        for key in keys:
            if key not in self.skip_layer:
                weights = data_dict[key][0]
                biases = data_dict[key][1]
                print(key)
                print("weights shape: ", weights.shape)
                print("biases  shape: ", biases.shape)

                with tf.variable_scope(key, reuse=True):
                    for subkey, data in zip(("weights", "biases"), data_dict[key]):
                        session.run(tf.get_variable(subkey).assign(data))
        print("Load %s SUCCESS!" % self.model_path)

    # Network
    def vgg16(self):
        # vgg16
        conv1_1 = conv_layer(self.input_data, 3, 3, 1, 1, 64, "conv1_1")
        conv1_2 = conv_layer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
        pool1_1 = pool_layer(conv1_2, 2, 2, 2, 2, "pool1_1")

        conv2_1 = conv_layer(pool1_1, 3, 3, 1, 1, 128, "conv2_1")
        conv2_2 = conv_layer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
        pool2_1 = pool_layer(conv2_2, 2, 2, 2, 2, "pool2_1")

        conv3_1 = conv_layer(pool2_1, 3, 3, 1, 1, 256, "conv3_1")
        conv3_2 = conv_layer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
        conv3_3 = conv_layer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
        pool3_1 = pool_layer(conv3_3, 2, 2, 2, 2, "pool3_1")

        conv4_1 = conv_layer(pool3_1, 3, 3, 1, 1, 512, "conv4_1")
        conv4_2 = conv_layer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
        conv4_3 = conv_layer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
        conv4_4 = conv_layer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
        pool4_1 = pool_layer(conv4_4, 2, 2, 2, 2, "pool4_1")

        conv5_1 = conv_layer(pool4_1, 3, 3, 1, 1, 512, "conv5_1")
        conv5_2 = conv_layer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
        conv5_3 = conv_layer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
        conv5_4 = conv_layer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
        pool5_1 = pool_layer(conv5_4, 2, 2, 2, 2, "pool5_1")

        fc6 = fc_layer(pool5_1, 4096, "fc6")
        bn6 = batch_normalization(fc6, "bn6")
        fc7 = fc_layer(bn6, 4096, "fc7")
        bn7 = batch_normalization(fc7, "bn7")
        fc8 = fc_layer(bn7, self.class_num, "fc8")
        # self.fc8 = fc8

        return fc8




