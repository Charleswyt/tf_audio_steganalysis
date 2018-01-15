#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils
"""
Created on 2017.11.16
Finished on 2017.11.16
@author: Wang Yuntao
"""

# tf.variable_scope与tf.name_scope的区别:
"""
with tf.variable_scope("conv1"):
    weights = tf.get_variable(name="weights",
                              shape=[3, 3, 3, 16],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases",
                             shape=[16],
                             initializer=tf.constant_initializer(0.0))
    print("------ variable_scope ------")
    print("weights name: ", weights.name)
    print("biases name: ", biases.name)

with tf.name_scope("conv1"):
    weight = tf.get_variable(name="weights",
                             shape=[3, 3, 3, 16],
                             initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases",
                             shape=[16],
                             initializer=tf.constant_initializer(0.0))
    print("------ name_scope ------")
    print("weights name: ", weights.name)
    print("biases name: ", biases.name)

"""
