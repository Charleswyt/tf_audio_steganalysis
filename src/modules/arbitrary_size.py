#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019.03.05
Finished on 2019.03.05
Modified on

@author: Yuntao Wang
"""

import tensorflow as tf


def moments_extraction(input_data):
    """
    this function is used for dimension unification in Jessica's paper for steganalysis of arbitrary size
    calculate the moments of feature maps -- mean, variance, maximum and minimum
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :return:
        moments: a 4-D tensor [number, 1, 4, channel]
    """

    data_max = tf.reduce_max(input_data, axis=[1, 2], keep_dims=True, name="moments_max")
    data_min = tf.reduce_max(input_data, axis=[1, 2], keep_dims=True, name="moments_min")
    data_mean, data_variance = tf.nn.moments(input_data, axes=[1, 2], keep_dims=True, name="moments_mean_var")

    moments = tf.concat([data_max, data_min, data_mean, data_variance], axis=2, name="moments")

    return moments


def moments_extraction_enhancement(input_data):
    """
    this function is the enhancement version of moments extraction
    calculate the moments of feature maps -- mean, variance, maximum and minimum, kurtosis, skewness
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :return:
        moments: a 4-D tensor [number, 1, 6, channel]
    """

    data_max = tf.reduce_max(input_data, axis=[1, 2], keep_dims=True, name="moments_max")
    data_min = tf.reduce_min(input_data, axis=[1, 2], keep_dims=True, name="moments_min")
    data_mean, data_variance = tf.nn.moments(input_data, axes=[1, 2], keep_dims=True, name="moments_mean_var")

    input_data_sub_mean = tf.subtract(input_data, data_mean, name="input_data_sub_mean")
    data_variance_inverse = tf.divide(1.0, data_variance, name="data_variance_inverse")
    data_kurtosis = tf.multiply(tf.reduce_mean(tf.pow(input_data_sub_mean, 4)), tf.pow(data_variance_inverse, 4), name="kurtosis")
    data_skewness = tf.multiply(tf.reduce_mean(tf.pow(input_data_sub_mean, 3)), tf.pow(data_variance_inverse, 3), name="skewness")

    moments = tf.concat([data_max, data_min, data_mean, data_variance, data_kurtosis, data_skewness], axis=2, name="moments")

    return moments
