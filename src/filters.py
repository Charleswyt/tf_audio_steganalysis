#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2018.03.28
Finished on 2018.03.28
Modified on 2018.08.27

@author: Wang Yuntao
"""

import tensorflow as tf

kv_kernel = tf.constant(value=[-1, 2, -2, 2, -1, 2, -6, 8, -6, 2, -2, 8, -12, 8, -2, 2, -6, 8, -6, 2, -1, 2, -2, 2, -1],
                        dtype=tf.float32,
                        shape=[5, 5, 1, 1],
                        name="kv_kernel")

downsampling_kernel1 = tf.constant(value=[1, 0, 0, 0],
                                   dtype=tf.float32,
                                   shape=[2, 2, 1, 1],
                                   name="downsampling_kernel1")

downsampling_kernel2 = tf.constant(value=[0, 1, 0, 0],
                                   dtype=tf.float32,
                                   shape=[2, 2, 1, 1],
                                   name="downsampling_kernel2")

downsampling_kernel3 = tf.constant(value=[0, 0, 1, 0],
                                   dtype=tf.float32,
                                   shape=[2, 2, 1, 1],
                                   name="downsampling_kernel3")

downsampling_kernel4 = tf.constant(value=[0, 0, 0, 1],
                                   dtype=tf.float32,
                                   shape=[2, 2, 1, 1],
                                   name="downsampling_kernel4")
