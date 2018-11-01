#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2018.03.28
Finished on 2018.03.28
Modified on 2018.08.27

@author: Yuntao Wang
"""

import tensorflow as tf

kv_kernel = tf.constant(value=[-1, 2, -2, 2, -1, 2, -6, 8, -6, 2, -2, 8, -12, 8, -2, 2, -6, 8, -6, 2, -1, 2, -2, 2, -1],
                        dtype=tf.float32,
                        shape=[5, 5, 1, 1],
                        name="kv_kernel")
