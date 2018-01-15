#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage import io
from os.path import exists
import tensorflow as tf

"""
Created on 
Finished on 
@author: Wang Yuntao
"""

"""
function:
    imread(image_path, format=None, channel=3, name='image')                        图像读取
"""
def data_augment():
    pass


def tf_imread(image_path, image_format="jpg", channels=3, dtype="tf.uint8"):
    """
    读取图像
    :params: image_path: 图像路径
    :params: image_format: 图像格式
    :params: channel: 图像通道数
    :return: tensor
    """
    image_contents = tf.read_file(image_path)  
    if image_format == "bmp":
        image = tf.image.decode_bmp(image_contents, channels)
    elif image_format == "jpg":
        image = tf.image.decode_jpeg(image_contents, channels)
    elif image_format == "png":
        image = tf.image.decode_png(image_contents, channels, dtype)
    elif image_format == "gif":
        image = tf.image.decode_gif(image_contents)
    else:
        image = tf.image.decode_image(image_contents, channels)

    return image

