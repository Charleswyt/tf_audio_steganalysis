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

import numpy as np
from skimage import io, util
from skimage import filters
import matplotlib.pyplot as plt

"""
function:
    imread(image_path, format=None, channel=3, name='image')                        图像读取
"""


def read_image_batch(image_files_list, height=512, width=512):

    files_num = len(image_files_list)
    data = np.zeros([files_num, height, width, 1], dtype=np.float32)

    i = 0
    for image_file in image_files_list:
        img = io.imread(image_file)
        content = np.reshape(img, [height, width, 1])
        data[i] = content
        i = i + 1

    return data


if __name__ == "__main__":
    image_path = "2501.pgm"
    image = io.imread(image_path)
    # util.crop()
    # image = filters.sobel(image[:, :, 0])
    print(image[:10, :10])
    plt.imshow(image)
    plt.show()










# def data_augment():
#     pass
#
#
# def tf_imread(image_path, image_format="jpg", channels=3, dtype="tf.uint8"):
#     """
#     读取图像
#     :params: image_path: 图像路径
#     :params: image_format: 图像格式
#     :params: channel: 图像通道数
#     :return: tensor
#     """
#     image_contents = tf.read_file(image_path)
#     if image_format == "bmp":
#         image = tf.image.decode_bmp(image_contents, channels)
#     elif image_format == "jpg":
#         image = tf.image.decode_jpeg(image_contents, channels)
#     elif image_format == "png":
#         image = tf.image.decode_png(image_contents, channels, dtype)
#     elif image_format == "gif":
#         image = tf.image.decode_gif(image_contents)
#     else:
#         image = tf.image.decode_image(image_contents, channels)
#
#     return image

