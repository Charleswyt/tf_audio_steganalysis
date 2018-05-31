#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on
Finished on
@author: Wang Yuntao
"""

import numpy as np
from pre_process import *
from skimage import io, util
from skimage import filters
import matplotlib.pyplot as plt

"""
function:
    image_info_show(image_file_path)                                                                        图像信息显示
    read_image(image_file_path, height, width, as_grey=False, is_diff=False, order=2, direction=0,
               is_trunc=False, threshold=128, threshold_left=128, threshold_right=256)                      图像读取
    read_image_batch(image_files_list, height=512, width=512, is_diff=False, order=2, direction=0,
                     is_trunc=False, threshold=128, threshold_left=128, threshold_right=256)                批量图像读取
"""


def image_info_show(image_file_path):
    img = io.imread(image_file_path)
    print("The type of the image is %s" % type(img))
    print("The height of the image is %d" % img.shape[1])
    print("The width of the image is %d" % img.shape[0])
    try:
        print("The channel of the image is %d" % img.shape[2])
    except IndexError:
        print("The channel of the image is 1")
    print("The size of the image is %d" % img.size)
    print("The min value of the pixel in the image is %d" % img.min())
    print("The max value of the pixel in the image is %d" % img.max())
    print("The mean value of the pixel in the image is %d" % img.mean())
    plt.imshow(img)
    plt.show()


def image_read(image_file_path, height, width, as_grey=False, is_diff=False, order=2, direction=0,
               is_trunc=False, threshold=128, threshold_left=128, threshold_right=256):
    """
    read image
    :param image_file_path: the file path of image
    :param height: the height of image
    :param width: the width of image
    :param as_grey: whether grayscale or not (default: False)
    :param is_diff: whether difference or not (default: False)
    :param order: the order of difference
    :param direction: the direction of difference (0 - row | 1 - col)
    :param is_trunc: whether truncation or not (default: False)
    :param threshold: the threshold of truncation
    :param threshold_left: the smaller threshold of truncation
    :param threshold_right: the bigger threshold of truncation
    :return:
    """
    image = io.imread(image_file_path, as_grey=as_grey)
    if is_trunc is True:
        image = truncate(image, threshold=threshold, threshold_left=threshold_left, threshold_right=threshold_right)
    if is_diff is True:
        image = np.diff(image, n=order, axis=direction)

    # reshape
    image_shape = np.shape(image)
    if len(image_shape) == 2:
        image = image[:height, :width]
        image = np.reshape(image, [heigth, width, 1])
    else:
        image = image[:height, :width, :]

    return image


def image_read_batch(image_files_list, height=512, width=512, is_diff=False, order=2, direction=0,
                     is_trunc=False, threshold=128, threshold_left=128, threshold_right=256):
    """
    read images batch by batch
    :param image_files_list: image files list
    :param height: the height of images
    :param width: the width of images
    :param is_diff: whether difference or not (default: False)
    :param order: the order of difference
    :param direction: the direction of difference (0 - row | 1 - col)
    :param is_trunc: whether truncation or not (default: False)
    :param threshold: the threshold of truncation
    :param threshold_left: the smaller threshold of truncation
    :param threshold_right: the bigger threshold of truncation
    :return:
        data, a 4-D tensor, [batch_size, height, width, channel]
    """
    files_num = len(image_files_list)
    data = np.zeros([files_num, height, width, 1], dtype=np.float32)

    i = 0
    for image_file in image_files_list:
        content = image_read(image_file, height=height, width=width, is_diff=is_diff, order=order, direction=direction,
                             is_trunc=is_trunc, threshold=threshold, threshold_left=threshold_left, threshold_right=threshold_right)
        data[i] = content
        i = i + 1

    return data


if __name__ == "__main__":
    pass
