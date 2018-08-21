#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.04.20
Finished on 2018.04.20
@author: Wang Yuntao
"""

import numpy as np
from pre_process import *
from skimage import io, util
import matplotlib.pyplot as plt

"""
function:
    image_info_show(image_file_path)                                                                        show the information of image
    read_image(image_file_path, height, width, as_grey=False, is_diff=False, order=2, direction=0,
               is_trunc=False, threshold=128, threshold_left=128, threshold_right=256)                      read the image
    read_image_batch(image_files_list, height=512, width=512, is_diff=False, order=2, direction=0,
                     is_trunc=False, threshold=128, threshold_left=128, threshold_right=256)                read images in batch
"""

"""
Something about scikit-image:
            Submode                         Function
            io                              读取、保存和显示图片或视频
            data                            提供一些测试图片和样本数据
            color                           颜色空间变换
            filters                         图像增强、边缘检测、排序滤波器、自动阈值等
            draw                            操作于numpy数组上的基本图形绘制，包括线条、矩形、圆和文本等
            transform                       几何变换或其它变换，如旋转、拉伸和拉东变换等
            morphology                      形态学操作，如开闭运算、骨架提取等
            exposure                        图片强度调整，如亮度调整、直方图均衡等
            feature                         特征检测与提取等
            measure                         图像属性的测量，如相似性或等高线等
            segmentation                    图像分割
            restoration                     图像恢复
            util                            通用函数
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
    :param as_grey: whether grays-cale or not (default: False)
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
