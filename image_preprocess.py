#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.04.20
Finished on 2018.04.20
Modified on 2018.08.27

@author: Wang Yuntao
"""

import numpy as np
from skimage import io, util

"""
function:
    image_info_show(image_file_path)                                                                        show the information of image
    read_image(image_file_path, height=None, width=None)                                                    read the image
    read_image_batch(image_files_list, height=None, width=None)                                             read images in batch
"""


def image_info_show(image_file_path):
    """
    show the information of the image
    :param image_file_path: path of image
    :return:
        NULL
    """
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


def image_read(image_file_path, height=None, width=None, channel=None):
    """
    read image
    :param image_file_path: the file path of image
    :param height: the height of image (default: None)
    :param width: the width of image (default: None)
    :param channel: the channel of image (default: None)
    :return:
        image data ndarry type, shape: [height, width, channel]
    """
    if channel == 1:
        image = io.imread(image_file_path, as_grey=True)
    else:
        image = io.imread(image_file_path, as_grey=False)

    # reshape
    image_shape = np.shape(image)
    if len(image_shape) == 2:
        image = image[:height, :width]
        image = np.reshape(image, [heigth, width, 1])
    else:
        image = image[:height, :width, :]

    return image


def image_read_batch(image_files_list, height=None, width=None, channel=None):
    """
    read images batch by batch
    :param image_files_list: image files list
    :param height: the height of images
    :param width: the width of images
    :param channel: the channel of image (default: None)
    :return:
        data, a 4-D ndarray, [batch_size, height, width, channel]
    """
    files_num = len(image_files_list)
    data = np.zeros([files_num, height, width, channel], dtype=np.float32)

    i = 0
    for image_file in image_files_list:
        content = image_read(image_file, height=height, width=width, channel=channel)
        data[i] = content
        i = i + 1

    return data
