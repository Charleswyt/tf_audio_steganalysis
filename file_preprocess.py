#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

"""
Created on 2018.01.15
Finished on 2018.01.15
@author: Wang Yuntao
"""

"""
function:
    get_file_name(file_path)                                                        获取文件名
    get_file_size(file_path, unit="KB")                                             获取文件大小
    get_file_type(file_path)                                                        获取文件类型
"""


def get_file_name(file_path):
    """
    获取文件名
    :param file_path: 文件路径
    :return: 文件名
    """
    if os.path.exists(file_path):
        file_name = file_path.split(sep='/')[-1]
    else:
        file_name = None
    return file_name


def get_file_size(file_path, unit="KB"):
    """
    获取文件名
    :param file_path: 文件路径
    :param unit: 文件大小单位(B KB MB GB TB)
    :return: file_size
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    power = units.index(unit)
    divisor = 1024 ** power
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        file_size = round(file_size / divisor)
    else:
        file_size = -1

    return file_size


def get_file_type(file_path):
    """
    获取文件类型
    :param file_path: 文件路径
    :return: 文件类型
    """
    if os.path.exists(file_path):
        file_type = file_path.split(sep='.')[-1]
    else:
        file_type = None

    return file_type


