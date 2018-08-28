#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

"""
Created on 2018.01.15
Finished on 2018.01.15
Modified on 2018.08.27

@author: Wang Yuntao
"""

"""
function:
    get_file_name(file_path)                                                        get the name of file
    get_file_size(file_path, unit="KB")                                             get the size of file
    get_file_type(file_path)                                                        get the type of file
"""


def get_file_name(file_path, sep="/"):
    """
    get the name of file
    :param file_path: file path
    :param sep: separator
    :return:
        file name
    """
    if os.path.exists(file_path):
        file_path.replace("\\", "/")
        file_name = file_path.split(sep=sep)[-1]
    else:
        file_name = None

    return file_name


def get_file_size(file_path, unit="KB"):
    """
    get the size of file
    :param file_path: file path
    :param unit: the unit of file size (Unit: B KB MB GB TB)
    :return:
        file_size
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    power = units.index(unit)
    divisor = 1024 ** power
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        file_size = round(file_size / divisor)
    else:
        file_size = None

    return file_size


def get_file_type(file_path, sep="."):
    """
    get the type of file
    :param file_path: file path
    :param sep: separator
    :return:
        file type
    """
    if os.path.exists(file_path):
        file_type = file_path.split(sep=sep)[-1]
    else:
        file_type = None

    return file_type


