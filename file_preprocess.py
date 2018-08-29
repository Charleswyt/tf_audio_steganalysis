#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import filetype as ft

"""
Created on 2018.01.15
Finished on 2018.01.15
Modified on 2018.08.29

@author: Wang Yuntao
"""

"""
function:
    get_file_name(file_path)                                                        get the name of the file
    get_file_size(file_path, unit="KB")                                             get the size of the file
    get_file_type(file_path)                                                        get the type of the file
    get_path_type(path)                                                             get the type of the input path (file, folder or not exist)
"""


def get_file_name(file_path):
    """
    get the name of the file
    :param file_path: the path of the file
    :return:
        the name of the file
    """
    if os.path.exists(file_path):
        file_path.replace("\\", "/")
        file_name = file_path.split(sep="/")[-1]
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


def get_file_type(file_path):
    """
    get the type of file according to the suffix of the file name
    :param file_path: file path
    :return:
        file type
    """
    if os.path.exists(file_path):
        file_type = ft.guess(file_path).extension
    else:
        file_type = None

    return file_type


def get_path_type(path):
    """
    get type of input path
    :param path: input path
    :return:
        path_type:
            "file": the input path corresponds to a file
            "folder": the input path corresponds to a folder
            None: the input path does not exist
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            return "file"
        else:
            return "folder"
    else:
        return None
