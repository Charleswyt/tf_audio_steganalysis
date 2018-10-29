#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.08.13
Finished on 2018.08.14
@author: Wang Yuntao
"""

import os
import sys
import shutil


def get_file_name(file_path, sep="/"):
    """
    get the name of file
    :param file_path: file path
    :param sep: separator
    :return: file name
    """
    if os.path.exists(file_path):
        file_path.replace("\\", "/")
        file_name = file_path.split(sep=sep)[-1]
    else:
        file_name = None
    return file_name


def fullfile(file_dir, file_name):
    """
    fullfile as matlab
    :param file_dir: file dir
    :param file_name: file name
    :return: a full file path
    """
    full_file_path = os.path.join(file_dir, file_name)
    full_file_path = full_file_path.replace("\\", "/")

    return full_file_path


def get_file_type(file_path, sep="."):
    """
    get the type of file
    :param file_path: file path
    :param sep: separator
    :return: file type
    """
    if os.path.exists(file_path):
        file_type = file_path.split(sep=sep)[-1]
    else:
        file_type = None

    return file_type


def get_files_list(files_path, file_type="txt"):
    """
    :param files_path: path of MP3 files for move
    :param file_type: file type, default is "txt"
    :return: Null
    """
    filename = os.listdir(files_path)
    files_list = []
    for file in filename:
        file_path = fullfile(files_path, file)
        if get_file_type(file_path) == file_type:
            files_list.append(file_path)
    
    return files_list


def files_copy(root_old, root_new, file_type="txt"):
    """
    files move
    :param root_old: old files path
    :param root_new: new files path
    :param file_type: file type, default is "txt"
    """
    if not os.path.exists(root_new):
        os.mkdir(root_new)
    files_list_old = get_files_list(root_old, file_type)
    for file_path_old in files_list_old:
        file_name = get_file_name(file_path_old)
        file_path_new = fullfile(root_new, file_name)
        if not os.path.exists(file_path_new):
            shutil.copyfile(file_path_old, file_path_new)
        else:
            pass


def files_move(root_old, root_new, file_type="txt"):
    """
    files move
    :param root_old: old files path
    :param root_new: new files path
    :param file_type: file type, default is "txt"
    """
    if not os.path.exists(root_new):
        os.mkdir(root_new)
    files_list_old = get_files_list(root_old, file_type)
    for file_path_old in files_list_old:
        file_name = get_file_name(file_path_old)
        file_path_new = fullfile(root_new, file_name)
        if not os.path.exists(file_path_new):
            shutil.move(file_path_old, file_path_new)
        else:
            pass


if __name__ == "__main__":
    params_num = len(sys.argv)
    if params_num == 3:
        root_old = sys.argv[1]
        root_new = sys.argv[2]
        files_copy(root_old, root_new)
    elif params_num == 4:
        root_old = sys.argv[1]
        root_new = sys.argv[2]
        file_type = sys.argv[3]
        files_copy(root_old, root_new, file_type)
    else:
        print("Please input the command as the format of {python files_move.py \"root_old\" \"root_new\" \"file_type (defalut is \"txt\")\"} ")
