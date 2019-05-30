#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.08.13
Finished on 2018.08.14
Modified on 
@author: Yuntao Wang
"""

import os
import sys
import shutil
from utils import *


def files_copy(directory_old, directory_new, file_type="txt"):
    """
    files move
    :param directory_old: old files path
    :param directory_new: new files path
    :param file_type: file type, default is "txt"
    """
    if not os.path.exists(directory_new):
        os.mkdir(directory_new)
    files_list_old = get_files_list(directory_old, file_type)
    for file_path_old in files_list_old:
        file_name = get_file_name(file_path_old)
        file_path_new = fullfile(directory_new, file_name)
        if not os.path.exists(file_path_new):
            shutil.copyfile(file_path_old, file_path_new)
        else:
            pass


def files_move(directory_old, directory_new, file_type="txt"):
    """
    files move
    :param directory_old: old files path
    :param directory_new: new files path
    :param file_type: file type, default is "txt"
    """
    if not os.path.exists(directory_new):
        os.mkdir(directory_new)
    files_list_old = get_files_list(directory_old, file_type)
    for file_path_old in files_list_old:
        file_name = get_file_name(file_path_old)
        file_path_new = fullfile(directory_new, file_name)
        if not os.path.exists(file_path_new):
            shutil.move(file_path_old, file_path_new)
        else:
            pass


if __name__ == "__main__":
    params_num = len(sys.argv)
    if params_num == 3:
        args_directory_old = sys.argv[1]
        args_directory_new = sys.argv[2]
        files_copy(args_directory_old, args_directory_new)
    elif params_num == 4:
        args_directory_old = sys.argv[1]
        args_directory_new = sys.argv[2]
        args_file_type = sys.argv[3]
        files_copy(args_directory_old, args_directory_new, args_file_type)
    else:
        print("Please input the command as the format of {python files_move.py \"directory_old\" \"directory_new\" \"file_type (defalut is \"txt\")\"} ")
