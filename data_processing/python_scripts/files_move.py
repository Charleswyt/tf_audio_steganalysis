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
        args_root_old = sys.argv[1]
        args_root_new = sys.argv[2]
        files_copy(args_root_old, args_root_new)
    elif params_num == 4:
        args_root_old = sys.argv[1]
        args_root_new = sys.argv[2]
        args_file_type = sys.argv[3]
        files_copy(args_root_old, args_root_new, args_file_type)
    else:
        print("Please input the command as the format of {python files_move.py \"root_old\" \"root_new\" \"file_type (defalut is \"txt\")\"} ")
