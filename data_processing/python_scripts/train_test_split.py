#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.08.21
Finished on 2018.08.21
Modified on 
@author: Yuntao Wang
"""

import os
import sys
import shutil


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


def get_files_list(files_path, file_type=None):
    """
    :param files_path: path of MP3 files for move
    :param file_type: file type, default is None
    :return: Null
    """
    filename = os.listdir(files_path)
    files_list = []
    for file in filename:
        file_path = fullfile(files_path, file)
        if get_file_type(file_path) == file_type or file_type is None:
            files_list.append(file_path)

    return files_list


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


def files_move(files_list, files_path_new):
    """
    files move
    :param files_list: files list
    :param files_path_new: new files path
    :return
        NULL
    """
    for file_path in files_list:
        file_name = get_file_name(file_path)
        file_path_new = fullfile(files_path_new, file_name)
        if not os.path.exists(file_path_new):
            shutil.move(file_path, file_path_new)
        else:
            pass


def make_folder(files_path):
    """
        create folder
        :param files_path path of files to be created
        :return
            NULL
    """
    if not os.path.exists(files_path):
        os.mkdir(files_path)
    else:
        pass
    

def train_test_split(files_path, percent_train=0.7, percent_validation=0.3):
    """
        split the dataset into train, validation and test
        :param files_path: path of files to be spilted
        :param start_idx: start index of audio files
        :param end_idx: end index of audio files
        :param percent_train: percent of train data
        :param percent_validation: percent of validation data
        :param percent_test: percent of test data
        :return:
            NULL
    """
    if not percent_train + percent_validation == 1:
        print("The sum of percent of all split data is not 100%, please try again.")
    else:
        files_list = get_files_list(files_path)
        files_num = len(files_list)
        data_num_test = files_num % 10000
        train_validation_num = files_num - data_num_test
        
        data_num_train = int(train_validation_num * percent_train)
        data_num_valiadation = int(train_validation_num * percent_validation)
        
        # split the dataset
        files_list_train = files_list[:data_num_train]
        del files_list[:data_num_train]

        files_list_validation = files_list[:data_num_valiadation]
        del files_list[:data_num_valiadation]

        files_list_test = files_list[:data_num_test]
        del files_list[:data_num_test]

        # mkdir
        files_path_train = fullfile(files_path, "train")
        files_path_validation = fullfile(files_path, "validation")
        files_path_test = fullfile(files_path, "test")

        make_folder(files_path_train)
        make_folder(files_path_validation)
        make_folder(files_path_test)

        # file move
        print(files_list_train[0])
        files_move(files_list_train, files_path_train)
        files_move(files_list_validation, files_path_validation)
        files_move(files_list_test, files_path_test)


if __name__ == "__main__":
    params_num = len(sys.argv)
    if params_num == 2:
        files_path = sys.argv[1]
        train_test_split(files_path)
    elif params_num == 3:
        files_path = sys.argv[1]
        percent_train = float(sys.argv[2])
        percent_validation = 1 - float(percent_train)
        train_test_split(files_path, percent_train, percent_validation)
    elif params_num == 4:
        files_path = sys.argv[1]
        percent_train = float(sys.argv[2])
        percent_validation = float(sys.argv[3])
        train_test_split(files_path, percent_train, percent_validation)
    else:
        print("Please input the command as the format of {python train_test_split.py \"files_path\" \"percent_train\" \"percent_validation\"}")
