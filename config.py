#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018.01.05
Finished on 2018.01.05
@author: Wang Yuntao
"""

import os
import pip
import argparse
import platform
from subprocess import call

"""
function:
        command_parse()                                     parse command line parameters (命令行解析)
        file_path_setup(args)                               get file path (获取文件路径)
        get_files_list(file_dir, start_idx=0, end_idx=-1)   get file list (获取文件列表)
        get_packages()                                      get installed packages in the current machine (获取当前系统安装的库文件)
        show_package(_packages)                             show all installed  packages in the current machine (显示当前系统安装的库文件)
        package_download(packages_name)                     download defect packages automatically (如果缺少支持程序运行的包则自动下载)
        package_upgrade(package_name)                       update the package in the current machine (更新当前系统安装的指定库文件)
        packages_upgrade()                                  update all packages in the current machine (批量更新当前系统安装的所有库文件)
"""

"""
    在tensorflow的log日志等级如下： 
    - 0：显示所有日志（默认等级） 
    - 1：显示info、warning和error日志 
    - 2：显示warning和error信息 
    - 3：显示error日志信息 
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
needed_packages = ["tensorflow-gpu", "numpy", "matplotlib"]
system = platform.system()


def command_parse():
    """
    :param: NULL
    :return:
        argument
    """
    parser = argparse.ArgumentParser(description="Audio/Image steganalysis with CNN based on tensorflow.")
    print(parser.description)

    # get the current path
    current_path = os.getcwd()
    print("current path: %s" % current_path)

    # mode
    parser.add_argument("--gpu_selection", type=str, default="auto",
                        help="GPU selection mode, if \"auto\", no serial number is needed, otherwise appoint the serial number "
                             "(default: auto, other choice is manual)")
    parser.add_argument("--gpu", type=str, default="0", help="the index of GPU")
    parser.add_argument("--mode", type=str, default="train", help="run mode -- train | test (default: train)")
    parser.add_argument("--submode", type=str, default="one", help="one | batch (default one)")
    parser.add_argument("--carrier", type=str, default="audio", help="image | audio (default audio)")
    parser.add_argument("--network", type=str, default="network1", help="the index of the network (default: network1), "
                                                                        "the detailed introduction of each network is in readme")

    # data info
    parser.add_argument("--data_dir", type=str, help="data set path")
    parser.add_argument("--bitrate", type=int, default=128, help="the bitrate of MP3 files (default:128)")
    parser.add_argument("--relative_payload", type=str, default="2",
                        help="2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10, if the steganography algorithm is the EECS algorithm; 1 | 3 | 5 | 10, otherwise")
    parser.add_argument("--stego_method", type=str, default="EECS",
                        help="EECS | HCM-Gao | HCM- Yan (default: EECS)")
    parser.add_argument("--height", type=int, default=200, help="the height of the QMDCT matrix (default: 200)")
    parser.add_argument("--width", type=int, default=576, help="the width of the QMDCT matrix (default: 576)")
    parser.add_argument("--channel", type=int, default=1, help="the channel of the QMDCT matrix (default: 1)")
    parser.add_argument("--start_index_train", type=int, default=0, help="the start index of file in train folders (default: 0)")
    parser.add_argument("--end_index_train", type=int, default=16000, help="the end index of file in train folders (default: 16000)")
    parser.add_argument("--start_index_valid", type=int, default=0, help="the start index of file in valid folders (default: 0)")
    parser.add_argument("--end_index_valid", type=int, default=4000, help="the end index of file in valid folders (default: 4000)")
    parser.add_argument("--model_dir", type=str, help="model files path")
    parser.add_argument("--model_file_name", type=str, default="audio_steganalysis", help="model file name (default: audio_steganalysis)")
    parser.add_argument("--log_dir", type=str, help="log files path")

    # test
    parser.add_argument("--model_file_path", type=str, default=None, help="the model file path used for test (default is None)")
    parser.add_argument("--test_file_path", type=str, help="the file path used for test")
    parser.add_argument("--label", type=str, default=None, help="the label of tested file (default is None)")
    parser.add_argument("--test_files_path", type=str, help="the files folder path used for test")
    parser.add_argument("--label_file_path", type=str, help="the label file path used for test")

    # path
    parser.add_argument("--cover_train_path", type=str,
                        help="the path of directory containing cover files for train")
    parser.add_argument("--cover_valid_path", type=str,
                        help="the path of directory containing cover files for valid")
    parser.add_argument("--cover_test_path", type=str,
                        help="the path of directory containing cover files for test")
    parser.add_argument("--stego_train_path", type=str,
                        help="the path of directory containing stego files for train")
    parser.add_argument("--stego_valid_path", type=str,
                        help="the path of directory containing stego files for valid")
    parser.add_argument("--stego_test_path", type=str,
                        help="the path of directory containing stego files for test")
    parser.add_argument("--models_path", type=str, default=current_path + "/models",
                        help="the path of directory containing models")
    parser.add_argument("--logs_path", type=str, default=current_path + "/logs",
                        help="the path of directory containing logs")

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--batch_size_train", type=int, default=64, help="batch size for train (default: 64 (32 cover/stego pairs))")
    parser.add_argument("--batch_size_valid", type=int, default=64, help="batch size for valid (default: 64 (32 cover/stego pairs))")
    parser.add_argument("--batch_size_test", type=int, default=64, help="batch size for test (default: 64 (32 cover/stego pairs))")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="the value of initialized learning rate (default: 1e-3 (0.001))")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--is_regulation", type=bool, default=True, help="whether regulation or not (default: True)")
    parser.add_argument("--coeff_regulation", type=float, default=1e-3, help="the gain of regulation (default: 1e-3)")
    parser.add_argument("--loss_method", type=bool, default="sparse_softmax_cross_entropy",
                        help="the method of loss calculation (default: sparse_softmax_cross_entropy)")
    parser.add_argument("--class_num", type=int, default=2, help="the class number (default: 2)")
    parser.add_argument("--epoch", type=int, default=500, help="the number of epochs for network training stop (default: 500)")

    # learning rate parameters
    parser.add_argument("--decay_method", type=str, default="exponential", help="the method for learning rate decay (default: exponential)")
    parser.add_argument("--decay_step", type=int, default=5000, help="the step for learning rate decay (default: 5000)")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="the rate for learning rate decay (default: 0.95)")
    parser.add_argument("--staircase", type=bool, default=False,
                        help="whether the decay the learning rate at discrete intervals or not (default:False)")

    # model
    parser.add_argument("--max_to_keep", type=int, default=3, help="the number needed to be saved (default: 3)")
    parser.add_argument("--keep_checkpoint_every_n_hours", type=float, default=0.5,
                        help="how often to keep checkpoints (default: 0.5)")

    # pre-processing
    parser.add_argument("--is_abs", type=bool, default=False, help="abs or not (default: False)")
    parser.add_argument("--is_trunc", type=bool, default=False, help="truncation or not (default: False)")
    parser.add_argument("--threshold", type=int, default=15, help="threshold (default: 15)")
    parser.add_argument("--is_diff", type=bool, default=False, help="threshold (default: False)")
    parser.add_argument("--order", type=int, default=2, help="the order of the difference (default: 2)")
    parser.add_argument("--direction", type=int, default=0, help="0 - row, 1 - col (default: 0)")
    parser.add_argument("--is_diff_abs", type=bool, default=False, help="abs or not after difference (default: False)")
    parser.add_argument("--downsampling", type=bool, default=False, help="downsampling or not (default: False)")
    parser.add_argument("--block", type=int, default=2, help=" (default: 2)")

    arguments = parser.parse_args()

    return arguments


def file_path_setup(args):
    """
    :param args: command-line arguments
    :return:
        cover_train_files_path: the folder of cover files for train
        cover_valid_files_path: the folder of cover files for valid
        stego_train_files_path: the folder of stego files for train
        stego_valid_files_path: the folder of stego files for valid
        model_file_path       : the folder of model files
    """
    # default
    data_dir = args.data_dir                                            # data dir, e.g. E:\Myself\2.database\4.stego
    bitrate = args.bitrate                                              # bitrate, e.g. 128
    stego_method = args.stego_method                                    # stego_method, e.g. EECS
    relative_payload = args.relative_payload                            # relative payload, e.g. 2
    model_dir = args.model_dir                                          # model dir
    log_dir = args.log_dir                                              # log dir

    if stego_method == "EECS":
        stego_dir_name = str(bitrate) + "_W_" + relative_payload + "_H_7_ER_10"
    else:
        if int(relative_payload) < 10:
            stego_dir_name = str(bitrate) + "_0" + relative_payload
        else:
            stego_dir_name = str(bitrate) + "_" + relative_payload

    if data_dir is not None:
        cover_train_files_path = data_dir + "/" + "cover/" + str(bitrate) + "/" + "train"
        cover_valid_files_path = data_dir + "/" + "cover/" + str(bitrate) + "/" + "valid"
        stego_train_files_path = data_dir + "/stego/" + stego_method + "/" + stego_dir_name + "/" + "train"
        stego_valid_files_path = data_dir + "/stego/" + stego_method + "/" + stego_dir_name + "/" + "valid"
        model_file_path = model_dir + "/" + stego_dir_name
        log_file_path = log_dir + "/" + stego_dir_name
    else:
        cover_train_files_path = args.cover_train_path
        cover_valid_files_path = args.cover_valid_path
        stego_train_files_path = args.stego_train_path
        stego_valid_files_path = args.stego_valid_path
        model_file_path = args.models_path
        log_file_path = args.logs_path

    # make dir
    if os.path.exists(model_file_path) is False:
        os.mkdir(model_file_path)
    if os.path.exists(log_file_path) is False:
        os.mkdir(log_file_path)

    return cover_train_files_path, cover_valid_files_path, stego_train_files_path, stego_valid_files_path, model_file_path, log_file_path


def get_files_list(file_dir, start_idx=0, end_idx=10000):
    """
    get the files list
    :param file_dir: file directory
    :param start_idx: start index
    :param end_idx: end index
    :return:
    """
    filename = os.listdir(file_dir)
    file_list = [file_dir + "/" + file for file in filename]
    total_num = len(file_list)
    if start_idx > total_num:
        start_idx = 0
    if end_idx > total_num:
        end_idx = total_num + 1

    file_list = file_list[start_idx:end_idx]

    return file_list


def fullfile(file_dir, file_path):
    """
    full file path based on os.path.join
    :param file_dir: dir
    :param file_path: file path
    :return:
        path: final path
    """
    path = os.path.join(file_dir, file_path)
    path = path.replace("\\", "/")

    return path


def get_packages():
    """
    get the
    :return:
    """
    _packages = list()
    for distribution in pip.get_installed_distributions():
        package_name = distribution.project_name
        _packages.append(package_name)

    return _packages


def show_package(_packages):
    for i in range(len(_packages)):
        print(_packages[i])


def package_download(packages_name):
    _packages = get_packages()
    if isinstance(packages_name, str) is True:
        packages_list = list()
        packages_list.append(packages_name)
    else:
        packages_list = packages_name
    for package in packages_list:
        if package not in _packages:
            if system == "Windows":
                call("pip3 install " + package, shell=True)
            if system == "Linux":
                call("sudo pip3 install " + package, shell=True)


def package_upgrade(package_name):
    _input = input("Are you sure to upgrade package %s (Y or N): " % package_name)
    if _input == "Y" or _input == "y":
        if system == "Windows":
            call("pip3 install --upgrade " + package_name, shell=False)
        if system == "Linux":
            call("sudo pip3 install --upgrade " + package_name, shell=False)
    elif _input == "N" or _input == "n":
        print("upgrade quit.")
    else:
        print("input error.")


def packages_upgrade():
    _input = input("Are you sure (Y or N): ")
    if _input == "Y" or _input == "y":
        for dist in pip.get_installed_distributions():
            if system == "Windows":
                call("pip3 install --upgrade " + dist.project_name, shell=False)
            if system == "Linux":
                call("sudo pip3 install --upgrade " + dist.project_name, shell=False)
    elif _input == "N" or _input == "n":
        print("upgrade quit.")
    else:
        print("input error.")


if __name__ == "__main__":
    print(__doc__)
