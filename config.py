#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018.01.05
Finished on 2018.01.05
@author: Wang Yuntao
"""

import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def command_parse():
    """
    :param: NULL
    :return:
        args -> Namespace(batch_size, bitrate, data_dir, epoch, gpu, learning_rate, log_dir, mode, model_dir, relative_payload, seed, stego_method)
    """
    parser = argparse.ArgumentParser(description="Audio steganalysis with CNN")
    print(parser.description)

    # mode
    parser.add_argument("--mode", type=str, default="train", help="run mode -- train | test (default: train)")
    parser.add_argument("--test", type=str, default="one", help="one | batch (default one)")

    # data info
    parser.add_argument("--data_dir", type=str, help="data set path")
    parser.add_argument("--bitrate", type=int, default=128, help="the bitrate of MP3 files (default:128)")
    parser.add_argument("--relative_payload",
                        type=str,
                        help="2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10, if the steganography algorithm is the EECS algorithm; 1 | 3 | 5 | 10, otherwise")
    parser.add_argument("--stego_method", type=str, default="EECS",
                        help="EECS | HCM-Gao | HCM- Yan  (default: EECS)")
    parser.add_argument("--model_dir", type=str, help="model files path")
    parser.add_argument("--log_dir", type=str, help="log files path")

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--batch_size_train", type=int, default=128, help="batch size for train (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--batch_size_valid", type=int, default=128, help="batch size for valid (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--epoch", type=int, default=500, help="the number of epochs for training (default: 500)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="the value of initialized learning rate (default: 1e-3 (0.001))")
    parser.add_argument("--gpu", type=int, default=0, help="the index of gpu used (default: 0)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default:1)")
    parser.add_argument("--decay_step", type=int, default=1000, help="the step for learning rate decay (default:1000)")
    parser.add_argument("--decay_rate", type=float, default=0.95, help="the rate for learning rate decay (default:0.95)")
    parser.add_argument("--staircase", type=bool, default=False, help="whether the decay the learning rate at discrete intervals or not (default:False)")

    # path
    parser.add_argument("--cover_train_dir", type=str,
                        help="the path of directory containing cover files for train")
    parser.add_argument("--cover_valid_dir", type=str,
                        help="the path of directory containing cover files for valid")
    parser.add_argument("--stego_train_dir", type=str,
                        help="the path of directory containing stego files for train")
    parser.add_argument("--stego_valid_dir", type=str,
                        help="the path of directory containing stego files for valid")
    parser.add_argument("--models_path", type=str,
                        help="the path of directory containing models")
    parser.add_argument("--logs_path", type=str,
                        help="the path of directory containing logs")
    parser.add_argument("--network", type=int, default=1, help="the index of the network (default:1), the detailed introduction of each network is in readme")
                                                               
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
    if args.cover_train_dir is not None         \
        and args.cover_valid_dir is not None    \
        and args.stego_train_dir is not None    \
        and args.stego_train_dir is not None    \
        and args.models_path is not None        \
        and args.logs_path is not None:
        cover_train_files_path = args.cover_train_dir
        cover_valid_files_path = args.cover_valid_dir
        stego_train_files_path = args.stego_train_dir
        stego_valid_files_path = args.stego_train_dir
        model_file_path = args.models_path

    else:
        data_dir = args.data_dir
        bitrate = args.bitrate
        stego_method = args.stego_method
        relative_payload = args.relative_payload
        model_dir = args.model_dir

        if stego_method == "EECS":
            stego_dir_name = str(bitrate) + "_" + relative_payload + "_H_7_ER_10"
        else:
            if int(relative_payload) < 10:
                stego_dir_name = str(bitrate) + "_0" + relative_payload
            else:
                stego_dir_name = str(bitrate) + "_" + relative_payload

        cover_train_files_path = data_dir + "/" + "cover/" + str(bitrate) + "/" + "train"
        cover_valid_files_path = data_dir + "/" + "cover/" + str(bitrate) + "/" + "valid"
        stego_train_files_path = data_dir + "/" + stego_method + "/" + stego_dir_name + "/" + "train"
        stego_valid_files_path = data_dir + "/" + stego_method + "/" + stego_dir_name + "/" + "valid"
        model_file_path = model_dir + "/" + stego_dir_name

    print("cover_train_files_path: ", cover_train_files_path)
    print("cover_valid_files_path: ", cover_valid_files_path)
    print("stego_train_files_path: ", stego_train_files_path)
    print("stego_valid_files_path: ", stego_valid_files_path)
    print("model_file_path: ", model_file_path)

    return cover_train_files_path, cover_valid_files_path, stego_train_files_path, stego_valid_files_path, model_file_path
