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

"""
function:
        command_parse()         : command-line arguments parse
        file_path_setup(args)   : get the data and model files path
"""


def command_parse():
    """
    :param: NULL
    :return:
        args -> Namespace(batch_size, bitrate, data_dir, epoch, gpu, learning_rate, log_dir, mode, model_dir, relative_payload, seed, stego_method)
    """
    parser = argparse.ArgumentParser(description="Audio steganalysis with CNN")
    print(parser.description)

    # get the current path
    current_path = os.getcwd()
    print("current path: %s" % current_path)
    print("The introduction of the parameters -- \"python3 **.py\" -h, more detailed information is in readme.md")

    # mode
    parser.add_argument("--mode", type=str, default="train", help="run mode -- train | test (default: train)")
    parser.add_argument("--test", type=str, default="one", help="one | batch (default one)")

    # data info
    parser.add_argument("--data_dir", type=str, help="data set path")
    parser.add_argument("--bitrate", type=int, default=128, help="the bitrate of MP3 files (default:128)")
    parser.add_argument("--relative_payload", type=str, default="2",
                        help="2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10, if the steganography algorithm is the EECS algorithm; 1 | 3 | 5 | 10, otherwise")
    parser.add_argument("--stego_method", type=str, default="EECS",
                        help="EECS | HCM-Gao | HCM- Yan (default: EECS)")
    parser.add_argument("--height", type=int, default=200, help="the height of the QMDCT matrix (default: 200)")
    parser.add_argument("--width", type=int, default=576, help="the width of the QMDCT matrix (default: 576)")
    parser.add_argument("--start_index", type=int, default=0, help="the start index of file in train folders (default: 0)")
    parser.add_argument("--end_index", type=int, default=10000, help="the end index of file in train folders (default: 10000)")
    parser.add_argument("--model_dir", type=str, help="model files path")
    parser.add_argument("--model_file_name", type=str, default="audio_steganalysis", help="model file name (default: audio_steganalysis)")
    parser.add_argument("--log_dir", type=str, help="log files path")

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--batch_size_train", type=int, default=128, help="batch size for train (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--batch_size_valid", type=int, default=128, help="batch size for valid (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--epoch", type=int, default=500, help="the number of epochs for training (default: 500)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="the value of initialized learning rate (default: 1e-3 (0.001))")
    parser.add_argument("--gpu", type=int, default=0, help="the index of gpu used (default: 0)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--decay_step", type=int, default=1000, help="the step for learning rate decay (default: 1000)")
    parser.add_argument("--decay_rate", type=float, default=0.95, help="the rate for learning rate decay (default: 0.95)")
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
    parser.add_argument("--models_path", type=str, default=current_path+"/models",
                        help="the path of directory containing models")
    parser.add_argument("--logs_path", type=str, default=current_path+"/logs",
                        help="the path of directory containing logs")
    parser.add_argument("--network", type=str, default="network1", help="the index of the network (default: network1), "
                                                                        "the detailed introduction of each network is in readme")

    # model
    parser.add_argument("--max_to_keep", type=int, default=3, help="the number needed to be saved (default: 3)")
    parser.add_argument("--keep_checkpoint_every_n_hours", type=float, default=0.5, help="how often to keep checkpoints (default: 0.5)")

    # pre-processing
    parser.add_argument("--is_abs", type=bool, default=False, help="abs or not (default: False)")
    parser.add_argument("--is_trunc", type=bool, default=False, help="truncation or not (default: False)")
    parser.add_argument("--threshold", type=int, default=15, help="threshold (default: 15)")
    parser.add_argument("--is_diff", type=bool, default=15, help="threshold (default: False)")
    parser.add_argument("--order", type=int, default=2, help="the order of the difference (default: 2)")
    parser.add_argument("--direction", type=int, default=0, help="0 - row, 1 - col (default: 0)")
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

    cover_train_files_path = data_dir + "/" + "cover/" + str(bitrate) + "/" + "train"
    cover_valid_files_path = data_dir + "/" + "cover/" + str(bitrate) + "/" + "valid"
    stego_train_files_path = data_dir + "/stego/" + stego_method + "/" + stego_dir_name + "/" + "train"
    stego_valid_files_path = data_dir + "/stego/" + stego_method + "/" + stego_dir_name + "/" + "valid"
    model_file_path = model_dir + "/" + stego_dir_name
    log_file_path = log_dir + "/" + stego_dir_name

    # input mode
    if args.cover_train_dir is not None:
        cover_train_files_path = args.cover_train_dir
    if args.cover_valid_dir is not None:
        cover_valid_files_path = args.cover_valid_dir
    if args.stego_train_dir is not None:
        stego_train_files_path = args.stego_train_dir
    if args.stego_train_dir is not None:
        stego_valid_files_path = args.stego_train_dir
    if args.models_path is not None:
        model_file_path = args.models_path
    if args.logs_path is not None:
        log_file_path = args.logs_path

    # make dir
    if os.path.exists(model_file_path) is False:
        os.mkdir(model_file_path)
    if os.path.exists(log_file_path) is False:
        os.mkdir(log_file_path)

    return cover_train_files_path, cover_valid_files_path, stego_train_files_path, stego_valid_files_path, model_file_path, log_file_path
