#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018.01.05
Finished on 2018.01.05
Modified on 2018.08.23

@author: Wang Yuntao
"""

import os
import json
import argparse

"""
function:
        command_parse()                                     parse command line parameters
        config_train_file_read                              read config json file for trainng
        config_test_file_read                               read config json file for test
        config_steganalysis_file_read                       read config json file for steganalysis
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def command_parse():
    """
    :param: NULL
    :return:
        argument
    """
    parser = argparse.ArgumentParser(description="Audio/Image steganalysis with CNN based on tensorflow.")
    print(parser.description)

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
    parser.add_argument("--height", type=int, default=200, help="the height of the QMDCT matrix (default: 200)")
    parser.add_argument("--width", type=int, default=576, help="the width of the QMDCT matrix (default: 576)")
    parser.add_argument("--channel", type=int, default=1, help="the channel of the QMDCT matrix (default: 1)")
    parser.add_argument("--start_index_train", type=int, default=None, help="the start index of file in train folders (default: None)")
    parser.add_argument("--end_index_train", type=int, default=None, help="the end index of file in train folders (default: None)")
    parser.add_argument("--start_index_valid", type=int, default=None, help="the start index of file in valid folders (default: None)")
    parser.add_argument("--end_index_valid", type=int, default=None, help="the end index of file in valid folders (default: None)")
    parser.add_argument("--model_file_name", type=str, default="audio_steganalysis", help="model file name (default: audio_steganalysis)")

    # test and steganalysis
    parser.add_argument("--steganalysis_file_path", type=str, help="the file path used for steganalysis")
    parser.add_argument("--steganalysis_files_path", type=str, help="the files folder path used for steganalysis")

    # path of files
    parser.add_argument("--cover_train_path", type=str, help="the path of directory containing cover files for train")
    parser.add_argument("--cover_valid_path", type=str, help="the path of directory containing cover files for valid")
    parser.add_argument("--cover_test_path", type=str, help="the path of directory containing cover files for test")
    parser.add_argument("--stego_train_path", type=str, help="the path of directory containing stego files for train")
    parser.add_argument("--stego_valid_path", type=str, help="the path of directory containing stego files for valid")
    parser.add_argument("--stego_test_path", type=str, help="the path of directory containing stego files for test")
    parser.add_argument("--model_path", type=str, help="the path of directory containing models")
    parser.add_argument("--log_path", type=str, help="the path of directory containing logs")

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size (default: 128 (64 cover|stego pairs))")
    parser.add_argument("--batch_size_train", type=int, default=64, help="batch size for train (default: 64 (32 cover/stego pairs))")
    parser.add_argument("--batch_size_valid", type=int, default=64, help="batch size for valid (default: 64 (32 cover/stego pairs))")
    parser.add_argument("--batch_size_test", type=int, default=64, help="batch size for test (default: 64 (32 cover/stego pairs))")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="the value of initialized learning rate (default: 1e-3 (0.001))")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--is_regulation", type=bool, default=True, help="whether regulation or not (default: True)")
    parser.add_argument("--coeff_regulation", type=float, default=1e-3, help="the gain of regulation (default: 1e-3)")
    parser.add_argument("--loss_method", type=bool, default="sparse_softmax_cross_entropy", help="the method of loss calculation (default: sparse_softmax_cross_entropy)")
    parser.add_argument("--class_num", type=int, default=2, help="the class number (default: 2)")
    parser.add_argument("--epoch", type=int, default=500, help="the number of epochs for network training stop (default: 500)")

    # learning rate parameters
    parser.add_argument("--decay_method", type=str, default="exponential", help="the method for learning rate decay (default: exponential)")
    parser.add_argument("--decay_step", type=int, default=5000, help="the step for learning rate decay (default: 5000)")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="the rate for learning rate decay (default: 0.95)")
    parser.add_argument("--staircase", type=bool, default=False,
                        help="whether the decay the learning rate at discrete intervals or not (default:False)")

    # model
    parser.add_argument("--max_to_keep", type=int, default=3, help="the number of models needed to be saved (default: 3)")
    parser.add_argument("--keep_checkpoint_every_n_hours", type=float, default=0.5, help="how often to keep checkpoints (default: 0.5)")

    # pre-processing
    parser.add_argument("--is_abs", type=bool, default=False, help="abs or not (default: False)")
    parser.add_argument("--is_trunc", type=bool, default=False, help="truncation or not (default: False)")
    parser.add_argument("--threshold", type=int, default=15, help="threshold (default: 15)")
    parser.add_argument("--is_diff", type=bool, default=False, help="threshold (default: False)")
    parser.add_argument("--order", type=int, default=2, help="the order of the difference (default: 2)")
    parser.add_argument("--direction", type=int, default=0, help="0 - row, 1 - col (default: 0)")
    parser.add_argument("--is_diff_abs", type=bool, default=False, help="abs or not after difference (default: False)")
    parser.add_argument("--is_abs_diff", type=bool, default=False, help="difference or not after abs (default: False)")
    parser.add_argument("--downsampling", type=bool, default=False, help="downsampling or not (default: False)")
    parser.add_argument("--block", type=int, default=2, help=" (default: 2)")

    arguments = parser.parse_args()

    return arguments


def config_train_file_read(config_file_path):
    with open(config_file_path, encoding='utf-8') as json_file:
        file_content = json.load(json_file)

        class Variable:
            def __init__(self):
                # files_path
                self.cover_train_path = file_content['files_path']['cover_train_path']
                self.cover_valid_path = file_content['files_path']['cover_valid_path']
                self.stego_train_path = file_content['files_path']['stego_train_path']
                self.stego_valid_path = file_content['files_path']['stego_valid_path']
                self.model_path = file_content['files_path']['model_path']
                self.log_path = file_content['files_path']['log_path']
                self.model_file_name = file_content['files_path']['model_file_name']

                # mode_config
                self.gpu_selection = file_content['mode_config']['gpu_selection']
                self.gpu = file_content['mode_config']['gpu']
                self.mode = file_content['mode_config']['mode']
                self.carrier = file_content['mode_config']['carrier']
                self.network = file_content['mode_config']['network']

                # hyper_parameters
                self.batch_size_train = file_content['hyper_parameters']['batch_size_train']
                self.batch_size_valid = file_content['hyper_parameters']['batch_size_valid']
                self.learning_rate = file_content['hyper_parameters']['learning_rate']
                self.epoch = file_content['hyper_parameters']['epoch']
                self.seed = file_content['hyper_parameters']['seed']
                self.is_regulation = file_content['hyper_parameters']['is_regulation']
                self.coeff_regulation = file_content['hyper_parameters']['coeff_regulation']
                self.loss_method = file_content['hyper_parameters']['loss_method']
                self.class_num = file_content['hyper_parameters']['class_num']

                # preprocessing_method
                self.height = file_content['preprocessing_method']['height']
                self.width = file_content['preprocessing_method']['width']
                self.channel = file_content['preprocessing_method']['channel']
                self.is_abs = file_content['preprocessing_method']['is_abs']
                self.is_trunc = file_content['preprocessing_method']['is_trunc']
                self.is_diff = file_content['preprocessing_method']['is_diff']
                self.is_diff_abs = file_content['preprocessing_method']['is_diff_abs']
                self.is_abs_diff = file_content['preprocessing_method']['is_abs_diff']
                self.threshold = file_content['preprocessing_method']['threshold']
                self.order = file_content['preprocessing_method']['order']
                self.direction = file_content['preprocessing_method']['direction']

                # learning_rate_method
                self.decay_method = file_content['learning_rate_method']['decay_method']
                self.decay_step = file_content['learning_rate_method']['decay_step']
                self.decay_rate = file_content['learning_rate_method']['decay_rate']
                self.staircase = file_content['learning_rate_method']['staircase']

                # model
                self.max_to_keep = file_content['model']['max_to_keep']
                self.keep_checkpoint_every_n_hours = file_content['model']['keep_checkpoint_every_n_hours']

                # index
                self.start_index_train = file_content["index"]["start_index_train"]
                self.end_index_train = file_content["index"]["end_index_train"]
                self.start_index_valid = file_content["index"]["start_index_valid"]
                self.end_index_valid = file_content["index"]["end_index_valid"]

        argument = Variable()

        return argument


def config_test_file_read(config_file_path):
    with open(config_file_path, encoding='utf-8') as json_file:
        file_content = json.load(json_file)

        class Variable:
            def __init__(self):
                # files_path
                self.cover_test_path = file_content['files_path']['cover_test_path']
                self.stego_test_path = file_content['files_path']['stego_test_path']
                self.model_path = file_content['files_path']['model_path']

                # mode_config
                self.gpu_selection = file_content['mode_config']['gpu_selection']
                self.gpu = file_content['mode_config']['gpu']
                self.mode = file_content['mode_config']['mode']
                self.carrier = file_content['mode_config']['carrier']
                self.network = file_content['mode_config']['network']

                # hyper_parameters
                self.batch_size_test = file_content['hyper_parameters']['batch_size_test']
                self.class_num = file_content['hyper_parameters']['class_num']

                # preprocessing_method
                self.height = file_content['preprocessing_method']['height']
                self.width = file_content['preprocessing_method']['width']
                self.channel = file_content['preprocessing_method']['channel']
                self.is_abs = file_content['preprocessing_method']['is_abs']
                self.is_trunc = file_content['preprocessing_method']['is_trunc']
                self.is_diff = file_content['preprocessing_method']['is_diff']
                self.is_diff_abs = file_content['preprocessing_method']['is_diff_abs']
                self.is_abs_diff = file_content['preprocessing_method']['is_abs_diff']
                self.threshold = file_content['preprocessing_method']['threshold']
                self.order = file_content['preprocessing_method']['order']
                self.direction = file_content['preprocessing_method']['direction']

        argument = Variable()

        return argument


def config_steganalysis_file_read(config_file_path):
    with open(config_file_path, encoding='utf-8') as json_file:
        file_content = json.load(json_file)

        class Variable:
            def __init__(self):
                # files_path
                self.files_path = file_content['files_path']['files_path']
                self.model_path = file_content['files_path']['model_path']

                # mode_config
                self.gpu_selection = file_content['mode_config']['gpu_selection']
                self.gpu = file_content['mode_config']['gpu']
                self.mode = file_content['mode_config']['mode']
                self.carrier = file_content['mode_config']['carrier']
                self.network = file_content['mode_config']['network']

                # hyper_parameters
                self.class_num = file_content['hyper_parameters']['class_num']

                # preprocessing_method
                self.height = file_content['preprocessing_method']['height']
                self.width = file_content['preprocessing_method']['width']
                self.channel = file_content['preprocessing_method']['channel']
                self.is_abs = file_content['preprocessing_method']['is_abs']
                self.is_trunc = file_content['preprocessing_method']['is_trunc']
                self.is_diff = file_content['preprocessing_method']['is_diff']
                self.is_diff_abs = file_content['preprocessing_method']['is_diff_abs']
                self.is_abs_diff = file_content['preprocessing_method']['is_abs_diff']
                self.threshold = file_content['preprocessing_method']['threshold']
                self.order = file_content['preprocessing_method']['order']
                self.direction = file_content['preprocessing_method']['direction']

        argument = Variable()

        return argument
