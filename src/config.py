#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018.01.05
Finished on 2018.01.05
Modified on 2018.08.23

@author: Yuntao Wang
"""

import os
import json
import argparse
from utils import *

"""
function:
        command_parse()                                     parse command line parameters
        config_train_file_read                              read config json file for training
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
    parser.add_argument("--path_mode", type=str, default="full", help="mode of file path")
    parser.add_argument("--task_name", type=str, help="the name of the current task")
    parser.add_argument("--gpu_selection", type=str, default="auto",
                        help="GPU selection mode, if \"auto\", no serial number is needed, otherwise appoint the serial number "
                             "(default: auto, another choice is manu)")
    parser.add_argument("--gpu", type=str, default="0", help="the index of GPU")
    parser.add_argument("--mode", type=str, default="train", help="run mode -- train | test (default: train)")
    parser.add_argument("--siamese", type=bool, default=False, help="whether to use siamese mode (default: False)")
    parser.add_argument("--checkpoint", type=bool, default=True, help="whether there is checkpoint or not (default: True)")
    parser.add_argument("--submode", type=str, default="one", help="one | batch (default one)")
    parser.add_argument("--carrier", type=str, default="qmdct", help="qmdct | image | audio (default qmdct)")
    parser.add_argument("--network", type=str, default="network1", help="the index of the network (default: wasdn), "
                                                                        "the detailed introduction of each network is in readme")

    # data info
    parser.add_argument("--height", type=int, default=200, help="the height of the input data matrix (default: 200)")
    parser.add_argument("--width", type=int, default=576, help="the width of the input data matrix (default: 576)")
    parser.add_argument("--channel", type=int, default=1, help="the channel of the input data matrix (default: 1)")

    # index
    parser.add_argument("--start_index_train", type=int, default=None, help="the start index of file in train folders (default: None)")
    parser.add_argument("--end_index_train", type=int, default=None, help="the end index of file in train folders (default: None)")
    parser.add_argument("--start_index_valid", type=int, default=None, help="the start index of file in valid folders (default: None)")
    parser.add_argument("--end_index_valid", type=int, default=None, help="the end index of file in valid folders (default: None)")

    # path of steganalysis file(s)
    parser.add_argument("--cover_files_root", type=str, help="the directory of root containing cover files")
    parser.add_argument("--stego_files_root", type=str, help="the directory of root containing stego files")

    parser.add_argument("--cover_files_path", type=str, default=None, help="the directory of root containing cover files")
    parser.add_argument("--stego_files_path", type=str, default=None, help="the directory of root containing stego files")

    parser.add_argument("--steganalysis_file_path", type=str, help="the file path used for steganalysis")
    parser.add_argument("--steganalysis_files_path", type=str, help="the files folder path used for steganalysis")
    parser.add_argument("--cover_train_path", type=str, help="the path of directory containing cover files for train")
    parser.add_argument("--cover_valid_path", type=str, help="the path of directory containing cover files for validation")
    parser.add_argument("--cover_test_path", type=str, help="the path of directory containing cover files for test")
    parser.add_argument("--stego_train_path", type=str, help="the path of directory containing stego files for train")
    parser.add_argument("--stego_valid_path", type=str, help="the path of directory containing stego files for validation")
    parser.add_argument("--stego_test_path", type=str, help="the path of directory containing stego files for test")
    parser.add_argument("--tfrecords_path", type=str, help="the path of directory containing all tfrecord files")
    parser.add_argument("--models_path", type=str, help="the path of directory containing models")
    parser.add_argument("--logs_path", type=str, help="the path of directory containing logs")

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size (default: 128 (64 cover|stego pairs))")
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

    arguments = parser.parse_args()

    # full_samples_path
    if arguments.path_mode == "full" or arguments.path_mode == "semi":
        pass

    # simple_samples_path
    elif self.path_mode == "simple":
        stego_method = arguments.task_name.split("_")[0]
        samples_bitrate = arguments.task_name.split("_")[2]

        arguments.cover_train_path = fullfile(fullfile(arguments.cover_files_root, samples_bitrate), "train")
        arguments.cover_valid_path = fullfile(fullfile(arguments.cover_files_root, samples_bitrate), "validation")
        arguments.stego_train_path = fullfile(fullfile(fullfile(arguments.stego_files_root, stego_method), arguments.task_name), "train")
        arguments.stego_valid_path = fullfile(fullfile(fullfile(arguments.stego_files_root, stego_method), arguments.task_name), "validation")

        arguments.cover_files_path, arguments.stego_files_path = None, None

    else:
        arguments.train = False

    # create folder (tfrecord)
    # level: root
    tfrecords_path = arguments.tfrecords_path
    folder_make(tfrecords_path)

    # level: task
    tfrecords_path_task = fullfile(tfrecords_path, arguments.task_name)
    folder_make(tfrecords_path_task)

    # create folder (models and logs)
    # level: root
    models_path_root, logs_path_root = arguments.models_path, arguments.logs_path
    folder_make(models_path_root), folder_make(logs_path_root)

    # level: network
    models_path_network, logs_path_network = fullfile(models_path_root, arguments.network), fullfile(logs_path_root, arguments.network)
    folder_make(models_path_network), folder_make(logs_path_network)

    # level: task
    models_path_task, logs_path_task = fullfile(models_path_network, arguments.task_name), fullfile(logs_path_network, arguments.task_name)
    folder_make(models_path_task), folder_make(logs_path_task)

    # process for checkpoint
    sub_directory = get_sub_directory(models_path_task)
    if arguments.checkpoint is True and len(sub_directory) > 0:
        models_path_current = sub_directory[-1]
        logs_path_current = models_path_current.replace("models", "logs")
    else:
        # level: training start time
        current_time_stamp = str(get_unix_stamp(get_time()))
        models_path_current, logs_path_current = fullfile(models_path_task, current_time_stamp), fullfile(logs_path_task, current_time_stamp)
        folder_make(models_path_current), folder_make(logs_path_current)

    arguments.model_path = models_path_current
    arguments.log_path = logs_path_current

    return arguments


def config_train_file_read(config_file_path):
    with open(config_file_path, encoding='utf-8') as json_file:
        file_content = json.load(json_file)

        class Variable:
            def __init__(self):
                self.path_mode = file_content["path_mode"]
                self.task_name = file_content["task_name"]

                # full_samples_path
                if self.path_mode == "full":
                    self.cover_train_path = file_content['full_samples_path']['cover_train_path']
                    self.cover_valid_path = file_content['full_samples_path']['cover_valid_path']
                    self.stego_train_path = file_content['full_samples_path']['stego_train_path']
                    self.stego_valid_path = file_content['full_samples_path']['stego_valid_path']

                # semi_sample_path
                elif self.path_mode == "semi":
                    self.cover_files_path = file_content['semi_samples_path']['cover_files_path']
                    self.stego_files_path = file_content['semi_samples_path']['stego_files_path']

                    self.cover_train_path, self.cover_valid_path, self.stego_train_path, self.stego_valid_path = None, None, None, None

                # simple_samples_path
                elif self.path_mode == "simple":
                    self.cover_files_root = file_content['simple_samples_path']['cover_files_root']
                    self.stego_files_root = file_content['simple_samples_path']['stego_files_root']
                    stego_method = self.task_name.split("_")[0]
                    samples_bitrate = self.task_name.split("_")[2]

                    self.cover_train_path = fullfile(fullfile(self.cover_files_root, samples_bitrate), "train")
                    self.cover_valid_path = fullfile(fullfile(self.cover_files_root, samples_bitrate), "validation")
                    self.stego_train_path = fullfile(fullfile(fullfile(self.stego_files_root, stego_method), self.task_name), "train")
                    self.stego_valid_path = fullfile(fullfile(fullfile(self.stego_files_root, stego_method), self.task_name), "validation")
                    self.cover_files_path, self.stego_files_path = None, None

                else:
                    self.train = False

                # files_path
                self.tfrecords_path = file_content['files_path']['tfrecords_path']
                self.models_path = file_content['files_path']['models_path']
                self.logs_path = file_content['files_path']['logs_path']

                # mode_config
                self.gpu_selection = file_content['mode_config']['gpu_selection']
                self.gpu = file_content['mode_config']['gpu']
                self.mode = file_content['mode_config']['mode']
                self.checkpoint = file_content['mode_config']['checkpoint']
                self.carrier = file_content['mode_config']['carrier']
                self.network = file_content['mode_config']['network']
                self.siamese = file_content['mode_config']['siamese']

                if self.carrier == "mfcc" or self.carrier == "audio":
                    self.file_type = "mp3"
                elif self.carrier == "qmdct":
                    self.file_type = "txt"
                else:
                    self.file_type = self.carrier

                # hyper_parameters
                self.batch_size = file_content['hyper_parameters']['batch_size']
                self.learning_rate = file_content['hyper_parameters']['learning_rate']
                self.seed = file_content['hyper_parameters']['seed']
                self.epoch = file_content['hyper_parameters']['epoch']
                self.is_regulation = file_content['hyper_parameters']['is_regulation']
                self.coeff_regulation = file_content['hyper_parameters']['coeff_regulation']
                self.loss_method = file_content['hyper_parameters']['loss_method']
                self.class_num = file_content['hyper_parameters']['class_num']

                # data info
                self.height = file_content['shape']['height']
                self.width = file_content['shape']['width']
                self.channel = file_content['shape']['channel']

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

                self.train = True

        arguments = Variable()

        # create folder (tfrecord)
        # level: root
        tfrecords_path = arguments.tfrecords_path
        folder_make(tfrecords_path)

        # level: task
        tfrecords_path_task = fullfile(tfrecords_path, arguments.task_name)
        folder_make(tfrecords_path_task)
        arguments.tfrecord_path = tfrecords_path_task

        # create folder (models and logs)
        # level: root
        models_path_root, logs_path_root = arguments.models_path, arguments.logs_path
        folder_make(models_path_root), folder_make(logs_path_root)

        # level: network
        models_path_network, logs_path_network = fullfile(models_path_root, arguments.network), fullfile(logs_path_root, arguments.network)
        folder_make(models_path_network), folder_make(logs_path_network)

        # level: task
        models_path_task, logs_path_task = fullfile(models_path_network, arguments.task_name), fullfile(logs_path_network, arguments.task_name)
        folder_make(models_path_task), folder_make(logs_path_task)

        # process for checkpoint
        sub_directory = get_sub_directory(models_path_task)
        if arguments.checkpoint is True and len(sub_directory) > 0:
            models_path_current = sub_directory[-1]
            logs_path_current = models_path_current.replace("models", "logs")
        else:
            # level: training start time
            current_time_stamp = str(get_unix_stamp(get_time()))
            models_path_current, logs_path_current = fullfile(models_path_task, current_time_stamp), fullfile(logs_path_task, current_time_stamp)
            folder_make(models_path_current), folder_make(logs_path_current)

        arguments.model_path = models_path_current
        arguments.log_path = logs_path_current

        return arguments


def config_test_file_read(config_file_path):
    with open(config_file_path, encoding='utf-8') as json_file:
        file_content = json.load(json_file)

        class Variable:
            def __init__(self):
                # files_path
                self.cover_test_path = file_content['files_path']['cover_test_path']
                self.stego_test_path = file_content['files_path']['stego_test_path']
                self.models_path = file_content['files_path']['models_path']

                # mode_config
                self.gpu_selection = file_content['mode_config']['gpu_selection']
                self.gpu = file_content['mode_config']['gpu']
                self.mode = file_content['mode_config']['mode']
                self.carrier = file_content['mode_config']['carrier']
                self.network = file_content['mode_config']['network']

                # hyper_parameters
                self.batch_size = file_content['hyper_parameters']['batch_size']
                self.class_num = file_content['hyper_parameters']['class_num']

                # shape of input data
                self.height = file_content['shape']['height']
                self.width = file_content['shape']['width']
                self.channel = file_content['shape']['channel']

                # index
                self.start_index_test = file_content["index"]["start_index_test"]
                self.end_index_test = file_content["index"]["end_index_test"]

        argument = Variable()

        return argument


def config_steganalysis_file_read(config_file_path):
    with open(config_file_path, encoding='utf-8') as json_file:
        file_content = json.load(json_file)

        class Variable:
            def __init__(self):
                # files_path
                self.steganalysis_file_path = file_content['files_path']['steganalysis_file_path']
                self.steganalysis_files_path = file_content['files_path']['steganalysis_files_path']
                self.models_path = file_content['files_path']['models_path']

                # mode_config
                self.gpu_selection = file_content['mode_config']['gpu_selection']
                self.gpu = file_content['mode_config']['gpu']
                self.mode = file_content['mode_config']['mode']
                self.submode = file_content['mode_config']['submode']
                self.carrier = file_content['mode_config']['carrier']
                self.network = file_content['mode_config']['network']

                # hyper_parameters
                self.class_num = file_content['hyper_parameters']['class_num']

                # data info
                self.height = file_content['shape']['height']
                self.width = file_content['shape']['width']
                self.channel = file_content['shape']['channel']

        argument = Variable()

        return argument
