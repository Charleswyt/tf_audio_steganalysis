#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
Modified on 2018.08.24

@author: Wang Yuntao
"""

import os
import sys
from run import *
from config import *
from manager import *


def main():
    # command parsing
    params_num = len(sys.argv)

    # json config mode
    if params_num == 1:
        config_file_path = "./config_file/config_train.json"
        arguments = config_train_file_read(config_file_path)
    elif params_num == 2:
        config_file_path = "./config_file/config_" + sys.argv[1] + ".json"
        command = "config_" + sys.argv[1] + "_file_read(" + config_file_path + ")"
        arguments = eval(command)

    # command line mode
    else:
        arguments = command_parse()

    # mode of gpu selection: auto, manu and others
    if is_gpu_available():
        if arguments.gpu_selection == "auto":
            gm = GPUManager()
            gpu_index = gm.auto_choice()
            if not gpu_index == -1:
                allocated_gpu = gpu_index
            else:
                allocated_gpu = -1
        elif arguments.gpu_selection == "manu":
            allocated_gpu = arguments.gpu
        else:
            allocated_gpu = -1
    else:
        allocated_gpu = -1

    os.environ["CUDA_VISIBLE_DEVICES"] = allocated_gpu

    run_mode(arguments)


if __name__ == "__main__":
    main()
