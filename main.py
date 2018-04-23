#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
@author: Wang Yuntao
"""


import os
import tensorflow as tf
from manager import check_gpus
from config import command_parse
from run import train, steganalysis_one, steganalysis_batch

try:
    import win_unicode_console
    win_unicode_console.enable()
except ImportError:
    print("win_unicode_console import unsuccessfully.")

arguments = command_parse()

if check_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
else:
    print("No GPU.")

if arguments.mode == "train":                               # train mode
    train(arguments)
elif arguments.mode == "test":                              # test mode
    if arguments.submode == "one":
        steganalysis_one(arguments)
    if arguments.submode == "batch":
        steganalysis_batch(arguments)
else:
    print("Mode Error")
