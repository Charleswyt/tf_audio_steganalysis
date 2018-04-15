#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
@author: Wang Yuntao
"""


import os
import tensorflow as tf
from config import command_parse
from run import train_audio, train_image, steganalysis_one, steganalysis_batch

try:
    import win_unicode_console
    win_unicode_console.enable()
except ImportError:
    print("win_unicode_console import unsuccessfully.")


needed_packages = ["tensorflow-gpu", "numpy", "matplotlib"]
# packages_download(needed_packages)                              # if there is no needed packages, download

# gm = GPUManager()                                               # GPU distribution automatically
# with gm.auto_choice():
#     arguments = command_parse()
#     print(arguments)
#
#     if arguments.mode == "train":                               # train mode
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         train(arguments)
#     elif arguments.mode == "test":                              # test mode
#         if arguments.test == "one":
#             test(arguments)
#         if arguments.test == "batch":
#             test_batch(arguments)
#     else:
#         print("Mode Error")

arguments = command_parse()
print(arguments)

try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
except BaseException:
    print("No GPU.")

if arguments.mode == "train":                               # train mode
    if arguments.carrier == "audio":
        train_audio(arguments)
    elif arguments.carrier == "image":
        train_image(arguments)
    else:
        print("No appropriate network for this type of carrier now, please try again.")

elif arguments.mode == "test":                              # test mode
    if arguments.submode == "one":
        steganalysis_one(arguments)
    if arguments.submode == "batch":
        steganalysis_batch(arguments)
else:
    print("Mode Error")
