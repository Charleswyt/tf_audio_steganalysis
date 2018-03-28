#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
@author: Wang Yuntao
"""

from manager import *
import tensorflow as tf
from config import command_parse
from run import train_audio, train_image, test_batch, test
import os


needed_packages = ["tensorflow-gpu", "numpy", "matplotlib"]
packages_download(needed_packages)                              # if there is no needed packages, download

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

if arguments.mode == "train":                               # train mode
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if arguments.carrier == "audio":
        train_audio(arguments)
    elif arguments.carrier == "image":
        train_image(arguments)
    else:
        print("No appropriate network for this type of carrier now, please try again.")

elif arguments.mode == "test":                              # test mode
    if arguments.test == "one":
        test(arguments)
    if arguments.test == "batch":
        test_batch(arguments)
else:
    print("Mode Error")
