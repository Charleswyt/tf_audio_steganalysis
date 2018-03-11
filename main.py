#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
@author: Wang Yuntao
"""

from manager import *
from config import command_parse
from run import train, test_batch, test


needed_packages = ["tensorflow-gpu", "numpy", "matplotlib"]
packages_download(needed_packages)                              # if there is no needed packages, download

gm = GPUManager()                                               # GPU distribution automatically
with gm.auto_choice():
    arguments = command_parse()
    print(arguments)

    if arguments.mode == "train":                               # train mode
        train(arguments)
    elif arguments.mode == "test":                              # test mode
        if arguments.test == "one":
            test(arguments)
        if arguments.test == "batch":
            test_batch(arguments)
    else:
        print("Mode Error")
