#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
@author: Wang Yuntao
"""

import os
from run import *
from manager import *
import tensorflow as tf
from config import command_parse


try:
    import win_unicode_console
    win_unicode_console.enable()
except ImportError:
    print("win_unicode_console import unsuccessfully.")

arguments = command_parse()
if arguments.gpu_selection == "auto":
    gm = GPUManager()
    os.environ["CUDA_VISIBLE_DEVICES"] = gm.auto_choice()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
run_mode(arguments)
