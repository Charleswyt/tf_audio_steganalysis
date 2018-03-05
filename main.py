#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
@author: Wang Yuntao
"""

from config import command_parse
from run import train, test_batch, test

arguments = command_parse()
print(arguments)

if arguments.mode == "train":
    train(arguments)
elif arguments.mode == "test":
    if arguments.test == "one":
        test(arguments)
    if arguments.test == "batch":
        test_batch(arguments)
else:
    print("Mode Error")

# model_dir = "models/steganalysis/" + stego_dir_name
# model_name = "steganalysis_model.npz"
# logs_train_dir = os.path.join("logs/myself/train/", stego_dir_name)
# logs_val_dir = os.path.join("logs/myself/train/", stego_dir_name)

