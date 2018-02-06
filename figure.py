#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2018.01.16
Finished on 2018.01.16
@author: Wang Yuntao
"""

import numpy as np
import matplotlib.pyplot as plt

def loadCSVfile2(csv_file_name):
    tmp = np.loadtxt(csv_file_name, dtype=np.str, delimiter=",")
    data = tmp[1:,1:].astype(np.float)                      # 加载数据部分
    label = tmp[1:,0].astype(np.float)                      # 加载类别标签部分
    return data, label                                      # 返回array类型的数据
