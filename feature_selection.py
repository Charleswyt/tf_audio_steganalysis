#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.05.13
Finished on
@author: Wang Yuntao
"""

import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))
