#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matlab.engine

"""
matlab API for python: 
    http://ww2.mathworks.cn/help/matlab/matlab-engine-for-python.html
"""

matlab_engine = matlab.engine.start_matlab()           # 初始化matlab引擎

out = matlab_engine.read_jpeg_image("I:/jpeg/img_1009.jpg")
print(type(out))

