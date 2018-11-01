#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.07.03
Finished on  2018.07.03
@author: Yuntao Wang
"""

"""
This script is used for distributed calculation

Mode:
    1. single machine and single GPU (SMSG)
    2. single machine and multiple GPU (SMMG)
    3. multiple machine and multiple GPU (MMMG)

Illustration:
        The main consideration is the distribution of parameters servers and computing servers, and another consideration is the mode of update, 
    which can be divided into synchronous update and asynchronous update
    
    parameter server: the server which is used to collect all parameters from each computing servers and get an average value
    computing server: the server which is used to calculate 

    SMSG: Just adjusting batch size is OK, and parameter server and computing server is unified in the current using machine
    SMMG: 
"""




