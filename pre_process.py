#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.01.03
Finished on 2018.01.03
@author: Wang Yuntao
"""

import math
import numpy as np

"""
The pre-processing of the QMDCT matrix:
    truncate(matrix, threshold)                                                                 truncation of matrix
    preprocess(content, is_abs=False, is_diff=False, is_abs_diff=False,
        is_diff_abs=False, order=2, direction=0, is_trunc=False, threshold=15)                  pre-processing of matrix
"""


def truncate(matrix, threshold=None, threshold_left=None, threshold_right=None):
    """
    truncation

    :param matrix: the input matrix (numpy.ndarray)
    :param threshold: threshold
    :param threshold_left: threshold (for minimum)
    :param threshold_right: threshold (for maximum)
    :return:
    """
    if threshold_left is not None and threshold_right is not None:
        matrix[matrix > threshold_left] = threshold_left
        matrix[matrix > threshold_right] = threshold_right
    else:
        matrix[matrix > threshold] = threshold
        matrix[matrix < -threshold] = -threshold

    return matrix


def preprocess(content, is_abs=False, is_diff=False, is_abs_diff=False, is_diff_abs=False, order=2, direction=0, is_trunc=False, threshold=15):
    """
    pre-processing of input data (image, audio, qmdct)

    :param content: input data (image, audio, qmdct)
    :param is_abs: whether make absolute or not
    :param is_diff: whether make difference or not
    :param is_abs_diff: whether make absolute and difference or not
    :param is_diff_abs: whether make difference and absolute or not
    :param order: order of difference
    :param direction: direction of difference
    :param is_trunc: whether make truncation or not
    :param threshold: threshold of truncation
    :return:
    """
    if is_abs is True:
        content = abs(content)
    if is_trunc is True:
        content = truncate(content, threshold=threshold)
    if is_diff is True:
        content = np.diff(content, n=order, axis=direction)
    if is_diff_abs is True:
        content = np.diff(content, n=order, axis=direction)
        content = abs(content)
    if is_abs_diff is True:
        content = abs(content)
        content = np.diff(content, n=order, axis=direction)

    return content
