#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.05.13
Finished on 2018.05.30
@author: Wang Yuntao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest


def best_k_selection(feature, label, k=50):
    """
    select the best k feature
    :param feature: feature data
    :param label: feature label
    :param k: the best k feature
    :return:
        the selected feature
    """
    feature_new = SelectKBest(chi2, k=k).fit_transform(feature, label)

    return feature_new


def pca_selection(data):
    pca_sklearn = decomposition.PCA()
    pca_sklearn.fit(data)
    main_var = pca_sklearn.explained_variance_
    print(sum(main_var) * 0.9)
    n = 15
    plt.plot(main_var[:n])
    plt.show()


def get_confusion_matrix(data, label):
    pass
