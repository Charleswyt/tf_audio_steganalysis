#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018.03.16
Finished on 2018.03.16
@author: Wang Yuntao
"""

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from scipy import interp
import matplotlib.pyplot as plt


def plot_confusion_matrix(label, predict, matrix_title):
    """confusion matrix computation and display"""
    plt.figure(figsize=(9, 9), dpi=100)

    # use sklearn confusion matrix
    cm_array = confusion_matrix(label, predict)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(matrix_title, fontsize=16)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

    true_labels = np.unique(label)
    pred_labels = np.unique(predict)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))

    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    cv = StratifiedKFold(y, n_folds=6)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        #    print set(y[train])                     #set([0,1]) 即label有两个类别
        #    print len(X[train]),len(X[test])        #训练集有84个，测试集有16个
        #    print "++",probas_                      #predict_proba()函数输出的是测试集在lael各类别上的置信度，
        #    #在哪个类别上的置信度高，则分为哪类
        # Compute ROC curve and area the curve
        # 通过roc_curve()函数，求出fpr和tpr，以及阈值
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    # 画平均ROC曲线
    # print mean_fpr,len(mean_fpr)
    # print mean_tpr
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
