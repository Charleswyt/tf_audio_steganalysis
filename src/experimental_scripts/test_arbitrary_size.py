#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019.03.12
Finished on 2019.03.12
Modified on

@author: Yuntao Wang
"""

import os
import time
import datetime
from utils import *
import tensorflow as tf
from itertools import product
from networks.networks import networks
from networks.tested_steganalysis import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# hyper parameters
batch_size = 16
carrier = "qmdct"
classes_num = 2
start_index_test, end_index_test = 0, 1038
is_shuffle = True

# path
cover_test_files_path = "/home1/wyt/data/size_mismatch/cover_128"
stego_test_files_path = "/home1/wyt/data/size_mismatch/EECS_128_W_2_H_7_ER_10"
models_path = "/home1/wyt/code/tf_audio_steganalysis/models/steganalysis/google_net3/EECS_B_128_W_2_H_7_ER_10/1552005868"

heights = [200, 250, 300, 350, 400]
widths = [400, 430, 450, 480, 500, 530]
channel = 1
network = "google_net3"

for height, width in product(heights, widths):
    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, channel), name="data")
    labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, ), name="label")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    command = network + "(data, classes_num, is_bn)"
    logits = eval(command)

    accuracy, false_positive_rate, false_negative_rate = evaluation(logits=logits, labels=labels)

    # information output
    print("cover files path: %s" % cover_test_files_path)
    print("stego files path: %s" % stego_test_files_path)
    print("class number: %d" % classes_num)
    print("start load network...")

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    start_time = time.time()

    model = tf.train.Saver()
    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        model_file_path = get_model_file_path(models_path)

        if model_file_path is None:
            print("No model is loaded successfully.")
        else:
            model.restore(sess, model_file_path)
            print("The model is loaded successfully, model file: %s" % model_file_path)
            # read files list (train)
            cover_test_data_list, cover_test_label_list, \
                stego_test_data_list, stego_test_label_list = read_data(cover_test_files_path, stego_test_files_path, start_index_test, end_index_test, is_shuffle=is_shuffle)

            if len(cover_test_data_list) < batch_size:
                batch_size = len(cover_test_data_list)

            test_iterations, test_accuracy, test_fpr, test_fnr = 0, 0, 0, 0
            for x_test_batch, y_test_batch in \
                    minibatches(cover_test_data_list, cover_test_label_list, stego_test_data_list, stego_test_label_list, batch_size):
                # data read and process
                x_test_data = get_data_batch(x_test_batch, height=height, width=width, channel=channel, carrier=carrier)

                # get the accuracy and loss
                acc, fpr, fnr = sess.run([accuracy, false_positive_rate, false_negative_rate],
                                         feed_dict={data: x_test_data, labels: y_test_batch, is_bn: True})
                test_accuracy += acc
                test_fpr += fpr
                test_fnr += fnr
                test_iterations += 1

                print("Batch-%003d, accuracy: %f, fpr: %f, fnr: %f" % (test_iterations, acc, fpr, fnr))

            test_accuracy_average = test_accuracy / test_iterations
            test_fpr_average = test_fpr / test_iterations
            test_fnr_average = test_fnr / test_iterations
            print("Test accuracy: %.2f%%, FPR: %.2f%%, FNR: %.2f%%" % (100. * test_accuracy_average, 100. * test_fpr_average, 100. * test_fnr_average))

            with open("/home1/wyt/code/tf_audio_steganalysis/results_arbitrary_size.txt", "a") as file:
                file.write("height: %d, width: %d -- Test accuracy: %.2f%%\n" % (height, width, 100. * test_accuracy_average))

    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print("Run Time: %s" % et)
