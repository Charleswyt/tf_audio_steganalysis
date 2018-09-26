#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
from utils import *
from networks.networks import networks
from file_preprocess import get_file_name
from networks.audio_steganalysis import *
from networks.image_steganalysis import *
from networks.tested_steganalysis import *
from networks.image_classification import *

"""
Created on 2017.11.27
Finished on 2017.11.30
Modified on 2018.09.18

@author: Wang Yuntao
"""

"""
    function: 
        train(args)                                             # train
        test(args)                                              # test
        steganalysis_one(args)                                  # steganalysis for one sample
        steganalysis_batch(args)                                # steganalysis for multiple samples
"""


def run_mode(args):
    if args.mode == "train":                                    # train mode
        train(args)
    elif args.mode == "test":                                   # test mode
        test(args)
    elif args.mode == "steganalysis":                           # steganalysis mode
        if args.submode == "one":
            if get_path_type(args.steganalysis_file_path) == "file":
                steganalysis_one(args)
            else:
                print("The submode miss-matches the file path, please try again.")

        if args.submode == "batch":
            if get_path_type(args.steganalysis_files_path) == "folder":
                steganalysis_batch(args)
            else:
                print("The submode miss-matches the files path, please try again.")
    else:
        print("Mode Error")


def train(args):
    # hyper parameters
    batch_size = args.batch_size                                                            # batch size
    height, width, channel = args.height, args.width, args.channel                          # height and width of input matrix
    init_learning_rate = args.learning_rate                                                 # initialized learning rate
    n_epoch = args.epoch                                                                    # epoch
    decay_method = args.decay_method                                                        # decay method
    decay_steps, decay_rate = args.decay_step, args.decay_rate                              # decay steps | decay rate
    loss_method = args.loss_method                                                          # the calculation method of loss function
    is_regulation = args.is_regulation                                                      # regulation or not
    coeff_regulation = args.coeff_regulation                                                # the gain of regulation
    classes_num = args.class_num                                                            # classes number
    carrier = args.carrier                                                                  # carrier (qmdct | audio | image)
    checkpoint = args.checkpoint                                                            # checkpoint

    max_to_keep = args.max_to_keep                                                          # maximum number of recent checkpoints to keep
    keep_checkpoint_every_n_hours = args.keep_checkpoint_every_n_hours                      # how often to keep checkpoints
    start_index_train, end_index_train = args.start_index_train, args.end_index_train       # the scale of the dataset (train)
    start_index_valid, end_index_valid = args.start_index_valid, args.end_index_valid       # the scale of the dataset (valid)
    step_train, step_valid = 0, 0                                                           # train step and valid step
    max_accuracy, max_accuracy_epoch = 0, 0                                                 # max valid accuracy and corresponding epoch

    with tf.device("/cpu:0"):
        global_step = tf.Variable(initial_value=0,
                                  trainable=False,
                                  name="global_step",
                                  dtype=tf.int32)                                           # global step (Variable 变量不能直接分配GPU资源)

    # learning rate decay
    learning_rate = learning_rate_decay(init_learning_rate=init_learning_rate,
                                        decay_method=decay_method,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=decay_rate)                              # learning rate

    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, channel), name="data")
    labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, ), name="label")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # initialize the network
    if args.network not in networks:
        print("Network miss-match, please try again")
        return False

    command = args.network + "(data, classes_num, is_bn)"
    logits = eval(command)

    # evaluation
    loss = loss_layer(logits=logits, labels=labels, is_regulation=is_regulation, coeff=coeff_regulation, method=loss_method)
    train_optimizer = optimizer(losses=loss, learning_rate=learning_rate, global_step=global_step)
    accuracy = accuracy_layer(logits=logits, labels=labels)

    # file path
    cover_train_path = args.cover_train_path
    stego_train_path = args.stego_train_path
    cover_valid_path = args.cover_valid_path
    stego_valid_path = args.stego_valid_path

    model_path = args.model_path
    log_path = args.log_path

    # information output
    print("train files path(cover): %s" % cover_train_path)
    print("valid files path(cover): %s" % cover_valid_path)
    print("train files path(stego): %s" % stego_train_path)
    print("valid files path(stego): %s" % stego_valid_path)
    print("model files path: %s" % model_path)
    print("log files path: %s" % log_path)
    print("batch size: %d, total epoch: %d, class number: %d, initial learning rate: %f, decay method: %s, decay rate: %f, decay steps: %d"
          % (batch_size, n_epoch, classes_num, init_learning_rate, decay_method, decay_rate, decay_steps))
    print("start load network...")

    with tf.device("/cpu:0"):
        tf.summary.scalar("loss_train", loss)
        tf.summary.scalar("accuracy_train", accuracy)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=max_to_keep,
                               keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    # initialize
    with tf.Session() as sess:
        train_writer_train = tf.summary.FileWriter(log_path + "/train", tf.get_default_graph())
        train_writer_valid = tf.summary.FileWriter(log_path + "/validation", tf.get_default_graph())
        init = tf.global_variables_initializer()
        sess.run(init)

        # restore the model and keep training from the current breakpoint
        if checkpoint is True:
            model_file_path = get_model_file_path(args.model_path)
            if model_file_path is not None:
                saver.restore(sess, model_file_path)

        print("Start training...")
        print("Input data: (%d, %d, %d)" % (height, width, channel))

        start_time_all = time.time()
        for epoch in range(n_epoch):
            # read files list (train)
            cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list = read_data(cover_train_path,
                                                                                                                     stego_train_path,
                                                                                                                     start_index_train,
                                                                                                                     end_index_train)

            # read files list (validation)
            cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list = read_data(cover_valid_path,
                                                                                                                     stego_valid_path,
                                                                                                                     start_index_valid,
                                                                                                                     end_index_valid)
            # update the learning rate
            lr = sess.run(learning_rate)

            # train
            train_iterations, train_loss, train_accuracy = 0, 0, 0
            for x_train_batch, y_train_batch in \
                    minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size):
                # data read and process
                x_train_data = get_data_batch(x_train_batch, height=height, width=width, channel=channel, carrier=carrier)

                # get the accuracy and loss
                _, err, ac, summary_str_train = sess.run([train_optimizer, loss, accuracy, summary_op],
                                                         feed_dict={data: x_train_data, labels: y_train_batch, is_bn: True})

                train_loss += err
                train_accuracy += ac
                step_train += 1
                train_iterations += 1
                train_writer_train.add_summary(summary_str_train, global_step=step_train)

                et = time.time() - start_time_all
                et = str(datetime.timedelta(seconds=et))[:-7]
                print("[network: %s, task: %s] elapsed: %s, epoch: %003d, train iterations: %003d: train loss: %f, train accuracy: %f"
                      % (args.network, args.task_name, et, epoch + 1, train_iterations, err, ac))

            print("=====================================================================================================================================")

            # validation
            valid_iterations, valid_loss, valid_accuracy = 0, 0, 0
            for x_valid_batch, y_valid_batch in \
                    minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size):
                # data read and process
                x_valid_data = get_data_batch(x_valid_batch, height=height, width=width, channel=channel, carrier=carrier)

                # get the accuracy and loss
                err, ac, summary_str_valid = sess.run([loss, accuracy, summary_op],
                                                      feed_dict={data: x_valid_data, labels: y_valid_batch, is_bn: True})
                valid_loss += err
                valid_accuracy += ac
                valid_iterations += 1
                step_valid += 1
                train_writer_valid.add_summary(summary_str_valid, global_step=step_valid)

                et = time.time() - start_time_all
                et = str(datetime.timedelta(seconds=et))[:-7]
                print("[network: %s, task: %s] elapsed: %s, epoch: %003d, valid iterations: %003d, valid loss: %-8f, valid accuracy: %f"
                      % (args.network, args.task_name, et, epoch + 1, valid_iterations, err, ac))

            # calculate the average in a batch
            train_loss_average = train_loss / train_iterations
            valid_loss_average = valid_loss / valid_iterations
            train_accuracy_average = train_accuracy / train_iterations
            valid_accuracy_average = valid_accuracy / valid_iterations

            # model save
            if valid_accuracy_average > max_accuracy:
                max_accuracy = valid_accuracy_average
                max_accuracy_epoch = epoch + 1
                saver.save(sess, model_path, global_step=global_step)
                print("The model is saved successfully.")

            print("[network: %s, task: %s] epoch: %003d, learning rate: %f, train loss: %f, train accuracy: %f, valid loss: %f, valid accuracy: %f, "
                  "max valid accuracy: %f, max valid acc epoch: %d" % (args.network, args.task_name, epoch + 1, lr, train_loss_average, train_accuracy_average, valid_loss_average,
                                                                       valid_accuracy_average, max_accuracy, max_accuracy_epoch))

        train_writer_train.close()
        train_writer_valid.close()


def test(args):
    # hyper parameters
    batch_size = args.batch_size                                        # batch size
    height, width, channel = args.height, args.width, args.channel      # height and width of input matrix
    carrier = args.carrier                                              # carrier (qmdct | audio | image)
    classes_num = args.class_num                                        # classes number
    start_index_test, end_index_test = args.start_index_test, args.end_index_test

    # path
    cover_test_files_path = args.cover_test_path
    stego_test_files_path = args.stego_test_path

    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, channel), name="data")
    labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, ), name="label")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # initialize the network
    if args.network not in networks:
        print("Network miss-match, please try again")
        return False

    command = args.network + "(data, classes_num, is_bn)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    accuracy = accuracy_layer(logits=logits, labels=labels)

    # information output
    print("train files path(cover): %s" % cover_test_files_path)
    print("valid files path(stego): %s" % stego_test_files_path)
    print("class number: %d" % classes_num)
    print("start load network...")

    model = tf.train.Saver()
    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        model_file_path = get_model_file_path(args.models_path)

        if model_file_path is None:
            print("No model is loaded successfully.")
        else:
            model.restore(sess, model_file_path)
            print("The model is loaded successfully, model file: %s" % model_file_path)
            # read files list (train)
            cover_test_data_list, cover_test_label_list, stego_test_data_list, stego_test_label_list = read_data(cover_test_files_path,
                                                                                                                 stego_test_files_path,
                                                                                                                 start_index_test,
                                                                                                                 end_index_test)
            if len(cover_test_data_list) < batch_size:
                batch_size = len(cover_test_data_list)

            test_iterations, test_accuracy = 0, 0
            for x_test_batch, y_test_batch in \
                    minibatches(cover_test_data_list, cover_test_label_list, stego_test_data_list, stego_test_label_list, batch_size):
                # data read and process
                x_test_data = get_data_batch(x_test_batch, height=height, width=width, channel=channel, carrier=carrier)

                # get the accuracy and loss
                acc = sess.run(accuracy, feed_dict={data: x_test_data, labels: y_test_batch, is_bn: True})
                test_accuracy += acc
                test_iterations += 1

                print("Batch-%d, accuracy: %f" % (test_iterations, acc))

            test_accuracy_average = test_accuracy / test_iterations
            print("Test accuracy: %.2f%%" % (100. * test_accuracy_average))


def steganalysis_one(args):
    # hyper parameters
    height, width, channel = args.height, args.width, args.channel      # height and width of input matrix
    carrier = args.carrier                                              # carrier (qmdct | audio | image)
    classes_num = args.classes_num                                      # classes for classification

    # path
    steganalysis_file_path = args.steganalysis_file_path

    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(1, height, width, channel), name="data")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # initialize the network
    if args.network not in networks:
        print("Network miss-match, please try again")
        return False

    # network
    command = args.network + "(data," + classes_num + ",is_bn)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    model = tf.train.Saver()
    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        model_file_path = get_model_file_path(args.model_path)
        print(model_file_path)
        if model_file_path is None:
            print("No model is loaded successfully.")
        else:
            model.restore(sess, model_file_path)
            print("The model is loaded successfully, model file: %s" % model_file_path)

            steganalysis_data = get_data_batch([steganalysis_file_path], width=width, height=height, channel=channel, carrier=carrier)

            if steganalysis_data is None:
                print("No model can be used for this carrier. (need image or audio)")
            else:
                file_name = get_file_name(steganalysis_file_path)
                ret = sess.run(logits, feed_dict={data: steganalysis_data, is_bn: False})
                result = np.argmax(ret, 1)
                prob = 100 * ret[0][result]

                if result[0] == 0:
                    print("file name: %s, result: cover, prob of prediction: %.2f%%" % (file_name, prob))

                if result[0] == 1:
                    print("file name: %s, result: stego, prob of prediction: %.2f%%" % (file_name, prob))


def steganalysis_batch(args):
    # hyper parameters
    height, width, channel = args.height, args.width, args.channel      # height and width of input matrix
    carrier = args.carrier                                              # carrier (qmdct | audio | image)
    classes_num = args.class_num                                        # classes number

    # path
    steganalysis_files_path = args.steganalysis_files_path

    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(1, height, width, channel), name="data")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # initialize the network
    if args.network not in networks:
        print("Network miss-match, please try again")
        return False

    # network
    command = args.network + "(data," + classes_num + ",is_bn)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    model = tf.train.Saver()
    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        model_file_path = get_model_file_path(args.models_path)

        if model_file_path is None:
            print("No model is loaded successfully.")
        else:
            model.restore(sess, model_file_path)
            print("The model is loaded successfully, model file: %s" % model_file_path)

            file_list = get_files_list(steganalysis_files_path)
            print("files path: %s" % steganalysis_files_path)

            for file_path in file_list:
                steganalysis_data = get_data_batch([file_path], width=width, height=height, channel=channel, carrier=carrier)

                file_name = get_file_name(file_path)
                ret = sess.run(logits, feed_dict={data: steganalysis_data, is_bn: False})
                result = np.argmax(ret, 1)
                prob = 100 * ret[0][result]

                if result[0] == 0:
                    print("file name: %s, result: cover, prob of prediction: %.2f%%" % (file_name, prob))

                if result[0] == 1:
                    print("file name: %s, result: stego, prob of prediction: %.2f%%" % (file_name, prob))
