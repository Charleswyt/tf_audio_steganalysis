#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from utils import *
from networks.tested import *
from networks.audio_steganalysis import *
from networks.image_steganalysis import *
from file_preprocess import get_file_name

"""
Created on 2017.11.27
Finished on 2017.11.30
Modified on 2018.08.29

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
            steganalysis_one(args)
        if args.submode == "batch":
            steganalysis_batch(args)
    else:
        print("Mode Error")


def train(args):
    # hyper parameters
    batch_size_train, batch_size_valid = args.batch_size_train, args.batch_size_valid       # batch size (train and valid)
    init_learning_rate = args.learning_rate                                                 # initialized learning rate
    n_epoch = args.epoch                                                                    # epoch
    decay_method = args.decay_method                                                        # decay method
    decay_steps, decay_rate = args.decay_step, args.decay_rate                              # decay steps | decay rate
    loss_method = args.loss_method                                                          # the calculation method of loss function
    is_regulation = args.is_regulation                                                      # regulation or not
    coeff_regulation = args.coeff_regulation                                                # the gain of regulation
    classes_num = args.class_num                                                            # classes number
    carrier = args.carrier                                                                  # carrier (audio | image)

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

    # pre processing
    is_abs, is_trunc, threshold, is_diff, order, direction, is_diff_abs = \
        args.is_abs, args.is_trunc, args.threshold, args.is_diff, args.order, args.direction, args.is_diff_abs

    # learning rate decay
    learning_rate = learning_rate_decay(init_learning_rate=init_learning_rate,
                                        decay_method=decay_method,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=decay_rate)                             # learning rate

    # the height and width of input data
    height, width, channel = args.height, args.width, 1                                    # the height, width and channel of the QMDCT matrix
    if is_diff is True and direction == 0:
        height_new, width_new = height - order, width
    elif is_diff is True and direction == 1:
        height_new, width_new = height, width - order
    else:
        height_new, width_new = height, width

    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(None, height, width, channel), name="QMDCTs")
    labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name="labels")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # initialize the network
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

    models_file_path = args.models_path
    logs_path = args.logs_path

    log_path = fullfile(logs_path, args.network)
    model_file_path = fullfile(models_file_path, args.network)

    # information output
    print("train files path(cover): %s" % cover_train_path)
    print("valid files path(cover): %s" % cover_valid_path)
    print("train files path(stego): %s" % stego_train_path)
    print("valid files path(stego): %s" % stego_valid_path)
    print("model files path: %s" % model_file_path)
    print("log files path: %s" % log_path)
    print("batch size(train): %d, batch size(validation): %d, total epoch: %d, class number: %d, initial learning rate: %f, "
          "decay method: %s, decay rate: %f, decay steps: %d" % (batch_size_train, batch_size_valid, n_epoch, classes_num,
                                                                 init_learning_rate, decay_method, decay_rate, decay_steps))
    print("start load network...")

    with tf.device("/cpu:0"):
        tf.summary.scalar("loss_train", loss)
        tf.summary.scalar("accuracy_train", accuracy)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=max_to_keep,
                               keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    # initialize
    sess = tf.InteractiveSession()
    train_writer_train = tf.summary.FileWriter(log_path + "/train", tf.get_default_graph())
    train_writer_valid = tf.summary.FileWriter(log_path + "/validion", tf.get_default_graph())
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Start training...")
    for epoch in range(n_epoch):
        start_time = time.time()

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
                minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size_train):
            # data read and process
            x_train_data = get_data_batch(x_train_batch, height, width, carrier=carrier)

            # data preprocessing


            # get the accuracy and loss
            _, err, ac, summary_str_train = sess.run([train_optimizer, loss, accuracy, summary_op],
                                                     feed_dict={data: x_train_data, labels: y_train_batch, is_bn: True})

            train_loss += err
            train_accuracy += ac
            step_train += 1
            train_iterations += 1
            train_writer_train.add_summary(summary_str_train, global_step=step_train)

            print("epoch: %003d, train iterations: %003d: train loss: %f, train accuracy: %f" % (epoch + 1, train_iterations, err, ac))

        print("==================================================================================")

        # validation
        valid_iterations, valid_loss, valid_accuracy = 0, 0, 0
        for x_valid_batch, y_valid_batch in \
                minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size_valid):
            # data read and process
            x_valid_data = get_data_batch(x_valid_batch, height, width, carrier=carrier)

            # get the accuracy and loss
            err, ac, summary_str_valid = sess.run([loss, accuracy, summary_op],
                                                  feed_dict={data: x_valid_data, labels: y_valid_batch, is_bn: True})
            valid_loss += err
            valid_accuracy += ac
            valid_iterations += 1
            step_valid += 1
            train_writer_valid.add_summary(summary_str_valid, global_step=step_valid)

            print("epoch: %003d, valid iterations: %003d, valid loss: %f, valid accuracy: %f" % (epoch + 1, valid_iterations, err, ac))

        # calculate the average in a batch
        train_loss_average = train_loss / train_iterations
        valid_loss_average = valid_loss / valid_iterations
        train_accuracy_average = train_accuracy / train_iterations
        valid_accuracy_average = valid_accuracy / valid_iterations

        # model save
        if valid_accuracy_average > max_accuracy:
            max_accuracy = valid_accuracy_average
            max_accuracy_epoch = epoch + 1
            saver.save(sess, model_file_path, global_step=global_step)
            print("The model is saved successfully.")

        print("epoch: %003d, learning rate: %f, train loss: %f, train accuracy: %f, valid loss: %f, valid accuracy: %f, "
              "max valid accuracy: %f, max valid acc epoch: %d" % (epoch + 1, lr, train_loss_average, train_accuracy_average, valid_loss_average,
                                                                   valid_accuracy_average, max_accuracy, max_accuracy_epoch))

        end_time = time.time()
        print("Runtime: %.2fs" % (end_time - start_time))

    train_writer_train.close()
    train_writer_valid.close()
    sess.close()


def test(args):
    carrier = args.carrier
    batch_size_test = args.batch_size_test

    # pre processing
    is_abs, is_trunc, threshold, is_diff, order, direction, is_diff_abs = \
        args.is_abs, args.is_trunc, args.threshold, args.is_diff, args.order, args.direction, args.is_diff_abs

    # the height and width of input data
    height, width, channel = args.height, args.width, 1  # the height, width and channel of the QMDCT matrix
    if is_diff is True and direction == 0:
        height_new, width_new = height - order, width
    elif is_diff is True and direction == 1:
        height_new, width_new = height, width - order
    else:
        height_new, width_new = height, width

    # path
    cover_test_files_path = args.cover_test_path
    stego_test_files_path = args.stego_test_path

    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(None, height_new, width_new, channel), name="QMDCTs")
    labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name="labels")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # initialize the network
    command = args.network + "(data, 2, is_bn)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    accuracy = accuracy_layer(logits=logits, labels=labels)

    model = tf.train.Saver()
    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        if args.model_file_path is None and args.model_files_path is not None:
            model_file_path = tf.train.latest_checkpoint(args.model_files_path)
        elif args.model_file_path is not None:
            model_file_path = args.model_file_path
        else:
            model_file_path = None

        if model_file_path is None:
            print("No model is loaded successfully.")
        else:
            model.restore(sess, model_file_path)
            print("The model is loaded successfully, model file: %s" % model_file_path)
            cover_test_data_list, cover_test_label_list, stego_test_data_list, stego_test_label_list = read_data(cover_test_files_path,
                                                                                                                 stego_test_files_path)
            test_iterations, test_accuracy = 0, 0
            for x_test_batch, y_test_batch in \
                    minibatches(cover_test_data_list, cover_test_label_list, stego_test_data_list, stego_test_label_list, batch_size_test):
                # data read and process
                x_test_data = get_data(x_test_batch, height, width, carrier=carrier, is_diff=is_diff, order=order, direction=direction, is_diff_abs=is_diff_abs,
                                       is_trunc=is_trunc, threshold=threshold)

                # get the accuracy and loss
                acc = sess.run(accuracy, feed_dict={data: x_test_data, labels: y_test_batch, is_bn: True})
                test_accuracy += acc
                test_iterations += 1

                print("Batch-%d, accuracy: %f" % (test_iterations, acc))

            test_accuracy_average = test_accuracy / test_iterations
            print("Test accuracy: %.2f%%" % 100 * test_accuracy_average)


def steganalysis_one(args):
    # the info of carrier
    carrier = args.carrier
    test_file_path = args.test_file_path

    # pre-process
    is_diff, order, direction = args.is_diff, args.order, args.direction

    # the height, width and channel of the QMDCT matrix
    height, width, channel = args.height, args.width, args.channel
    if is_diff is True and direction == 0:
        height_new, width_new = height - order, width
    elif is_diff is True and direction == 1:
        height_new, width_new = height, width - order
    else:
        height_new, width_new = height, width

    # placeholder
    data = tf.placeholder(tf.float32, [None, height_new, width_new, channel], name="QMDCTs")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # network
    command = args.network + "(data, 2, is_bn)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    model = tf.train.Saver()
    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        if args.model_file_path is None and args.model_files_path is not None:
            model_file_path = tf.train.latest_checkpoint(args.model_files_path)
        elif args.model_file_path is not None:
            model_file_path = args.model_file_path
        else:
            model_file_path = None

        if model_file_path is None:
            print("No model is loaded successfully.")
        else:
            model.restore(sess, model_file_path)
            print("The model is loaded successfully, model file: %s" % model_file_path)

            # predict
            if carrier == "audio":
                media = read_text(test_file_path, width=width, is_diff=args.is_diff,
                                  order=args.order, direction=args.direction)
            elif carrier == "image":
                media = io.imread(test_file_path)
            else:
                media = None

            if media is None:
                print("No model can be used for this carrier. (need image or audio)")
            else:
                media_name = get_file_name(test_file_path, sep="\\")
                media = np.reshape(media, [1, height_new, width_new, channel])
                ret = sess.run(logits, feed_dict={data: media, is_bn: False})
                result = np.argmax(ret, 1)
                prob = 100 * ret[0][result]

                media_label = args.label
                if result[0] == 0:
                    print("file name: %s, result: cover, label: %r, prob of prediction: %.2f%%" % (media_name, media_label, prob))

                if result[0] == 1:
                    print("file name: %s, result: stego, label: %r, prob of prediction: %.2f%%" % (media_name, media_label, prob))


def steganalysis_batch(args):
    # the info of carrier
    carrier = args.carrier
    test_files_path = args.test_files_path
    label_file_path = args.label_file_path

    # pre-process
    is_diff, order, direction = args.is_diff, args.order, args.direction

    # the height, width and channel of the QMDCT matrix
    height, width, channel = args.height, args.width, args.channel
    if is_diff is True and direction == 0:
        height_new, width_new = height - order, width
    elif is_diff is True and direction == 1:
        height_new, width_new = height, width - order
    else:
        height_new, width_new = height, width

    # placeholder
    data = tf.placeholder(tf.float32, [None, height_new, width_new, channel], name="QMDCTs")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # network
    command = args.network + "(data, 2, is_bn)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    model = tf.train.Saver()
    with tf.Session() as sess:
        # load model
        sess.run(tf.global_variables_initializer())
        if args.model_file_path is None and args.model_files_path is not None:
            model_file_path = tf.train.latest_checkpoint(args.model_files_path)
        elif args.model_file_path is not None:
            model_file_path = args.model_file_path
        else:
            model_file_path = None

        if model_file_path is None:
            print("No model is loaded successfully.")
        else:
            model.restore(sess, model_file_path)
            print("The model is loaded successfully, model file: %s" % model_file_path)

            test_file_list = get_files_list(test_files_path)

            # get file label
            if label_file_path is not None:
                labels = list()
                with open(label_file_path) as file:
                    for line in file.readlines():
                        labels.append(int(line))
            else:
                labels = -np.ones([image_num, 1])

            results, number = list(), 0
            for test_file in test_file_list:
                # predict
                if carrier == "audio":
                    media = read_text(test_file, width=width, is_diff=args.is_diff,
                                      order=args.order, direction=args.direction)
                elif carrier == "image":
                    media = io.imread(test_file)
                else:
                    media = None

                if media is None:
                    print("No model can be used for this carrier. (need image or audio)")
                else:
                    media_name = get_file_name(test_file, sep="\\")
                    media = np.reshape(media, [1, height_new, width_new, channel])
                    ret = sess.run(logits, feed_dict={data: media, is_bn: False})
                    result = np.argmax(ret, 1)
                    prob = 100 * ret[0][result]
                    results.append(result[0])

                    media_label = labels[number]
                    if result[0] == 0:
                        print("file name: %s, result: cover, label: %r, prob of prediction: %.2f%%" % (media_name, media_label, prob))

                    if result[0] == 1:
                        print("file name: %s, result: stego, label: %r, prob of prediction: %.2f%%" % (media_name, media_label, prob))

                    number += 1
            accuracy = 100 * (np.count_nonzero(np.array(results) == labels) / len(results))
            print("Accuracy = %.2f%%" % accuracy)
