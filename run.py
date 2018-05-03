#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
from utils import *
from network import *
from config import file_path_setup
from file_preprocess import get_file_name

"""
Created on 2017.11.27
Finished on 2017.11.30
@author: Wang Yuntao
"""

"""
    function: 
        train(args)                                             # 训练
        steganalysis_one(args)                                  # 隐写分析网络测试 (单个文件)
        steganalysis_batch(args)                                # 隐写分析网络测试 (多个文件)
"""


def run_mode(args):
    if args.mode == "train":  # train mode
        train(args)
    elif args.mode == "test":  # test mode
        if args.submode == "one":
            steganalysis_one(args)
        if args.submode == "batch":
            steganalysis_batch(args)
    else:
        print("Mode Error")


def train(args):
    # hyper parameters (超参)
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
    model_file_name = args.model_file_name                                                  # model file name
    step_train, step_valid = 0, 0                                                           # train step and valid step
    max_accuracy, max_accuracy_epoch = 0, 0                                                 # max valid accuracy and corresponding epoch

    with tf.device("/cpu:0"):
        global_step = tf.Variable(initial_value=0,
                                  trainable=False,
                                  name="global_step",
                                  dtype=tf.int32)                                           # global step (Variable 变量不能直接分配GPU资源)

    # pre processing
    is_abs, is_trunc, threshold, is_diff, order, direction, is_diff_abs, downsampling, block = \
        args.is_abs, args.is_trunc, args.threshold, args.is_diff, args.order, args.direction, args.is_diff_abs, args.downsampling, args.block

    # learning rate decay (学习率递减方法)
    learning_rate = learning_rate_decay(init_learning_rate=init_learning_rate,
                                        decay_method=decay_method,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=decay_rate)                             # learning rate

    # placeholder
    height, width, channel = args.height, args.width, 1                                    # the height, width and channel of the QMDCT matrix
    if is_diff is True and direction == 0:
        height_new, width_new = height - order, width
    elif is_diff is True and direction == 1:
        height_new, width_new = height, width - order
    else:
        height_new, width_new = height, width

    data = tf.placeholder(dtype=tf.float32, shape=(None, height_new, width_new, channel), name="QMDCTs")
    label = tf.placeholder(dtype=tf.int32, shape=(None, ), name="label")

    # start session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # initialize the network
    command = args.network + "(data, classes_num)"
    logits = eval(command)

    # file path (文件路径)
    cover_train_files_path, cover_valid_files_path, stego_train_files_path, stego_valid_files_path, model_file_path, log_file_path = file_path_setup(args)
    print("train files path(cover): %s" % cover_train_files_path)
    print("valid files path(cover): %s" % cover_valid_files_path)
    print("train files path(stego): %s" % stego_train_files_path)
    print("valid files path(stego): %s" % stego_valid_files_path)
    print("model files path: %s" % model_file_path)
    print("log files path: %s" % log_file_path)

    # information output
    print("batch size(train): %d, batch size(valid): %d, total epoch: %d, class number: %d, initial learning rate: %f, "
          "decay method: %s, decay rate: %f, decay steps: %d" % (batch_size_train, batch_size_valid, n_epoch, classes_num,
                                                                 init_learning_rate, decay_method, decay_rate, decay_steps))
    print("start load network...")

    # evaluation
    loss = loss_layer(logits=logits, label=label, is_regulation=is_regulation, coeff=coeff_regulation, method=loss_method)
    train_optimizer = optimizer(losses=loss, learning_rate=learning_rate, global_step=global_step)
    accuracy = accuracy_layer(logits=logits, label=label)

    with tf.device("/cpu:0"):
        tf.summary.scalar("loss_train", loss)
        tf.summary.scalar("accuracy_train", accuracy)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=max_to_keep,
                               keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    # initialize
    sess = tf.InteractiveSession()
    train_writer_train = tf.summary.FileWriter(log_file_path + "/train", tf.get_default_graph())
    train_writer_valid = tf.summary.FileWriter(log_file_path + "/valid", tf.get_default_graph())
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Start training...")
    for epoch in range(n_epoch):
        start_time = time.time()

        # read files list(train, 读取文件列表, default: shuffle)
        cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list = read_data(cover_train_files_path,
                                                                                                                 stego_train_files_path,
                                                                                                                 start_index_train,
                                                                                                                 end_index_train)

        # read files list(valid, 读取文件列表, default: shuffle)
        cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list = read_data(cover_valid_files_path,
                                                                                                                 stego_valid_files_path,
                                                                                                                 start_index_valid,
                                                                                                                 end_index_valid)
        # update the learning rate (学习率更新)
        lr = sess.run(learning_rate)

        # train (训练)
        train_iterations, train_loss, train_accuracy = 0, 0, 0
        for x_train_batch, y_train_batch in \
                minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size_train):
            # data read and process (数据读取与处理)
            x_train_data = get_data(x_train_batch, height, width, carrier=carrier, is_diff=is_diff, order=order, direction=direction, is_diff_abs=is_diff_abs,
                                    is_trunc=is_trunc, threshold=threshold)

            # get the accuracy and loss (训练与指标显示)
            _, err, ac, summary_str_train = sess.run([train_optimizer, loss, accuracy, summary_op], feed_dict={data: x_train_data, label: y_train_batch})

            train_loss += err
            train_accuracy += ac
            step_train += 1
            train_iterations += 1
            train_writer_train.add_summary(summary_str_train, global_step=step_train)

            print("epoch: %003d, train iterations: %003d: train loss: %f, train accuracy: %f" % (epoch + 1, train_iterations, err, ac))

        print("==================================================================================")

        # valid (验证)
        valid_iterations, valid_loss, valid_accuracy = 0, 0, 0
        for x_valid_batch, y_valid_batch in \
                minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size_valid):
            # data read and process (数据读取与处理)
            x_valid_data = get_data(x_valid_batch, height, width, carrier=carrier, is_diff=is_diff, order=order, direction=direction, is_diff_abs=is_diff_abs,
                                    is_trunc=is_trunc, threshold=threshold)

            # get the accuracy and loss (验证与指标显示)
            err, ac, summary_str_valid = sess.run([loss, accuracy, summary_op], feed_dict={data: x_valid_data, label: y_valid_batch})
            valid_loss += err
            valid_accuracy += ac
            valid_iterations += 1
            step_valid += 1
            train_writer_valid.add_summary(summary_str_valid, global_step=step_valid)

            print("epoch: %003d, valid iterations: %003d, valid loss: %f, valid accuracy: %f" % (epoch + 1, valid_iterations, err, ac))

        # calculate the average in a batch (计算每个batch内的平均值)
        train_loss_average = train_loss / train_iterations
        valid_loss_average = valid_loss / valid_iterations
        train_accuracy_average = train_accuracy / train_iterations
        valid_accuracy_average = valid_accuracy / valid_iterations

        # model save (保存模型)
        if valid_accuracy_average > max_accuracy:
            max_accuracy = valid_accuracy_average
            max_accuracy_epoch = epoch + 1
            saver.save(sess, os.path.join(model_file_path, model_file_name), global_step=global_step)
            print("The model is saved successfully.")

        print("epoch: %003d, learning rate: %f, train loss: %f, train accuracy: %f, valid loss: %f, valid accuracy: %f, "
              "max valid accuracy: %f, max valid acc epoch: %d" % (epoch + 1, lr, train_loss_average, train_accuracy_average, valid_loss_average,
                                                                   valid_accuracy_average, max_accuracy, max_accuracy_epoch))

        end_time = time.time()
        print("Runtime: %.2fs" % (end_time - start_time))

    train_writer_train.close()
    train_writer_valid.close()
    sess.close()


def steganalysis_one(args):
    height, width = args.height, args.width
    model_file_path = args.model_file_path
    image_file_path = args.test_file_path

    data = tf.placeholder(tf.float32, [1, height, width, 1], name="image")

    command = args.network + "(data, 2, is_bn=False)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    # read image
    img = io.imread(image_path)
    img = np.reshape(img, [1, height, width, 1])
    image_name = get_file_name(image_file_path)

    # 加载模型
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_file_path)
        print("The model is loaded successfully.")

        # predict
        ret = sess.run(logits, feed_dict={data: img})
        result = np.argmax(ret, 1)

        if result == 1:
            print("%s: stego" % image_name)
        if result == 0:
            print("%s: cover" % image_name)


def steganalysis_batch(args):
    height, width = args.height, args.width
    model_file_path = args.model_file_path
    image_files_path = args.test_files_dir
    label_file_path = args.label_file_path

    image_list = get_files_list(image_files_path)
    image_num = len(image_list)
    data = tf.placeholder(tf.float32, [1, height, width, 1], name="image")

    if label_file_path is not None:
        label = list()
        with open(label_file_path) as file:
            for line in file.readlines():
                label.append(line)
    else:
        label = np.zeros([image_num, 1])

    command = args.network + "(data, 2, is_bn=False)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    # 加载模型
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_file_path)
        print("The model is loaded successfully.")

        # read image
        count, i = 0, 0
        for image_file_path in image_list:
            img = io.imread(image_path)
            img = np.reshape(img, [1, height, width, 1])
            image_name = get_file_name(image_file_path)

            # predict
            ret = sess.run(logits, feed_dict={data: img})
            ret[0][0] = ret[0][0]
            ret[0][1] = ret[0][1]
            result = np.argmax(ret, 1)

            if result == 1:
                print("%s: stego" % image_name)
                if int(label[i]) == 1:
                    count = count + 1
            if result == 0:
                print("%s: cover" % image_name)
                if int(label[i]) == 0:
                    count = count + 1
            i = i + 1

    if label_file_path is not None:
        print("Accuracy = %.2f" % (count / image_num))
