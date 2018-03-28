#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from model import *
from utils import *
from config import file_path_setup

"""
Created on 2017.11.27
Finished on 2017.11.30
@author: Wang Yuntao
"""

"""
    function: 
        test: test
        test_batch: batch test
"""


def train_audio(args):
    # hyper parameters (超参)
    batch_size = args.batch_size                                                            # batch size
    init_learning_rate = args.learning_rate                                                 # initialized learning rate
    n_epoch = args.epoch                                                                    # epoch
    decay_steps, decay_rate = args.decay_step, args.decay_rate                              # decay steps | decay rate
    is_regulation = args.is_regulation                                                      # regulation or not
    classes_num = 2                                                                         # classes number
    max_to_keep = args.max_to_keep                                                          # maximum number of recent checkpoints to keep
    keep_checkpoint_every_n_hours = args.keep_checkpoint_every_n_hours                      # how often to keep checkpoints
    start_index_train, end_index_train = args.start_index_train, args.end_index_train       # the scale of the dataset (train)
    start_index_valid, end_index_valid = args.start_index_valid, args.end_index_valid       # the scale of the dataset (valid)
    model_file_name = args.model_file_name                                                  # model file name
    step_train, step_test = 0, 0

    global_step = tf.Variable(initial_value=0,
                              trainable=False,
                              name="global_step",
                              dtype=tf.int32)  # global step

    # pre processing
    is_abs, is_trunc, threshold, is_diff, order, direction, downsampling, block = \
        args.is_abs, args.is_trunc, args.threshold, args.is_diff, args.order, args.direction, args.downsampling, args.block

    # learning rate decay (学习率递减方法)
    learning_rate = learning_rate_decay(init_learning_rate=init_learning_rate,
                                        decay_method="exponential",
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=decay_rate)                      # learning rate

    # file path (文件路径)
    cover_train_files_path, cover_valid_files_path, stego_train_files_path, stego_valid_files_path, model_file_path, log_file_path = file_path_setup(args)
    print("train files path(cover): %s" % cover_train_files_path)
    print("valid files path(cover): %s" % cover_valid_files_path)
    print("train files path(stego): %s" % stego_train_files_path)
    print("valid files path(stego): %s" % stego_valid_files_path)
    print("model files path: %s" % model_file_path)
    print("log files path: %s" % log_file_path)

    # placeholder
    height, width, channel = args.height, args.width, 1                             # the height, width and channel of the QMDCT matrix
    if is_diff is True and direction == 0:
        height_new, width_new = height - order, width
    elif is_diff is True and direction == 1:
        height_new, width_new = height, width - order
    else:
        height_new, width_new = height, width

    data = tf.placeholder(tf.float32, [batch_size, height_new, width_new, channel], name="QMDCTs")
    label = tf.placeholder(tf.int32, [batch_size, ], name="label")

    # start session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)

    # initialize the network
    command = args.network + "(data, classes_num)"
    logits = eval(command)

    # information output
    print("batch_size: %d, total_epoch: %d, class_num: %d" % (batch_size, n_epoch, classes_num))
    print("start load network...")
    time.sleep(3)

    # evaluation
    if is_regulation is True:
        tv = tf.trainable_variables()
        regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    else:
        regularization_cost = 0
    loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits) + regularization_cost
    # loss = tf.reduce_sum(cross_entropy) + regularization_cost
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), label)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", acc)

    # print(logits, logits.eval, logits.get_shape())
    # tv = tf.trainable_variables()
    # loss = loss_layer(logits, label, is_regulation=False, coeff=1e-4)                       # loss
    # train_op = optimizer(loss, learning_rate=learning_rate)                                         # the result of the optimizer
    # acc = accuracy_layer(logits, label)                                                             # accuracy

    # initialize tensorboard
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=max_to_keep,
                           keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
    sess = tf.InteractiveSession()
    train_writer_train = tf.summary.FileWriter(log_file_path + "/train", tf.get_default_graph())
    train_writer_val = tf.summary.FileWriter(log_file_path + "/validation", tf.get_default_graph())
    init = tf.global_variables_initializer()
    sess.run(init)

    max_acc = 0
    print("Start training...")
    for epoch in range(n_epoch):
        start_time = time.time()

        cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list = read_data(cover_train_files_path,
                                                                                                                 stego_train_files_path,
                                                                                                                 start_index_train,
                                                                                                                 end_index_train)   # 读取文件列表(默认shuffle)

        cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list = read_data(cover_valid_files_path,
                                                                                                                 stego_valid_files_path,
                                                                                                                 start_index_valid,
                                                                                                                 end_index_valid)   # 读取文件列表(默认shuffle)
        # update the learning rate (学习率更新)
        lr = sess.run(learning_rate)

        # training (训练)
        n_batch_train, train_loss, train_acc = 0, 0, 0
        for x_train_a, y_train_a in minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size):
            # data read and process (数据读取与处理)
            x_train_data = read_text_batch(x_train_a, height, width, is_diff=is_diff, order=order, direction=direction, is_trunc=is_trunc, threshold=threshold)

            # get the accuracy and loss (训练与指标显示)
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={data: x_train_data, label: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch_train += 1
            step_train += 1
            summary_str_train = sess.run(summary_op, feed_dict={data: x_train_data, label: y_train_a})
            train_writer_train.add_summary(summary_str_train, step_train)

            print("train_iter-%d: train_loss: %f, train_acc: %f" % (n_batch_train, err, ac))

        # validation (验证)
        n_batch_val, val_loss, val_acc = 0, 0, 0
        for x_val_a, y_val_a in minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size):
            # data read and process (数据读取与处理)
            x_val_data = read_text_batch(x_val_a, height, width, is_diff=is_diff, order=order, direction=direction, is_trunc=is_trunc, threshold=threshold)

            # get the accuracy and loss (验证与指标显示)
            err, ac = sess.run([loss, acc], feed_dict={data: x_val_data, label: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch_val += 1
            step_test += 1
            summary_str_val = sess.run(summary_op, feed_dict={data: x_val_data, label: y_val_a})
            train_writer_val.add_summary(summary_str_val, step_test)
            
            print("validation_iter-%d: loss: %f, acc: %f" % (n_batch_val, err, ac))

        print("epoch: %d, learning_rate: %f -- train loss: %f, train acc: %f, validation loss: %f, validation acc: %f"
              % (epoch + 1, lr, train_loss / n_batch_train, train_acc / n_batch_train,
                 val_loss / n_batch_val, val_acc / n_batch_val))

        end_time = time.time()
        print("Runtime: %.2fs" % (end_time - start_time))

        # model save (保存模型)
        if val_acc > max_acc:
            max_acc = val_acc
            saver.save(sess, os.path.join(model_file_path, model_file_name), global_step=global_step)
            print("The model is saved successfully.")

    train_writer_train.close()
    train_writer_val.close()
    sess.close()


def train_image(args):
    # hyper parameters (超参)
    batch_size = args.batch_size                                                                    # batch size
    init_learning_rate = args.learning_rate                                                         # initialized learning rate
    n_epoch = args.epoch                                                                            # epoch
    decay_method, decay_steps, decay_rate = args.decay_method, args.decay_step, args.decay_rate     # decay_method, decay steps | decay rate
    is_regulation = args.is_regulation                                                              # regulation or not
    classes_num = 2                                                                                 # classes number
    max_to_keep = args.max_to_keep                                                                  # maximum number of recent checkpoints to keep
    keep_checkpoint_every_n_hours = args.keep_checkpoint_every_n_hours                              # how often to keep checkpoints
    start_index_train, end_index_train = args.start_index_train, args.end_index_train               # the scale of the dataset (train)
    start_index_valid, end_index_valid = args.start_index_valid, args.end_index_valid               # the scale of the dataset (valid)
    model_file_name = args.model_file_name                                                          # model file name
    step_train, step_test = 0, 0                                                                    # the step of train and test

    global_step = tf.Variable(initial_value=0,
                              trainable=False,
                              name="global_step",
                              dtype=tf.int32)                                               # global step (as the concept of iterations in caffe)

    # learning rate decay (学习率递减方法)
    learning_rate = learning_rate_decay(init_learning_rate=init_learning_rate,
                                        decay_method=decay_method,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=decay_rate)                              # learning rate

    # file path (文件路径)
    cover_train_files_path, cover_valid_files_path, stego_train_files_path, stego_valid_files_path, model_file_path, log_file_path = file_path_setup(args)
    print("train files path(cover): %s" % cover_train_files_path)
    print("valid files path(cover): %s" % cover_valid_files_path)
    print("train files path(stego): %s" % stego_train_files_path)
    print("valid files path(stego): %s" % stego_valid_files_path)
    print("model files path: %s" % model_file_path)
    print("log files path: %s" % log_file_path)

    # placeholder
    height, width, channel = args.height, args.width, 1                                     # the height, width and channel of the image
    data = tf.placeholder(tf.float32, [batch_size, height, width, channel], name="image")
    label = tf.placeholder(tf.int32, [batch_size, ], name="label")

    # start session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)

    # initialize the network
    command = args.network + "(data, classes_num)"
    logits = eval(command)

    # information output
    print("batch_size: %d, total_epoch: %d, class_num: %d" % (batch_size, n_epoch, classes_num))
    print("start load network...")
    time.sleep(3)

    # evaluation
    if is_regulation is True:
        tv = tf.trainable_variables()
        regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
    else:
        regularization_cost = 0
    loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits) + regularization_cost
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), label)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", acc)

    # print(logits, logits.eval, logits.get_shape())
    # tv = tf.trainable_variables()
    # loss = loss_layer(logits, label, is_regulation=False, coeff=1e-4)                               # loss
    # train_op = optimizer(loss, learning_rate=learning_rate)                                         # the result of the optimizer
    # acc = accuracy_layer(logits, label)                                                             # accuracy

    # initialize tensorboard
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=max_to_keep,
                           keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
    sess = tf.InteractiveSession()
    train_writer_train = tf.summary.FileWriter(log_file_path + "/train", tf.get_default_graph())
    train_writer_val = tf.summary.FileWriter(log_file_path + "/validation", tf.get_default_graph())
    init = tf.global_variables_initializer()
    sess.run(init)

    max_acc = 0
    print("Start training...")
    for epoch in range(n_epoch):
        start_time = time.time()

        cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list = read_data(cover_train_files_path,
                                                                                                                 stego_train_files_path,
                                                                                                                 start_index_train,
                                                                                                                 end_index_train)  # 读取文件列表(默认shuffle)

        cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list = read_data(cover_valid_files_path,
                                                                                                                 stego_valid_files_path,
                                                                                                                 start_index_valid,
                                                                                                                 end_index_valid)  # 读取文件列表(默认shuffle)
        # update the learning rate (学习率更新)
        lr = sess.run(learning_rate)

        # training (训练)
        n_batch_train, train_loss, train_acc = 0, 0, 0
        for x_train_a, y_train_a in minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size):
            # data read and process (数据读取与处理)
            x_train_data = read_image_batch(x_train_a)

            # get the accuracy and loss (训练与指标显示)
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={data: x_train_data, label: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch_train += 1
            step_train += 1
            summary_str_train = sess.run(summary_op, feed_dict={data: x_train_data, label: y_train_a})
            train_writer_train.add_summary(summary_str_train, step_train)

            print("train_iter-%d: train_loss: %f, train_acc: %f" % (n_batch_train, err, ac))

        # validation (验证)
        n_batch_val, val_loss, val_acc = 0, 0, 0
        for x_val_a, y_val_a in minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size):
            # data read and process (数据读取与处理)
            x_val_data = read_image_batch(x_val_a)

            # get the accuracy and loss (验证与指标显示)
            err, ac = sess.run([loss, acc], feed_dict={data: x_val_data, label: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch_val += 1
            step_test += 1
            summary_str_val = sess.run(summary_op, feed_dict={data: x_val_data, label: y_val_a})
            train_writer_val.add_summary(summary_str_val, step_test)

            print("validation_iter-%d: loss: %f, acc: %f" % (n_batch_val, err, ac))

        print("epoch: %d, learning_rate: %f -- train loss: %f, train acc: %f, validation loss: %f, validation acc: %f"
              % (epoch + 1, lr, train_loss / n_batch_train, train_acc / n_batch_train,
                 val_loss / n_batch_val, val_acc / n_batch_val))

        end_time = time.time()
        print("Runtime: %.2fs" % (end_time - start_time))

        # model save (保存模型)
        if val_acc > max_acc:
            max_acc = val_acc
            saver.save(sess, os.path.join(model_file_path, model_file_name), global_step=global_step)
            print("The model is saved successfully.")

    train_writer_train.close()
    train_writer_val.close()
    sess.close()


def test_batch(model_dir, files_dir):
    """
    单个文件测试
    :param model_dir: 模型存储路径
    :param files_dir: 待检测文件目录
    :return: NULL
    """
    height = 198
    width = 576

    # 设定占位符
    x = tf.placeholder(tf.float32, [1, height, width, 1], name="QMDCTs")
    y_ = tf.placeholder(tf.int32, [1, ], name="label")
    network1(x, 2)

    # 加载模型
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, model_file)

    files_list = get_files_list(files_dir)  # 获取文件列表
    data = np.zeros([1, 198, 576, 1], dtype=np.float32)
    labels = []
    for file in files_list:
        data[0, :, :, :] = read_text(file)
        # labels.append(file_label)
        labels = np.asarray(labels, np.float32)
        ret = sess.run(y_, feed_dict={x: data, y_: labels})

    return ret


def test(model_dir, file_path, file_label):
    """
    单个文件测试
    :param model_dir: 模型存储路径
    :param file_path: 待检测文件路径
    :param file_label: 待检测文件label
    :return: NULL
    """
    height = 198
    width = 576
    label = ["cover", "stego"]

    # 设定占位符
    x = tf.placeholder(tf.float32, [1, height, width, 1], name="QMDCTs")
    y_ = tf.placeholder(tf.int32, [1, ], name="label")
    network1(x, 2)

    # 加载模型
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, model_file)

    # 读取数据
    data = np.zeros([1, 198, 576, 1], dtype=np.float32)
    labels = []
    data[0, :, :, :] = read_text(file_path)
    labels.append(file_label)
    labels = np.asarray(labels, np.float32)
    ret = sess.run(y_, feed_dict={x: data, y_: labels})

    print("测试结果: %s, 实际结果: %s" % (label[ret[0]], label[file_label]))


def steganalysis_batch(model_dir, files_dir):
    """
    单个文件测试
    :param model_dir: 模型存储路径
    :param files_dir: 待检测文件目录
    :return: NULL
    """
    height = 198
    width = 576
    label = ["cover", "stego"]

    # 设定占位符
    x = tf.placeholder(tf.float32, [1, height, width, 1], name="QMDCTs")
    y = network1(x, 2, is_bn=False)
    logits = tf.nn.softmax(y, 1)

    # 加载模型
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model_file = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, model_file)
    print("The model is loaded successfully.")

    files_list = get_files_list(files_dir)  # 获取文件列表
    data = np.zeros([1, 198, 576, 1], dtype=np.float32)
    for file in files_list:
        data[0, :, :, :] = read_text(file)
        ret = sess.run(logits, feed_dict={x: data})
        print(ret)
        result = ret.argmax()
        file_name = file.split(sep="/")[-1]
        print("文件: %s, 分析结果: %s" % (file_name, label[int(result)]))

    sess.close()
