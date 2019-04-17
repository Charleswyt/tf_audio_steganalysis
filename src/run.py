#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2017.11.27
Finished on 2017.11.30
Modified on 2018.09.18

@author: Yuntao Wang
"""

import time
import datetime
import traceback
from utils import *
from networks.networks import networks
from file_preprocess import get_file_name
from networks.audio_steganalysis import *
from networks.image_steganalysis import *
from networks.tested_steganalysis import *
from networks.image_classification import *

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
    decay_method = args.decay_method                                                        # decay method
    decay_steps, decay_rate = args.decay_step, args.decay_rate                              # decay steps | decay rate
    loss_method = args.loss_method                                                          # the calculation method of loss function
    is_regulation = args.is_regulation                                                      # regulation or not
    coeff_regulation = args.coeff_regulation                                                # the gain of regulation
    classes_num = args.class_num                                                            # classes number
    carrier = args.carrier                                                                  # carrier (qmdct | audio | image)
    task_name = args.task_name                                                              # task name
    checkpoint = args.checkpoint                                                            # checkpoint
    n_epoch = args.epoch if args.indicator == "epoch" else 500                              # epoch

    max_to_keep = args.max_to_keep                                                          # maximum number of recent checkpoints to keep
    keep_checkpoint_every_n_hours = args.keep_checkpoint_every_n_hours                      # how often to keep checkpoints
    start_index_train, end_index_train = args.start_index_train, args.end_index_train       # the scale of the dataset (train)
    start_index_valid, end_index_valid = args.start_index_valid, args.end_index_valid       # the scale of the dataset (valid)
    step_train, step_valid = 0, 0                                                           # train step and valid step
    file_type = args.file_type                                                              # file type
    max_accuracy, max_accuracy_epoch = 0, 0                                                 # max validation accuracy and corresponding epoch
    corresponding_fpr, corresponding_fnr = 0, 0                                             # the corresponding fpr and fnr in the max validation accuracy

    # file path (full mode)
    cover_train_path = args.cover_train_path
    stego_train_path = args.stego_train_path
    cover_valid_path = args.cover_valid_path
    stego_valid_path = args.stego_valid_path

    # file path (semi mode)
    cover_files_path = args.cover_files_path
    stego_files_path = args.stego_files_path

    model_path = args.model_path
    log_path = args.log_path

    # information output
    if cover_train_path is not None:
        print("train files path(cover): %s" % cover_train_path)
        print("valid files path(cover): %s" % cover_valid_path)
        print("train files path(stego): %s" % stego_train_path)
        print("valid files path(stego): %s" % stego_valid_path)
    else:
        print("cover files path: %s" % cover_files_path)
        print("stego files path: %s" % stego_files_path)

    print("model files path: %s" % model_path)
    print("log files path: %s" % log_path)
    print("batch size: %d, total epoch: %d, class number: %d, initial learning rate: %f, decay method: %s, decay rate: %f, decay steps: %d"
          % (batch_size, n_epoch, classes_num, init_learning_rate, decay_method, decay_rate, decay_steps))
    print("start load network...")

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
    with tf.variable_scope("placeholder"):
        data = tf.placeholder(dtype=tf.float32, shape=(None, height, width), name="data") if channel == 0 else \
            tf.placeholder(dtype=tf.float32, shape=(None, height, width, channel), name="data")
        labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name="labels")
        is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

        # placeholder for siamese network
        data_siamese = tf.placeholder(dtype=tf.float32, shape=(None, height, width), name="data_siamese") if channel == 0 else \
            tf.placeholder(dtype=tf.float32, shape=(None, height, width, channel), name="data_siamese")
        labels_siamese = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels_siamese")
        new_labels = tf.placeholder(dtype=tf.float32, shape=(None, ), name="new_labels")

    # initialize the network
    if args.network not in networks:
        print("Network miss-match, please try again.")
        return False

    if args.siamese is not True:
        try:
            command = args.network + "(data, classes_num, is_bn)"
            logits = eval(command)
        except TypeError:
            command = args.network + "(data, classes_num)"
            logits = eval(command)

        # evaluation
        loss_train = loss_layer(logits=logits, labels=labels, is_regulation=is_regulation, coeff=coeff_regulation, method=loss_method)
        accuracy_train = accuracy_layer(logits=logits, labels=labels)
        accuracy_validation, false_positive_rate_validation, false_negative_rate_validation = evaluation(logits=logits, labels=labels)
        train_optimizer = optimizer(losses=loss_train, learning_rate=learning_rate, global_step=global_step)
        loss_validation = loss_layer(logits=logits, labels=labels, is_regulation=is_regulation, coeff=coeff_regulation, method=loss_method)
        # accuracy_check = accuracy_layer(logits=logits, labels=labels)

    else:
        with tf.variable_scope('siamese') as scope:
            command1 = args.network + "(data, classes_num, is_bn)"
            command2 = args.network + "(data_siamese, classes_num, is_bn)"
            logits = eval(command1)
            scope.reuse_variables()
            logits_siamese = eval(command2)

        # evaluation
        loss_train = loss_layer(logits=logits, logits_siamese=logits_siamese, labels=new_labels, is_regulation=is_regulation, coeff=coeff_regulation, method="siamese_loss")
        accuracy1 = evaluation(logits=logits, labels=labels)
        accuracy2 = accuracy_layer(logits=logits_siamese, labels=labels_siamese)
        accuracy_train = tf.multiply(tf.add(accuracy1, accuracy2), 0.5)
        train_optimizer = optimizer(losses=loss_train, learning_rate=learning_rate, global_step=global_step)
        loss_validation = loss_layer(logits=logits, labels=labels, is_regulation=is_regulation, coeff=coeff_regulation, method=loss_method)
        accuracy_validation, false_positive_rate_validation, false_negative_rate_validation = evaluation(logits=logits, labels=labels)

    with tf.device("/cpu:0"):
        tf.summary.scalar("loss_train", loss_train)
        tf.summary.scalar("loss_validation", loss_validation)
        tf.summary.scalar("accuracy_train", accuracy_train)
        tf.summary.scalar("accuracy_validation", accuracy_validation)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=max_to_keep,
                               keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    # initialize
    try:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            train_writer_train = tf.summary.FileWriter(log_path + "/train", tf.get_default_graph())
            train_writer_valid = tf.summary.FileWriter(log_path + "/validation", tf.get_default_graph())
            init = tf.global_variables_initializer()
            sess.run(init)

            # restore the model and keep training from the current breakpoint
            if checkpoint is True:
                if args.fine_tune_model_file_path is None:
                    model_file_path = get_model_file_path(model_path)
                    if model_file_path is not None:
                        saver.restore(sess, model_file_path)
                    else:
                        model_file_path = fullfile(model_path, task_name)
                        print("No such model file, training starts from the beginning.")
                else:
                    if not os.path.exists(args.fine_tune_model_file_path + ".index"):
                        model_file_path = fullfile(model_path, task_name)
                        print("No such model file, training starts from the beginning.")
                    else:
                        saver.restore(sess, args.fine_tune_model_file_path)
                        model_file_path = fullfile(model_path, task_name)
                        print("Fine-Tune on the model %s" % args.fine_tune_model_file_path)
            else:
                model_file_path = fullfile(model_path, task_name)
                print("Training starts from the beginning.")

            print("Start training...")
            print("Numbers of network %s: %d" % (args.network, get_variables_number(tf.trainable_variables())))
            print("Input data: (%d, %d)" % (height, width)) if channel == 0 else print("Input data: (%d, %d, %d)" % (height, width, channel))

            start_time_all = time.time()
            for epoch in range(n_epoch):

                # read data
                if cover_train_path is not None:
                    # read files list (train)
                    cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list \
                        = read_data(cover_train_path, stego_train_path, start_idx=start_index_train, end_idx=end_index_train, file_type=file_type)

                    # read files list (validation)
                    cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list \
                        = read_data(cover_valid_path, stego_valid_path, start_idx=start_index_valid, end_idx=end_index_valid, file_type=file_type)
                else:
                    # read files list (train)
                    cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list \
                        = read_data(cover_files_path, stego_files_path, start_idx=start_index_train, end_idx=end_index_train, file_type=file_type)

                    # read files list (validation)
                    cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list \
                        = read_data(cover_files_path, stego_files_path, start_idx=start_index_valid, end_idx=end_index_valid, file_type=file_type)

                # update the learning rate
                lr = sess.run(learning_rate)

                # train
                train_iterations, train_loss, train_accuracy = 0, 0, 0
                for x_train_batch, y_train_batch in \
                        minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size):
                    # data read and process
                    x_train_data = get_data_batch(x_train_batch, height=height, width=width, channel=channel, carrier=carrier)

                    if args.siamese is not True:
                        # get the accuracy and loss
                        _, err, ac, summary_str_train = sess.run([train_optimizer, loss_train, accuracy_train, summary_op],
                                                                 feed_dict={data: x_train_data, labels: y_train_batch, is_bn: True})
                    else:
                        x_train_data1 = x_train_data[:int(batch_size / 2)]
                        x_train_data2 = x_train_data[int(batch_size / 2):]
                        y_train_batch1 = y_train_batch[:int(batch_size / 2)]
                        y_train_batch2 = y_train_batch[int(batch_size / 2):]

                        new_label = list(np.array(np.array(np.array(y_train_batch1) == np.array(y_train_batch2), dtype=np.float32), dtype=np.str))

                        _, err, ac, summary_str_train = sess.run([train_optimizer, loss_train, accuracy_train, summary_op],
                                                                 feed_dict={data: x_train_data1, data_siamese: x_train_data2,
                                                                            labels: y_train_batch1, labels_siamese: y_train_batch2,
                                                                            new_labels: new_label, is_bn: True})

                    train_loss += err
                    train_accuracy += ac
                    step_train += 1
                    train_iterations += 1
                    train_writer_train.add_summary(summary_str_train, global_step=step_train)

                    et = time.time() - start_time_all
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print("[network: %s, task: %s, global_step: %000005d] elapsed: %s, epoch: %003d, train iterations: %003d: train loss: %f, train accuracy: %f"
                          % (args.network, args.task_name, step_train, et, epoch + 1, train_iterations, err, ac))

                print("=" * 150)

                # validation
                valid_iterations, valid_loss, valid_accuracy, valid_fpr, valid_fnr = 0, 0, 0, 0, 0
                for x_valid_batch, y_valid_batch in \
                        minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size):
                    # data read and process
                    x_valid_data = get_data_batch(x_valid_batch, height=height, width=width, channel=channel, carrier=carrier)

                    # get the accuracy and loss
                    new_label = np.zeros(batch_size, dtype=np.float32)
                    err, ac, fpr, fnr, summary_str_valid = sess.run([loss_validation, accuracy_validation,
                                                                     false_negative_rate_validation, false_positive_rate_validation, summary_op],
                                                                    feed_dict={data: x_valid_data, data_siamese: x_valid_data,
                                                                               labels: y_valid_batch, labels_siamese: y_valid_batch,
                                                                               new_labels: new_label, is_bn: True})
                    valid_loss += err
                    valid_accuracy += ac
                    valid_fpr += fpr
                    valid_fnr += fnr
                    valid_iterations += 1
                    step_valid += 1
                    train_writer_valid.add_summary(summary_str_valid, global_step=step_valid)

                    et = time.time() - start_time_all
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print("[network: %s, task: %s] elapsed: %s, epoch: %003d, valid iterations: %003d, valid loss: %-8f, "
                          "valid accuracy: %f, valid fpr: %f, valid fnr: %f"
                          % (args.network, args.task_name, et, epoch + 1, valid_iterations, err, ac, fpr, fnr))

                # calculate the average in a batch
                train_loss_average = train_loss / train_iterations
                valid_loss_average = valid_loss / valid_iterations
                train_accuracy_average = train_accuracy / train_iterations
                valid_accuracy_average = valid_accuracy / valid_iterations
                valid_fpr_average = valid_fpr / valid_iterations
                valid_fnr_average = valid_fnr / valid_iterations

                # training stop and model save
                if valid_accuracy_average > max_accuracy:
                    max_accuracy_epoch = epoch + 1
                    saver.save(sess, model_file_path, global_step=global_step)
                    print("The model is saved successfully.")
                    max_accuracy, corresponding_fpr, corresponding_fnr = valid_accuracy_average, valid_fpr_average, valid_fnr_average

                print("[network: %s, task: %s] epoch: %003d, learning rate: %f, train loss: %f, train accuracy: %f, valid loss: %f, "
                      "valid accuracy: %f, valid FPR: %f, valid FNR: %f, "
                      "max valid accuracy: %f -- FPR: %f, FNR: %f, "
                      "max valid acc epoch: %d"
                      % (args.network, args.task_name, epoch + 1, lr, train_loss_average, train_accuracy_average,
                         valid_loss_average, valid_accuracy_average, valid_fpr_average, valid_fnr_average,
                         max_accuracy, corresponding_fpr, corresponding_fnr, max_accuracy_epoch))
                print("Current Time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

                # training stop
                if (args.indicator == "validation_accuracy_value" and max_accuracy > args.validation_accuracy_expectation / 100) or\
                        (args.indicator == "epoch_interval" and max_accuracy_epoch - epoch > args.epoch_interval and
                         max_accuracy >= args.validation_accuracy_least / 100):
                    return

            train_writer_train.close()
            train_writer_valid.close()

    except Exception as e:
        print("An error occurred, please try again.")
        print("Error Type: ", e)
        print(traceback.print_exc())
        os.system("rm -rf " + model_path)
        os.system("rm -rf " + log_path)


def test(args):
    # hyper parameters
    batch_size = args.batch_size                                        # batch size
    height, width, channel = args.height, args.width, args.channel      # height and width of input matrix
    carrier = args.carrier                                              # carrier (qmdct | audio | image)
    classes_num = args.class_num                                        # classes number
    start_index_test, end_index_test = args.start_index_test, args.end_index_test
    is_shuffle = args.is_shuffle                                        # whether the files list shuffle or not

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
    output = tf.nn.softmax(logits)

    accuracy, false_positive_rate, false_negative_rate = evaluation(logits=logits, labels=labels)

    # information output
    print("cover files path: %s" % cover_test_files_path)
    print("stego files path: %s" % stego_test_files_path)
    print("class number: %d" % classes_num)
    print("Start load network...")

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    start_time = time.time()

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
            cover_test_data_list, cover_test_label_list, \
                stego_test_data_list, stego_test_label_list = read_data(cover_test_files_path, stego_test_files_path, start_index_test, end_index_test, is_shuffle=is_shuffle)

            if len(cover_test_data_list) < batch_size:
                batch_size = len(cover_test_data_list)

            csv_file = open('/home1/wyt/code/tf_audio_steganalysis/logit1.csv', 'w', encoding='utf-8', newline='\n')
            csvwriter = csv.writer(datacsv, delimiter=',')
            test_iterations, test_accuracy, test_fpr, test_fnr = 0, 0, 0, 0
            for x_test_batch, y_test_batch in \
                    minibatches(cover_test_data_list, cover_test_label_list, stego_test_data_list, stego_test_label_list, batch_size):
                # data read and process
                x_test_data = get_data_batch(x_test_batch, height=height, width=width, channel=channel, carrier=carrier)

                # get the accuracy and loss
                result, acc, fpr, fnr = sess.run([output, accuracy, false_positive_rate, false_negative_rate],
                                                 feed_dict={data: x_test_data, labels: y_test_batch, is_bn: True})
                for content in result:
                    csvwriter.writerow(str([content[0], content[1]]))

                test_accuracy += acc
                test_fpr += fpr
                test_fnr += fnr
                test_iterations += 1

                # print("Batch-%003d, accuracy: %f, fpr: %f, fnr: %f" % (test_iterations, acc, fpr, fnr))

            test_accuracy_average = test_accuracy / test_iterations
            test_fpr_average = test_fpr / test_iterations
            test_fnr_average = test_fnr / test_iterations
            print("Test accuracy: %.2f%%, FPR: %.2f%%, FNR: %.2f%%" % (100. * test_accuracy_average, 100. * test_fpr_average, 100. * test_fnr_average))
            csv_file.close()

    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print("Run Time: %s" % et)


def steganalysis_one(args):
    # hyper parameters
    height, width, channel = args.height, args.width, args.channel      # height and width of input matrix
    carrier = args.carrier                                              # carrier (qmdct | audio | image)

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
    command = args.network + "(data, args.class_num, is_bn)"
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

    # files index
    start_index_steganalysis = args.start_index_steganalysis
    end_index_steganalysis = args.end_index_steganalysis

    # path
    steganalysis_files_path = args.steganalysis_files_path

    # placeholder
    data = tf.placeholder(dtype=tf.float32, shape=(None, height, width, channel), name="data")
    is_bn = tf.placeholder(dtype=tf.bool, name="is_bn")

    # initialize the network
    if args.network not in networks:
        print("Network miss-match, please try again")
        return False

    # network
    command = args.network + "(data, args.class_num, is_bn)"
    logits = eval(command)
    logits = tf.nn.softmax(logits)

    start_time = time.time()

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
            file_list = file_list[start_index_steganalysis:end_index_steganalysis]
            print("files path: %s" % steganalysis_files_path)
            count_cover, count_stego = 0, 0

            for file_path in file_list:
                steganalysis_data = get_data_batch([file_path], width=width, height=height, channel=channel, carrier=carrier)

                file_name = get_file_name(file_path)
                ret = sess.run(logits, feed_dict={data: steganalysis_data, is_bn: False})
                sess.run(tf.local_variables_initializer())
                print(ret)
                result = np.argmax(ret, 1)
                prob = 100 * ret[0][result]

                if result[0] == 0:
                    print("file name: %s, result: cover, prob of prediction: %.2f%%" % (file_name, prob))
                    count_cover += 1

                if result[0] == 1:
                    print("file name: %s, result: stego, prob of prediction: %.2f%%" % (file_name, prob))
                    count_stego += 1

            print("Number of cover samples: %d" % count_cover)
            print("Number of stego samples: %d" % count_stego)

    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print("Run Time: %s" % et)
