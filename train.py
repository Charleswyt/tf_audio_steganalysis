#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 
Finished on 
@author: Wang Yuntao
"""
import time
from utils import read_data, minibatches, read_image_batch


def train_epoch(sess, n_epoch, learning_rate, batch_size, data, label):
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
        iters_one_epoch_train, train_loss, train_accuracy = 0, 0, 0
        for x_train_a, y_train_a in minibatches(cover_train_data_list, cover_train_label_list, stego_train_data_list, stego_train_label_list, batch_size):
            # data read and process (数据读取与处理)
            x_train_data = read_image_batch(x_train_a)

            # get the accuracy and loss (训练与指标显示)
            _, iter_loss, iter_accuracy = sess.run([train_op, loss, acc], feed_dict={data: x_train_data, label: y_train_a})
            train_loss += iter_loss
            train_accuracy += iter_accuracy
            iters_one_epoch_train += 1
            step_train += 1
            summary_str_train = sess.run(summary_op, feed_dict={data: x_train_data, label: y_train_a})
            train_writer_train.add_summary(summary_str_train, step_train)

            print("epoch-%d, train_iter-%d: train_loss: %f, train_acc: %f" % (epoch, iters_one_epoch, iter_loss, iter_accuracy))

        # valid (验证)
        iters_one_epoch_valid, valid_loss, valid_accuracy = 0, 0, 0
        for x_val_a, y_val_a in minibatches(cover_valid_data_list, cover_valid_label_list, stego_valid_data_list, stego_valid_label_list, batch_size):
            # data read and process (数据读取与处理)
            x_val_data = read_image_batch(x_val_a)

            # get the accuracy and loss (验证与指标显示)
            iter_loss, iter_accuracy = sess.run([loss, acc], feed_dict={data: x_val_data, label: y_val_a})
            valid_loss += iter_loss
            valid_accuracy += iter_accuracy
            iters_one_epoch_valid += 1
            step_valid += 1
            summary_str_val = sess.run(summary_op, feed_dict={data: x_val_data, label: y_val_a})
            train_writer_val.add_summary(summary_str_val, step_valid)

            print("valid_iter-%d: loss: %f, acc: %f" % (iters_one_epoch_valid, valid_loss, valid_accuracy))

        print("epoch: %d, learning_rate: %f -- train loss: %f, train acc: %f, valid loss: %f, valid acc: %f"
              % (epoch + 1, lr, train_loss / iters_one_epoch_train, train_accuracy / iters_one_epoch_train,
                 valid_loss / iters_one_epoch_train, valid_accuracy / iters_one_epoch_train))

        end_time = time.time()
        print("Runtime: %.2fs" % (end_time - start_time))

        # model save (保存模型)
        if val_acc > max_acc:
            max_acc = val_acc
            saver.save(sess, os.path.join(model_file_path, model_file_name), global_step=global_step, latest_filename="best_model.ckpt")
            print("The model is saved successfully.")