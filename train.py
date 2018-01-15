#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from model import *
import utils

"""
Created on 2017.11.27
Finished on 2017.11.30
@author: Wang Yuntao
"""

N_CLASSES = 2
BATCH_SIZE = 128
CAPACITY = 256
MAX_STEP = 1000
TRAIN = True
learning_rate = 0.05

image_width = 32
image_height = 32
image_depth = 3
image_pixel = image_depth * image_width * image_height
label_bytes = 1
image_bytes = image_pixel


# def run_training():
#     train_dir = "data/cifar-10/cifar-10-batches-bin/"
#     logs_train_dir = "logs/cifar-10/train/"
#     train, train_label = utils.get_file(train_dir)
#     train_batch, train_label_batch = utils.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#     train_logits = cats_vs_dogs_network(train_batch, BATCH_SIZE, N_CLASSES)
#     train_loss = loss(train_logits, train_label_batch)
#     train_op = optimizer(train_loss, learning_rate)
#     train_acc = accuracy(train_logits, train_label_batch)
#
#     summary_op = tf.summary.merge_all()
#     sess = tf.Session()
#     train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
#     saver = tf.train.Saver()
#
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     try:
#         for step in np.arange(MAX_STEP):
#             if coord .should_stop():
#                 break
#             _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
#
#             if step % 50 == 0:
#                 print("Step: %d, loss: %.4f" % (step, tra_loss))
#
#             if step % 100 == 0:
#                 print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (step, tra_loss, tra_acc))
#                 summary_str = sess.run(summary_op)
#                 train_writer.add_summary(summary_str, step)
#
#             if step % 2000 == 0:
#                 checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
#                 saver.save(sess, checkpoint_path, global_step=step)
#     except tf.errors.OutOfRangeError:
#         print("Done trainning -- epoch limit reached.")
#     finally:
#         coord.request_stop()
#
#     coord.join(threads)
#     sess.close()
#
#
# def evaluate():
#     train_dir = "data/cats_vs_dogs/train/"
#     train, train_label = utils.get_file(train_dir)
#     image_array = utils.get_one_image(train)
#
#     with tf.Graph().as_default():
#         BATCH_SIZE = 1
#         N_CLASSES = 2
#         image = tf.cast(image_array, tf.float32)
#         image = tf.reshape(image, [1, 208, 208, 3])
#         logit = cats_vs_dogs_network(image, BATCH_SIZE, N_CLASSES)
#
#         logit = tf.nn.softmax(logit)
#         x = tf.nn.softmax(logit)
#         x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#         logs_trian_dir = "data/cats_vs_dogs/train/"
#
#         saver = tf.train.Saver()
#
#         with tf.Session() as sess:
#             print("Reading Checkpoints...")
#             ckpt = tf.train.get_checkpoint_state(logs_trian_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split("/")[-1].splir(".")[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print("Loading success, global step is %s" % global_step)
#             else:
#                 print("No checkpoint file found.")
#
#             prediction = sess.run(logit, feed_dict={x: image_array})
#             max_index = np.argmax(prediction)
#             if max_index == 0:
#                 print("This is a cat with possibility %.6f" % prediction[:,0])
#             else:
#                 print("This is a dog with possibility %.6f" % prediction[:, 1])


def train_steganalysis():
    pass


def test_steganalysis_batch(model_dir, files_dir):
   pass


def test_steganalysis(model_dir, file_path, file_label):
    # 加载模型
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    model_file = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, model_file)

    # 读取文件
    matrix = read
