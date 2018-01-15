#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.16
Finished on 2017.12.19
@author: Wang Yuntao
"""

import time
from model import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 隐写参数设置 128_W_2_H_7_ER_3
stego_method = "EECS"
bitrate = 320                   # bitrates = [128, 320]
capacity = 2                    # alpha = 1 / capacity capacity = [2, 4, 5, 10] -> alpha = [0.5, 0.25, 0.2, 0.1]
embedding_rate = 10

# 路径设置
home = ["/home/zhanghong", "/home/zq"]
root_path = home[1]
data_dir = os.path.join(root_path, "data/train")
stego_dir_name = str(bitrate) + "_W_" + str(capacity) + "_H_7_ER_" + str(embedding_rate)

# EECS
cover_files_path = os.path.join(data_dir, "cover/" + str(bitrate))
stego_files_path = os.path.join(data_dir, "stego/EECS/" + stego_dir_name)

# HCM
# cover_files_path = os.path.join(data_dir, "cover/" + str(bitrate))
# stego_files_path = os.path.join(data_dir, "stego/HCM/128_01")

# sign_bit
# cover_files_path = "steganalysis/128/cover"
# stego_files_path = "steganalysis/128/stego"

model_dir = "models/steganalysis/" + stego_dir_name
model_name = "steganalysis_model.npz"
logs_train_dir = os.path.join("logs/myself/train/", stego_dir_name)
logs_val_dir = os.path.join("logs/myself/train/", stego_dir_name)

# 超参设置
batch_size = 100                                                            # batch_size
init_learning_rate = 1e-4                                                   # learning_rate
n_epoch = 500                                                               # epoch
classes_num = 2                                                             # classes_num
ratio = 0.5                                                                 # 训练集所占比例
start_index, end_index = 0, 20000                                           # 实际用于训练和验证的数据集规模

files_list = get_files_list(cover_files_path, start_index, end_index)       # 文件列表(用于计算iteration)
files_num = len(files_list)                                                 # 总文件数量
train_files_num = files_num * ratio                                         # 训练集文件数量
validation_files_num = files_num - train_files_num                          # 验证集文件数量
iters = train_files_num // batch_size                                       # 每个epoch的iters

# 预处理参数
is_abs = False
is_trunc = False
is_diff = False
is_downsampling = False
threshold, stride = 3, 2

# 信息输出
print("cover file path: %s" % cover_files_path)
print("stego file path: %s" % stego_files_path)
print("param: stego_method: %s, bitrate: %d kbs, W: %d, H: 7, ER: %d" % (stego_method, bitrate, capacity, embedding_rate))
print("Pre-Process method: abs:", is_abs, "trunc:", is_trunc, "diff:", is_diff, "downsampling:", is_downsampling)
if is_trunc is True:
    print("threshold = %d" % threshold)
if is_downsampling is True:
    print("stride = %d" % stride)
print("cover/stego sample pairs: %d, train_set pairs: %d, validation_set pairs:%d, every epoch includes %d iters"
      % (files_num, train_files_num, validation_files_num, 2 * iters))
print("batch_size: %d, total_epoch: %d, class_num: %d" % (batch_size, n_epoch, classes_num))
print("start load network...")
time.sleep(3)

# 占位符
height = 200                                                                # data matrix height
width = 576                                                                 # data matrix width
channel = 1                                                                 # data matrix channel

x = tf.placeholder(tf.float32, [batch_size, height-2, width, channel], name="QMDCTs")
y_ = tf.placeholder(tf.int32, [batch_size, ], name="label")

# 网络结构
logits = network1_5(x, 2)
time.sleep(3)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=init_learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", acc)
summary_op = tf.summary.merge_all()
sess = tf.InteractiveSession()
train_writer_train = tf.summary.FileWriter(logs_train_dir + "/train", sess.graph)
train_writer_val = tf.summary.FileWriter(logs_val_dir + "/validation")
init = tf.global_variables_initializer()
sess.run(init)

# 保存模型
saver = tf.train.Saver(max_to_keep=3)
max_acc = 0

if os.path.exists("models/steganalysis") is False:
    os.mkdir("models/steganalysis")
if os.path.exists(model_dir) is False:
    os.mkdir(model_dir)

step_train = 0
step_test = 0
print("Start training...")
for epoch in range(n_epoch):
    start_time = time.time()

    data_list, label_list = read_data(cover_files_path, stego_files_path, start_index, end_index)               # 读取文件列表(默认shuffle)
    x_train, x_val, y_train, y_val = make_train_set(data_list, label_list, ratio=ratio)                         # 制备训练集和验证集

    # 学习率设置
    weight = [1, 0.1, 0.05, 0.01, 0.001]
    epochs = [0, 20, 50, 100, 300]
    if epochs[0] <= epoch < epochs[1]:
        learning_rate = weight[0] * init_learning_rate
    elif epochs[1] <= epoch < epochs[2]:
        learning_rate = weight[1] * init_learning_rate
    elif epochs[2] <= epoch < epochs[3]:
        learning_rate = weight[2] * init_learning_rate
    elif epochs[3] <= epoch < epochs[4]:
        learning_rate = weight[3] * init_learning_rate
    else:
        learning_rate = weight[4] * init_learning_rate

    # training
    n_batch_train, train_loss, train_acc = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size):
        # 数据读取与处理
        x_train_data = read_text_batch(x_train_a, height, width,
                                       is_abs=is_abs, is_trunc=is_trunc, threshold=threshold, is_downsampling=is_downsampling, stride=stride)

        # 训练与指标显示
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_data, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch_train += 1
        step_train += 1
        summary_str_train = sess.run(summary_op, feed_dict={x: x_train_data, y_: y_train_a})
        if step_train > 200:
            train_writer_train.add_summary(summary_str_train, step_train)

        print("train_iter-%d: train_loss: %f, train_acc: %f" % (n_batch_train, err, ac))

    # validation
    n_batch_val, val_loss, val_acc = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size):
        # 数据读取与处理
        x_val_data = read_text_batch(x_val_a, height, width,
                                     is_abs=is_abs, is_trunc=is_trunc, threshold=threshold, is_downsampling=is_downsampling, stride=stride)

        # 验证与指标显示
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_data, y_: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch_val += 1
        step_test += 1
        summary_str_val = sess.run(summary_op, feed_dict={x: x_val_data, y_: y_val_a})
        if step_train > 200:
            train_writer_val.add_summary(summary_str_val, step_test)
        print("validation_iter-%d: loss: %f, acc: %f" % (n_batch_val, err, ac))

    print("epoch: %d, learning_rate: %f -- train loss: %f, train acc: %f, validation loss: %f, validation acc: %f"
          % (epoch + 1, learning_rate, train_loss / n_batch_train, train_acc / n_batch_train,
             val_loss / n_batch_val, val_acc / n_batch_val))

    end_time = time.time()
    print("Runtime: %.2fs" % (end_time - start_time))

    # 保存模型
    if val_acc > max_acc:
        max_acc = val_acc
        saver.save(sess, os.path.join(model_dir, model_name), global_step=epoch+1)
        print("模型保存成功")

train_writer.close()
sess.close()

print("End")
