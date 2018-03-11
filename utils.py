#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.11.20
Finished on 2017.11.20
@author: Wang Yuntao
"""
from image_preprocess import *
from PIL import Image
from matplotlib.pylab import plt
from text_preprocess import *

"""
function:

"""


def get_file(file_dir, is_shuffle=True):
    """
    :param file_dir: 文件目录
    :param is_shuffle: 是否shuffle
    :return: 文件路径与其对应标签
    """
    image_cats = []
    image_dogs = []
    label_cats = []
    label_dogs = []

    for file in os.listdir(file_dir):
        file_name = file.split(sep='.')
        if file_name[0] == "cat":
            image_cats.append(file_dir + file)
            label_cats.append(0)
        if file_name[0] == "dog":
            image_dogs.append(file_dir + file)
            label_dogs.append(1)

    image_list = np.hstack((image_cats, image_dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()

    if is_shuffle is True:
        np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_w, image_h, batch_size, capacity, is_standardization=True, is_shuffle=True):
    """
    :param image: 图像数据
    :param label: 图像标签
    :param image_w: 图像宽度
    :param image_h: 图像高度
    :param batch_size: 批次大小
    :param capacity: 队列最大容纳的图像数量
    :param is_standardization: 是否进行标准化
    :param is_shuffle: 是否shuffle
    :return: 
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype = tf.int32
    """
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成数据队列
    input_queue = tf.train.slice_input_producer([image, label])
    print("input_queue:", input_queue)
    print(input_queue)

    label = input_queue[1]
    image = tf_imread(input_queue[0], image_format="jpg")

    ########################################
    # data argumentation should go to here #
    ########################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
    if is_standardization is True:
        image = tf.image.per_image_standardization(image)

    # 输入的image label数据已shuffle
    if is_shuffle is False:
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)

    # 输入的image label数据未shuffle
    else:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          num_threads=64,
                                                          capacity=capacity,
                                                          min_after_dequeue=capacity-1)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


def get_one_image(image_list):
    """
    从图像列表中获取一张图像用于检测
    :param image_list: 图像列表
    :return: image
    """
    total_num = len(image_list)
    index = np.random.randint(0, total_num)
    image_path = image_list[index]

    image = Image.open(image_path)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    plt.show()

    return image
            

def make_tfrecord(files_path, tfrecord_file_path="steganalysis.tfrecords"):
    """
    :param: files_path: 文件路径(cover, stego)
    :param: tfrecord_file_path: 生成的tfrecord文件路径
    """
    classes = {'cover', 'stego'}                                            # cover stego两类
    writer = tf.python_io.TFRecordWriter(tfrecord_file_path)                # 要生成的文件
 
    for index, name in enumerate(classes):
        class_path = files_path + name + '/'
        for text_name in os.listdir(class_path):
            text_path = class_path + text_name                              # 每一个txt文本的地址

            data = read_text(text_path)

            data_raw = data.tobytes()                                       # 将图片转化为二进制格式

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
            }))                                                             # example对象对label和text_data数据进行封装
    
            writer.write(example.SerializeToString())                       # 序列化为字符串

    writer.close()


def read_tfrecord(tfrecord_file_path):
    filename_queue = tf.train.string_input_producer([tfrecord_file_path])   # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)                     # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={'label': tf.FixedLenFeature([], tf.int64),
                                                 'data_raw': tf.FixedLenFeature([], tf.string),
                                                 })                         # 将image数据和label取出来

    data = tf.decode_raw(features['data_raw'], tf.float32)
    print(data)
    data = tf.cast(data, tf.float32)
    label = tf.cast(features['label'], tf.int32)                            # 在流中抛出label张量

    return data, label


def tfrecord_export(files_path, tfrecord_file_path):
    """
    将tfrecord文件内的数据导出
    :param files_path: 导出文件路径
    :param tfrecord_file_path: tfrecord文件路径
    :return: 
    """
    filename_queue = tf.train.string_input_producer([tfrecord_file_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)                 # return file and file_name
    features = tf.parse_single_example(serialized_example,
                                       features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    data = tf.decode_raw(features['data_raw'], tf.float32)
    data = tf.reshape(data, [200, 576, 1])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            example, l = sess.run([data, label])                    # take out image and label
            img = Image.fromarray(example, 'RGB')
            img.save(files_path + str(i) + '_''Label_' + str(l) + '.jpg')          # save image
            print(example, l)
        coord.request_stop()
        coord.join(threads)


def get_files_list(file_dir, start_idx=0, end_idx=-1):
    filename = os.listdir(file_dir)
    file_list = [file_dir + "/" + file for file in filename]
    total_num = len(file_list)
    if start_idx > total_num:
        start_idx = 0
    if end_idx > total_num:
        end_idx = -1

    file_list = file_list[start_idx:end_idx]

    return file_list


def read_data(cover_files_path, stego_files_path, start_idx, end_idx, is_shuffle=True):
    """
    读取数据(当前数据为文件名)
    :param cover_files_path: cover文件路径
    :param stego_files_path: stego文件路径
    :param start_idx: 起始文件下标
    :param end_idx: 终止文件下标
    :param is_shuffle: 是否乱序
    :return:
        data_list: 数据列表
        label_list: 标签列表
    """
    cover_files_list = get_files_list(cover_files_path)                     # cover文件列表
    stego_files_list = get_files_list(stego_files_path)                     # stego文件列表
    sample_num = len(cover_files_list)                                      # 样本对数
    cover_label_list = np.zeros(sample_num, np.int32)                       # cover标签列表
    stego_label_list = np.ones(sample_num, np.int32)                        # stego标签列表
    
    temp_cover = np.array([cover_files_list, cover_label_list])
    temp_cover = temp_cover.transpose()

    temp_stego = np.array([stego_files_list, stego_label_list])
    temp_stego = temp_stego.transpose()

    if is_shuffle is True:
        np.random.shuffle(temp_cover)
        np.random.shuffle(temp_stego)

    if start_idx > sample_num:
        start_idx = 0
    if end_idx > sample_num:
        end_idx = -1

    cover_data_list = list(temp_cover[start_idx:end_idx, 0])
    stego_data_list = list(temp_stego[start_idx:end_idx, 0])
    cover_label_list = list(temp_cover[start_idx:end_idx, 1])
    stego_label_list = list(temp_stego[start_idx:end_idx, 1])

    return cover_data_list, cover_label_list, stego_data_list, stego_label_list


# 定义一个函数，按批次取数据
def minibatches(cover_datas=None, cover_labels=None, stego_datas=None, stego_labels=None, batchsize=None):
    """
    批次读取数据
    :param cover_datas: data list (cover)
    :param cover_labels: data label (cover)
    :param stego_datas: data list (stego)
    :param stego_labels: data label (stego)
    :param batchsize: batch size
    :return:
    """
    for start_idx in range(0, len(cover_datas) - batchsize // 2 + 1, batchsize // 2):
        excerpt = slice(start_idx, start_idx + batchsize // 2)
        datas = cover_datas[excerpt]
        datas.extend(stego_datas[excerpt])
        labels = cover_labels[excerpt]
        labels.extend(stego_labels[excerpt])
        yield datas, labels


def model_load(model_file):
    if not os.path.exists(model_file):
        print("There is no such file, try again please.")
    else:
        model = np.load(model_file, encoding="latin1").item()
        model_keys = model.keys()

        for k in model.keys():
            print(k)


if __name__ == "__main__":
    model_load("E:/Myself/1.source_code/tf_audio_steganalysis/vgg19.npy")
