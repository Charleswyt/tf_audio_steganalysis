#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2018.02.07
Finished on 2018.03.08
@author: Wang Yuntao
"""

import os
import tensorflow as tf

"""
Example:
gm = GPUManager()
with gm.auto_choice():
    balabala
"""


def check_gpus():
    """
    GPU available check
    reference : http://feisky.xyz/machine-learning/tensorflow/gpu_list.html
    """
    # ============================================================================================== #
    #     all_gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]     #
    # ============================================================================================== #
    system = platform.system()
    if system == "Linux":
        first_gpus = os.popen("nvidia-smi --query-gpu=index --format=csv,noheader").readlines()[0].strip()
        if not first_gpus == "0":
            print("This script could only be used to manage NVIDIA GPUs,but no GPU found in your device")
            return False
        elif "NVIDIA System Management" not in os.popen("nvidia-smi -h").read():
            print("nvidia-smi tool not found.")
            return False
    else:
        print("The current system is not Linux.")
        return False

    return True


if check_gpus():
    def parse(line, query_args):
        """
        line:
            a line of text
        query_args:
            query arguments
        return:
            a dict of gpu info
        Parsing a line of csv format text returned by nvidia-smi
        解析一行nvidia-smi返回的csv格式文本
        """
        countable_args = ["memory.free", "memory.total", "power.draw", "power.limit"]               # 可计数的参数
        power_manage_enable = lambda v: ("Not Support" not in v)                                    # 显卡是否支持power management（笔记本可能不支持）
        to_countable = lambda v: float(v.upper().strip().replace("MIB", "").replace("W", ""))       # 带单位字符串去掉单位
        process = lambda k, v: ((int(to_countable(v)) if power_manage_enable(v) else 1) if k in countable_args else v.strip())

        return {k: process(k, v) for k, v in zip(query_args, line.strip().split(","))}

    def query_gpu(query_args=None):
        """
        query_args:
            query arguments
        return:
            a list of dict
        Querying GPUs info
        查询GPU信息
        """
        if query_args is None:
            query_args = []
        query_args = ["index", "gpu_name", "memory.free", "memory.total", "power.draw", "power.limit"] + query_args
        cmd = "nvidia-smi --query-gpu={} --format=csv,noheader".format(",".join(query_args))
        results = os.popen(cmd).readlines()
        return [parse(line, query_args) for line in results]

    def by_power(d):
        """
        helper function for sorting gpus by power
        """
        power_info = (d["power.draw"], d["power.limit"])
        if any(v == 1 for v in power_info):
            print("Power management unable for GPU {}".format(d["index"]))
            return 1

        return float(d["power.draw"]) / d["power.limit"]

    class GPUManager:
        """
        query_args:
            query arguments
        A manager which can list all available GPU devices
        and sort them and choice the most free one.Unspecified
        ones pref.
        GPU设备管理器, 考虑列举出所有可用GPU设备, 并加以排序，自动选出
        最空闲的设备. 在一个GPUManager对象内会记录每个GPU是否已被指定,
        优先选择未指定的GPU.
        """

        def __init__(self, query_args=None):
            if query_args is None:
                query_args = []
            self.query_args = query_args
            self.gpus = query_gpu(query_args)
            for gpu in self.gpus:
                gpu["specified"] = False
            self.gpu_num = len(self.gpus)

        @staticmethod
        def _sort_by_memory(gpus, by_size=False):
            if by_size:
                print("Sorted by free memory size")
                return sorted(gpus, key=lambda d: d["memory.free"], reverse=True)
            else:
                print("Sorted by free memory rate")
                return sorted(gpus, key=lambda d: float(d["memory.free"]) / d["memory.total"], reverse=True)

        @staticmethod
        def _sort_by_power(gpus):
            return sorted(gpus, key=by_power)

        @staticmethod
        def _sort_by_custom(gpus, key, reverse=False, query_args=None):
            if query_args is None:
                query_args = []
            if isinstance(key, str) and (key in query_args):
                return sorted(gpus, key=lambda d: d[key], reverse=reverse)
            if isinstance(key, type(lambda a: a)):
                return sorted(gpus, key=key, reverse=reverse)
            raise ValueError("The argument 'key' must be a function or a key in query args,please read the documentation of nvidia-smi")

        def auto_choice(self, mode=0):
            """
            mode:
                0:(default)sorted by free memory size
            return:
                a TF device object
            Auto choice the freest GPU device,not specified
            ones
            自动选择最空闲GPU
            """
            for old_info, new_info in zip(self.gpus, query_gpu(self.query_args)):
                old_info.update(new_info)
            unspecified_gpus = [gpu for gpu in self.gpus if not gpu["specified"]] or self.gpus

            if mode == 0:
                print("Choosing the GPU device has largest free memory...")
                chosen_gpu = self._sort_by_memory(unspecified_gpus, True)[0]
            elif mode == 1:
                print("Choosing the GPU device has highest free memory rate...")
                chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
            elif mode == 2:
                print("Choosing the GPU device by power...")
                chosen_gpu = self._sort_by_power(unspecified_gpus)[0]
            else:
                print("Given an unavailable mode,will be chosen by memory")
                chosen_gpu = self._sort_by_memory(unspecified_gpus)[0]
            chosen_gpu["specified"] = True
            index = chosen_gpu["index"]
            print("Using GPU {i}:\n{info}".format(i=index, info="\n".join([str(k) + ":" + str(v) for k, v in chosen_gpu.items()])))
            return tf.device("/gpu:{}".format(index))
else:
    raise ImportError("GPU available check failed.")


if __name__ == "__main__":
    gm = GPUManager()
    with gm.auto_choice():

        for i in range(20):
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(global_step)
            print("%d s" % i)
