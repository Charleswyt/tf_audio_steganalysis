#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import exists
import wave
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

"""
Created on 
Finished on 
@author: Wang Yuntao
"""

"""
function:
    audio_analysis(audio_file_path)                                                 音频文件解析
    get_audio_data(audio_file_path, audio_data_length=-1, is_normalization=True)    获取音频数据
"""


# def get_audio_info(auido_file_path,)


def audio_file_convert(audio_file_path, covert_formation):
    """
    audio file convert via pydub
    :param audio_file_path: the path of original audio
    :param covert_formation: the formation needed to be coverted
    :return:
    """
    audio = AudioSegment.from_wav(audio_file_path)






def audio_analysis(audio_file_path):
    """
    音频文件解析
    :param audio_file_path: 音频文件路径
    :return: audio_info: 频文件信息(字典变量)
    键值信息:
        audio_name, channel, sampwidth, sample_rate, total_samples, time
    """
    if os.path.exists(audio_file_path):
        audio = wave.open(audio_file_path)
        params = audio.getparams()
        audio_info = {}
        # print(audio.getparams())
        # _wave_params(nchannels=2, sampwidth=2, framerate=44100, nframes=441000, comptype='NONE', compname='not compressed')
        audio_info["audio_name"] = get_file_name(audio_file_path)
        audio_info["channel"] = params[0]
        audio_info["sampwidth"] = params[1] * 8
        audio_info["sample_rate"] = params[2]
        audio_info["total_samples"] = params[3]
        audio_info["time"] = params[3] / params[2]
        # 将字典变量按照键值排序
        audio_info = sorted(audio_info.items(), key=lambda k: k[0])
    else:
        audio_info = None
    return audio_info


def get_audio_data(audio_file_path, audio_data_begin=0, audio_data_end=-1,
                   is_normalization=True, channel=None):
    """
    获取音频数据
    :param audio_file_path: 音频文件路径
    :param audio_data_begin: 待获取的音频文件的起始位置
    :param audio_data_end: 待获取的音频文件的位置
    :param is_normalization: 是否对音频数据幅值进行归一化(默认为True)
    :param channel: 所有声道(Left, Right, None)
    :return: audio_data: 音频数据
    """
    if exists(audio_file_path):
        audio = wave.open(audio_file_path)
        params = audio.getparams()
        nchannles, sampwidth, framerate, nframes = params[:4]
        audio_data_string = audio.readframes(nframes)                   # 读取音频数据(str格式)
        audio_data = np.fromstring(audio_data_string, np.int16)         # 将str格式转换为int16格式
        audio.close()

        if is_normalization is True:
            audio_data = audio_data * 1.0 / (max(abs(audio_data)))
        
        audio_data = np.reshape(audio_data, [nframes, nchannles])
        audio_data = audio_data[audio_data_begin:audio_data_end,:]
        total_samples = np.shape(audio_data)[0]

        if channel == "Left":
            audio_data = audio_data[:,1]
        elif channel == "Right":
            audio_data = audio_data[:,2]
        else:
            pass
        time = np.arange(0, total_samples) * (1 / framerate)
        print(total_samples)
        print(time)
    else:
        audio_data = None

    return audio_data, time


wav_path = "E:/Myself/2.database/2.wav_cut/wav_10s/wav001_10s.mp3"
mp3_path = "E:/Myself/2.database/3.cover/old/128/cover001_b_128.mp3"
# audio_data, time = get_audio_data(mp3_path)
# print("time: ", time)
# plt.plot(time, audio_data)
# plt.xlabel('Time(s)')
# plt.ylabel('Amplitude')
# plt.title('Single Channel WaveData')
# plt.grid('on')
# plt.show()

audiofile = eyed3.load(mp3_path)
print(audiofile)
