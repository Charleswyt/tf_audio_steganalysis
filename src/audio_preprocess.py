#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import numpy as np

"""
Created on 2018.03.05
Finished on 2018.03.10
Modified on 2018.09.12

@author: Yuntao Wang
"""


def audio_read(audio_file_path, sampling_rate=44100, channel="left", offset=0, duration=None):
    """
    read audio data
    :param audio_file_path: audio file path
    :param sampling_rate: sampling rate
    :param channel: "left", "right", "both"
    :param offset: start reading after this time (in seconds), default is 0
    :param duration: only load up to this much audio (in seconds), default is None
    :return:
        2-D Variable, audio data in np.float32 format
    """

    audio_tuple = librosa.load(audio_file_path, sr=sampling_rate, mono=False, offset=offset, duration=duration)
    if channel == "both":
        audio = audio_tuple[0]
    elif channel == "left":
        audio = audio_tuple[0][0]
    elif channel == "right":
        audio = audio_tuple[0][1]
    else:
        audio = None

    return audio


def audio_read_batch(audio_files_list, sampling_rate=44100, channel="both", offset=0, duration=None):
    """
    read audio data in batch
    :param audio_files_list: audio files list
    :param sampling_rate: sampling rate
    :param channel: "left", "right", "both"
    :param offset: start reading after this time (in seconds), default is 0
    :param duration: only load up to this much audio (in seconds), default is None
    :return:
        data, a 3-D ndarray, [batch_size, height, width]
    """
    files_num = len(audio_files_list)

    audio = audio_read(audio_files_list[0], sampling_rate=sampling_rate, channel=channel, offset=offset, duration=duration)
    channels, samples = np.shape(audio)[0], np.shape(audio)[1]

    data = np.zeros([files_num, channels, samples], dtype=np.float32)

    i = 0
    for audio_file in audio_files_list:
        content = audio_read(audio_file, sampling_rate=sampling_rate, channel=channel, offset=offset, duration=duration)
        data[i] = content
        i = i + 1

    return data


def get_mfcc_statistics(audio_data, sampling_rate=44100, n_mfcc=40):
    """
    calculate the statistics of mfcc coefficients
    :param audio_data: audio data, ndarray [length, channel]
    :param sampling_rate: sampling rate of audio data, default is 44100
    :param n_mfcc: number of mfcc, default is 40
    :return:
    """
    mfcc_coefficients = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc)

    mfcc_feature = []

    mfcc_max = np.max(mfcc_coefficients.T, axis=0)
    mfcc_feature.append(mfcc_max)

    mfcc_min = np.min(mfcc_coefficients.T, axis=0)
    mfcc_feature.append(mfcc_min)

    mfcc_mean = np.mean(mfcc_coefficients.T, axis=0)
    mfcc_feature.append(mfcc_mean)

    mfcc_var = np.var(mfcc_coefficients.T, axis=0)
    mfcc_feature.append(mfcc_var)

    mfcc_var_inverse = np.divide(1.0, mfcc_var)
    mfcc_kurtosis = np.multiply(np.power(mfcc_mean, 4), np.power(mfcc_var_inverse, 4))
    mfcc_feature.append(mfcc_kurtosis)

    mfcc_skewness = np.multiply(np.power(mfcc_mean, 3), np.power(mfcc_var_inverse, 3))
    mfcc_feature.append(mfcc_skewness)

    mfcc_feature = np.array(mfcc_feature)

    return mfcc_feature


def get_mfcc(audio_file_path, sampling_rate=44100, offset=0, duration=1.3, n_mfcc=40):
    """
    extract mel frequency ceptral coefficients of audio
    :param audio_file_path: audio file path
    :param sampling_rate: sampling rate
    :param offset: start reading after this time (in seconds), default is 0
    :param duration: only load up to this much audio (in seconds), default is 1.3
    :param n_mfcc: number of mfcc, default is 24
    :return:
        MFCC vector: [frames, n_mfcc * channel_number]
    """

    audio_data = librosa.load(audio_file_path, sr=sampling_rate, mono=False, offset=offset, duration=duration)
    mfcc_left = librosa.feature.mfcc(y=audio_data[0][0], sr=sampling_rate, n_mfcc=n_mfcc)
    mfcc_right = librosa.feature.mfcc(y=audio_data[0][1], sr=sampling_rate, n_mfcc=n_mfcc)

    mfcc_dif1_left = librosa.feature.mfcc(y=np.diff(audio_data[0][0], 1, 0), sr=sampling_rate, n_mfcc=n_mfcc)
    mfcc_dif2_left = librosa.feature.mfcc(y=np.diff(audio_data[0][0], 2, 0), sr=sampling_rate, n_mfcc=n_mfcc)

    mfcc_dif1_right = librosa.feature.mfcc(y=np.diff(audio_data[0][1], 1, 0), sr=sampling_rate, n_mfcc=n_mfcc)
    mfcc_dif2_right = librosa.feature.mfcc(y=np.diff(audio_data[0][1], 2, 0), sr=sampling_rate, n_mfcc=n_mfcc)

    frames = np.shape(mfcc_left)[1]
    mfcc = np.zeros([frames, n_mfcc * 2, 3])
    mfcc[:, :n_mfcc, 0], mfcc[:, n_mfcc:, 0] = np.transpose(mfcc_left), np.transpose(mfcc_right)
    mfcc[:, :n_mfcc, 0], mfcc[:, n_mfcc:, 1] = np.transpose(mfcc_dif1_left), np.transpose(mfcc_dif1_right)
    mfcc[:, :n_mfcc, 0], mfcc[:, n_mfcc:, 2] = np.transpose(mfcc_dif2_left), np.transpose(mfcc_dif2_right)

    return mfcc


def get_mfcc_batch(audio_files_list, sampling_rate=44100, offset=0, duration=1.3, n_mfcc=40):
    """
    get mfcc vector of audio in batch
    :param audio_files_list: audio files list
    :param sampling_rate: sampling rate
    :param offset: start reading after this time (in seconds), default is 0
    :param duration: only load up to this much audio (in seconds), default is None
    :param n_mfcc: number of mfcc, default: 24
    :return:
        data, a 3-D ndarray, [batch_size, height, width]
    """
    files_num = len(audio_files_list)

    mfcc = get_mfcc(audio_files_list[0], sampling_rate=sampling_rate, offset=offset, duration=duration, n_mfcc=n_mfcc)
    height, width, channel = np.shape(mfcc)[0], np.shape(mfcc)[1], np.shape(mfcc)[2]

    data = np.zeros([files_num, height, width, channel], dtype=np.float32)

    i = 0
    for audio_file in audio_files_list:
        content = get_mfcc(audio_file, sampling_rate=sampling_rate, offset=offset, duration=duration, n_mfcc=n_mfcc)
        data[i] = content
        i = i + 1

    return data
