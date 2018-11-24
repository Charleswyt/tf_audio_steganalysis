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

"""
function:
    audio_read(audio_file_path, sampling_rate=44100, to_mono=False)             read audio file from disk
    audio_read_batch(audio_file_path, sampling_rate=44100, to_mono=False)       read audio files in batch
"""


def audio_read(audio_file_path, sampling_rate=44100, mono=False, offset=0, duration=None):
    """
    read audio data
    :param audio_file_path: audio file path
    :param sampling_rate: sampling rate
    :param mono: whether convert signal to mono, default is False
    :param offset: start reading after this time (in seconds), default is 0
    :param duration: only load up to this much audio (in seconds), default is None
    :return:
        2-D Variable, audio data in np.float32 format
    """
    audio_tuple = librosa.load(audio_file_path, sr=sampling_rate, mono=mono, offset=offset, duration=duration)
    audio_samples = len(audio_tuple[0])
    audio = np.zeros(shape=[2, audio_samples], dtype=np.float32)
    audio[0] = audio_tuple[0]
    audio[1] = audio_tuple[1]

    return audio


def audio_read_batch(audio_files_list, sampling_rate=44100, mono=False, offset=0, duration=None):
    """
    read audio data in batch
    :param audio_files_list: audio files list
    :param sampling_rate: sampling rate
    :param mono: whether convert signal to mono, default is False
    :param offset: start reading after this time (in seconds), default is 0
    :param duration: only load up to this much audio (in seconds), default is None
    :return:
        data, a 3-D ndarray, [batch_size, height, width]
    """
    files_num = len(audio_files_list)

    audio = audio_read(audio_files_list[0], sampling_rate=sampling_rate, mono=mono, offset=offset, duration=duration)
    channel, samples = np.shape(audio)[0], np.shape(audio)[1]

    data = np.zeros([files_num, channel, samples], dtype=np.float32)

    i = 0
    for audio_file in audio_files_list:
        content = audio_read(audio_file, sampling_rate=sampling_rate, mono=mono, offset=offset, duration=duration)
        data[i] = content
        i = i + 1

    return data


def get_mfcc_statistics(audio_data, sampling_rate=44100, n_mfcc=40):
    """
    calculate the statistics of mfcc coefficients
    :param audio_data: audio data, ndarray [length, channel]
    :param sampling_rate: sampling rate of audio data, default: 44100
    :param n_mfcc: number of mfcc, default: 40
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
