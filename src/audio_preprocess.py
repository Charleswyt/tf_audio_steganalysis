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
