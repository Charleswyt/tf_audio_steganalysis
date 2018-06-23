#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import numpy as np
import matplotlib.pyplot as plt

"""
Created on 2018.03.05
Finished on 2018.03.10
@author: Wang Yuntao
"""

"""
function:
    audio_read(audio_file_path, sampling_rate=44100, to_mono=False)         read audio file from disk

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
        audio data in np.float32 format
    """
    audio = librosa.load(audio_file_path, sr=sampling_rate, mono=mono, offset=offset, duration=duration)

    return audio


if __name__ == "__main__":
    audio_read("E:/Myself/1.source_code/audio_steganalysis/module/test/wav10s_00001_stego_128.mp3", 44100)
