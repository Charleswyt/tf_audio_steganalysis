#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

"""
Created on 2018.03.05
Finished on 208.03.10
@author: Wang Yuntao
"""

"""
function:
    audio_read(audio_file_path, sampling_rate=44100, to_mono=False)         read audio file from disk

"""


def audio_read(audio_file_path, sampling_rate=44100, to_mono=False):
    audio = librosa.load(audio_file_path, sr=sampling_rate)
    if to_mono is True:
        audio = librosa.to_mono(audio)

    return audio


if __name__ == "__main__":
    audio_read("")