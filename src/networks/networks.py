#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.09.17
Finished on 2018.09.17
Modified on 2019.03.05

@author: Yuntao Wang
"""

"""
    When you design a new network, write the name into this file for a successful running.
"""

# networks for experiments
rnn_network = ["rnn_lstm", "rnn_gru", "rnn_bi_lstm"]
audio_steganalysis_ih_mmsec = ["wasdn1_1", "wasdn1_2", "wasdn1_3", "wasdn1_4", "wasdn1_5", "wasdn1_6", "wasdn1_7", "wasdn1_8", "wasdn1_9",
                               "wasdn2_1", "wasdn2_2", "wasdn2_3"]
audio_steganalysis_icassp = ["rhfcn1_1", "rhfcn1_2", "rhfcn1_3"]
audio_steganalysis_dense_net = ["dense_net_mp3", "dense_net_mp3_42", "dense_net_mp3_18", "dense_net_mp3_6"]
audio_steganalysis_dissertation = ["chap4", "chap4_1", "chap4_2", "chap4_3", "chap4_4", "chap4_5", "chap4_6", "chap4_7", "chap4_8", "chap4_9", "chap4_10",
                                   "mfcc_net", "mfcc_net1", "mfcc_net2", "mfcc_net3", "mfcc_net4",
                                   "google_net", "google_net1", "google_net2", "google_net3", "google_net4", "google_net5"]

# the proposed networks
audio_steganalysis = ["wasdn", "rhfcn"]
image_steganalysis = ["s_xu_net", "j_xu_net"]
image_classification = ["le_net", "vgg16", "vgg19"]

# list of networks
networks = []

# networks for audio steganalysis
networks.extend(audio_steganalysis)
networks.extend(audio_steganalysis_ih_mmsec)
networks.extend(audio_steganalysis_icassp)
networks.extend(audio_steganalysis_dense_net)
networks.extend(audio_steganalysis_dissertation)
networks.extend(rnn_network)

# networks for image steganalysis
networks.extend(image_steganalysis)

# networks for image classification
networks.extend(image_classification)
