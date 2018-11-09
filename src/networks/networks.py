#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.09.17
Finished on 2018.09.17
Modified on 2018.11.09

@author: Yuntao Wang
"""

"""
    When you design a new network, write the name into this file for a successful running.
"""

audio_steganalysis_ih_mmsec = ["wasdn1_1", "wasdn1_2", "wasdn1_3", "wasdn1_4", "wasdn1_5", "wasdn1_6", "wasdn1_7", "wasdn1_8", "wasdn1_9",
                               "wasdn2_1", "wasdn2_2", "wasdn2_3"]
audio_steganalysis_icassp = ["rhfcn1_1", "rhfcn1_2", "rhfcn1_3"]
audio_steganalysis = ["wasdn", "rhfcn"]
rnn_network = ["rnn_lstm", "rnn_gru", "rnn_bi_lstm"]
image_steganalysis = ["s_xu_net", "j_xu_net"]
image_classification = ["le_net", "vgg16", "vgg19"]

networks = []

# audio steganalysis
networks.extend(audio_steganalysis)
networks.extend(audio_steganalysis_ih_mmsec)
networks.extend(audio_steganalysis_icassp)
networks.extend(rnn_network)

# image steganalysis
networks.extend(image_steganalysis)

# image classification
networks.extend(image_classification)
