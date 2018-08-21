#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.05.30
Finished on
@author: Wang Yuntao
"""

import os
from utils import fullfile, get_files_list
from file_preprocess import get_file_name


"""
function:
    cover_make_lame(wav_files_path, mp3_files_path, bitrate, start_idx=None, end_idx=None)                              make cover samples via lame encoder
    cover_make_mp3stego(wav_files_path, mp3_files_path, bitrate, start_idx=None, end_idx=None)                          make cover samples via MP3Stego encoder
    stego_make_mp3stego(wav_files_path, mp3_files_path, bitrate, embedding_rate="10", start_idx=None, end_idx=None)     make stego samples via MP3Stego encoder
    stego_make_hcm(wav_files_path, mp3_files_path, bitrate, cost="2",
                   embed=embedding_file_path, frame_num="50", embedding_rate="10", start_idx=None, end_idx=None)        make stego samples via HCM encoder
    stego_make_eecs(wav_files_path, mp3_files_path, bitrate, width, height="7",
                    embed=embedding_file_path, frame_num="50", embedding_rate="10", start_idx=None, end_idx=None)       make stego samples via EECS encoder
    stego_make_acs(wav_files_path, mp3_files_path, bitrate, width, height="7",
                   embed=embedding_file_path, embedding_rate="10", start_idx=None, end_idx=None)                        make stego samples via ACS encoder
    cover_make(wav_files_path, mp3_files_path, bitrate, cover_types="lame", start_idx=None, end_idx=None)               cover make
    
    # make samples in batch
    stego_make_mp3stego_batch(wav_files_path, mp3_files_path, start_idx=None, end_idx=None)                             make stego samples via MP3Stego encoder in batch
    stego_make_hcm_batch(wav_files_path, mp3_files_path, frame_num="50",
                         embed=embedding_file_path, start_idx=None, end_idx=None)                                       make stego samples via HCM encoder in batch
    stego_make_eecs_batch(wav_files_path, mp3_files_path, frame_num="50",
                          embed=embedding_file_path, start_idx=None, end_idx=None)                                      make stego samples via EECS encoder in batch
    stego_make_acs_batch(wav_files_path, mp3_files_path, embed=embedding_file_path, start_idx=None, end_idx=None)       make stego samples via ACS encoder in batch
    cover_make_batch(wav_files_path, mp3_files_path, start_idx=None, end_idx=None)                                      make cover samples via lame encoder in batch
    
    calibration(mp3_files_path, calibration_files_path, bitrate, start_idx=None, end_idx=None)                          calibration(recode) via lame encoder
"""

# global variable
embedding_file_path = "E:/Myself/2.database/5.stego_info/stego_info_full.txt"
embedding_files_mp3stego_path = "E:/Myself/2.database/5.stego_info/MP3Stego"


def cover_make_lame(wav_files_path, mp3_files_path, bitrate, start_idx=None, end_idx=None):
    """
    make mp3 cover samples via lame encoder
    :param wav_files_path: path of wav audio files
    :param mp3_files_path:path of mp3 audio files
    :param bitrate: bitrate (128, 192, 256, 320)
    :param start_idx: the start index of audio files to be processed
    :param end_idx: the end index of audio files to be processed
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        wav_files_list = get_files_list(file_dir=wav_files_path, start_idx=start_idx, end_idx=end_idx)
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for wav_file_path in wav_files_list:
            file_name = get_file_name(wav_file_path)
            mp3_file_name = file_name.replace(".wav", ".mp3")
            mp3_file_path = fullfile(mp3_files_path, mp3_file_name)
            if not os.path.exists(mp3_file_path):
                command = "encode.exe -b " + bitrate + " " + wav_file_path + " " + mp3_file_path
                os.system(command)
            else:
                pass
        print("cover samples with bitrate %s are completed." % bitrate)


def cover_make_mp3stego(wav_files_path, mp3_files_path, bitrate, start_idx=None, end_idx=None):
    """
    make mp3 cover samples via mp3stego encoder
    :param wav_files_path: path of wav audio files
    :param mp3_files_path:path of mp3 audio files
    :param bitrate: bitrate (128, 192, 256, 320)
    :param start_idx: the start index of audio files to be processed
    :param end_idx: the end index of audio files to be processed
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        wav_files_list = get_files_list(file_dir=wav_files_path, start_idx=start_idx, end_idx=end_idx)
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for wav_file_path in wav_files_list:
            file_name = get_file_name(wav_file_path)
            mp3_file_name = file_name.replace(".wav", ".mp3")
            mp3_file_path = fullfile(mp3_files_path, mp3_file_name)
            if not os.path.exists(mp3_file_path):
                command = "encode_MP3Stego.exe -b " + bitrate + " " + wav_file_path + " " + mp3_file_path
                os.system(command)
            else:
                pass
        print("MP3Stego cover samples with bitrate %s are completed." % bitrate)


def stego_make_mp3stego(wav_files_path, mp3_files_path, bitrate, embedding_rate="10", start_idx=None, end_idx=None):
    """
    make stego samples via MP3Stego
    for 10s wav audio, secret messages of 1528 bits (191 Bytes) will be embedded, and the length of secret messages is independent of bitrate
    analysis unit: 50 frames (for 10s mp3 audio, there are 384 frames), 24.83 bytes messages will be embedded
    relative embedding rate         secret messages length
             10%                           3  Bytes
             30%                           8  Bytes
             50%                           13 Bytes
             80%                           20 Bytes
             100%                          24 Bytes
    in the process of MP3stego, the messages are compressed
    :param wav_files_path: path of wav audio files
    :param mp3_files_path:path of mp3 audio files
    :param bitrate: bitrate (128, 192, 256, 320)
    :param embedding_rate: embedding rate, default is "10"
    :param start_idx: the start index of audio files to be processed
    :param end_idx: the end index of audio files to be processed
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        wav_files_list = get_files_list(file_dir=wav_files_path, start_idx=start_idx, end_idx=end_idx)
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        embedding_file_name = "stego_0" + embedding_rate + ".txt" if len(embedding_rate) == 1 else "stego_" + embedding_rate + ".txt"
        embedding_file = fullfile(embedding_files_mp3stego_path, embedding_file_name)
        for wav_file_path in wav_files_list:
            file_name = get_file_name(wav_file_path)
            mp3_file_name = file_name.replace(".wav", ".mp3")
            mp3_file_path = fullfile(mp3_files_path, mp3_file_name)
            if not os.path.exists(mp3_file_path):
                command = "encode_MP3Stego.exe -b " + bitrate + " -E " + embedding_file + " -P pass " + wav_file_path + " " + mp3_file_path
                os.system(command)
            else:
                pass
        print("stego samples are made completely, bitrate %s, stego algorithm %s." % (bitrate, "MP3Stego"))


def stego_make_hcm(wav_files_path, mp3_files_path, bitrate, cost="2",
                   embed=embedding_file_path, frame_num="50", embedding_rate="10", start_idx=None, end_idx=None):
    """
    make stego samples (HCM)
    :param wav_files_path: path of wav audio files
    :param mp3_files_path: path of mp3 audio files
    :param bitrate: bitrate
    :param cost: type of cost function, default is "2"
    :param embed: path of embedding file
    :param frame_num: frame number of embedding message, default is "50"
    :param embedding_rate: embedding rate, default is "10"
    :param start_idx: start index of audio files
    :param end_idx: end index of audio files
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        wav_files_list = get_files_list(file_dir=wav_files_path, start_idx=start_idx, end_idx=end_idx)
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for wav_file_path in wav_files_list:
            file_name = get_file_name(wav_file_path)
            mp3_file_name = file_name.replace(".wav", ".mp3")
            mp3_file_path = fullfile(mp3_files_path, mp3_file_name)
            if not os.path.exists(mp3_file_path):
                command = "encode_HCM.exe -b " + bitrate + " -embed " + embed + " -cost " + cost + " -er " + embedding_rate \
                          + " -framenumber " + frame_num + " " + wav_file_path + " " + mp3_file_path
                os.system(command)
            else:
                pass


def stego_make_eecs(wav_files_path, mp3_files_path, bitrate, width, height="7",
                    embed=embedding_file_path, frame_num="50", embedding_rate="10", start_idx=None, end_idx=None):
    """
    make stego samples (EECS)
    :param wav_files_path: path of wav audio files
    :param mp3_files_path: path of mp3 audio files
    :param bitrate: bitrate
    :param width: width of parity-check matrix
    :param height: height of parity-check matrix, default is "7"
    :param embed: path of embedding file
    :param frame_num: frame number of embedding message, default is "50"
    :param embedding_rate: embedding rate, default is "10"
    :param start_idx: start index of audio files
    :param end_idx: end index of audio files
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        wav_files_list = get_files_list(file_dir=wav_files_path, start_idx=start_idx, end_idx=end_idx)
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for wav_file_path in wav_files_list:
            file_name = get_file_name(wav_file_path)
            mp3_file_name = file_name.replace(".wav", ".mp3")
            mp3_file_path = fullfile(mp3_files_path, mp3_file_name)
            if not os.path.exists(mp3_file_path):
                command = "encode_EECS.exe -b " + bitrate + " -embed " + embed + " -width " + width + " -height " + height + " -er " + embedding_rate \
                          + " -framenumber " + frame_num + " " + wav_file_path + " " + mp3_file_path
                os.system(command)
            else:
                pass


def stego_make_acs(wav_files_path, mp3_files_path, bitrate, width, height="7",
                   embed=embedding_file_path, embedding_rate="10", start_idx=None, end_idx=None):
    """
    make stego samples (ACS)
    :param wav_files_path: path of wav audio files
    :param mp3_files_path: path of mp3 audio files
    :param bitrate: bitrate
    :param width: width of parity-check matrix
    :param height: height of parity-check matrix, default is "7"
    :param embed: path of embedding file
    :param embedding_rate: embedding rate, default is "10"
    :param start_idx: start index of audio files
    :param end_idx: end index of audio files
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        wav_files_list = get_files_list(file_dir=wav_files_path, start_idx=start_idx, end_idx=end_idx)
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for wav_file_path in wav_files_list:
            file_name = get_file_name(wav_file_path)
            mp3_file_name = file_name.replace(".wav", ".mp3")
            mp3_file_path = fullfile(mp3_files_path, mp3_file_name)
            if not os.path.exists(mp3_file_path):
                command = "C:/Users/Charles_CatKing/Desktop/ACS/lame.exe -b " + bitrate + " -embed " + embed + " -width " + width + " -height " + height + \
                          " -er " + embedding_rate + " -region 2 -layerii 1 -threshold 2 -key 123456 " + wav_file_path + " " + mp3_file_path
                os.system(command)
            else:
                pass


def stego_make_mp3stego_batch(wav_files_path, mp3_files_path, start_idx=None, end_idx=None):
    """
    make stego samples (MP3Stego)
    :param wav_files_path: path of wav audio files
    :param mp3_files_path: path of mp3 audio files
    :param start_idx: start index of audio files
    :param end_idx: end index of audio files
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        stego_files_dir = fullfile(mp3_files_path, "MP3Stego")
        if not os.path.exists(stego_files_dir):
            os.mkdir(stego_files_dir)
        bitrates = ["320", "256", "192", "128"]
        embedding_rates = ["1", "3", "5", "8", "10"]
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for bitrate in bitrates:
            for embedding_rate in embedding_rates:
                folder_name_prefix = "MP3Stego_B_" + bitrate + "_ER_"
                folder_name = folder_name_prefix + "0" + embedding_rate if len(embedding_rate) == 1 else folder_name_prefix + embedding_rate
                mp3_files_sub_path = fullfile(stego_files_dir, folder_name)
                stego_make_mp3stego(wav_files_path, mp3_files_sub_path, bitrate=bitrate, embedding_rate=embedding_rate, start_idx=start_idx, end_idx=end_idx)

        print("stego samples are made completely, stego algorithm MP3Stego.")


def stego_make_hcm_batch(wav_files_path, mp3_files_path, frame_num="50",
                         embed=embedding_file_path, start_idx=None, end_idx=None):
    """
    make stego samples (HCM)
    :param wav_files_path: path of wav audio files
    :param mp3_files_path: path of mp3 audio files
    :param embed: path of embedding file
    :param frame_num: frame number of embedding message, default is "50"
    :param start_idx: start index of audio files
    :param end_idx: end index of audio files
    :return: NULL
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        stego_files_dir = fullfile(mp3_files_path, "HCM")
        if not os.path.exists(stego_files_dir):
            os.mkdir(stego_files_dir)
        bitrates = ["128", "192", "256", "320"]
        costs = ["2"]
        embedding_rates = ["1", "3", "5", "8", "10"]
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for bitrate in bitrates:
            for cost in costs:
                for embedding_rate in embedding_rates:
                    folder_name = "HCM_B_" + bitrate + "_ER_" + embedding_rate
                    mp3_files_sub_path = fullfile(stego_files_dir, folder_name)
                    stego_make_hcm(wav_files_path, mp3_files_sub_path, bitrate=bitrate, cost=cost,
                                   embed=embed, frame_num=frame_num, embedding_rate=embedding_rate, start_idx=start_idx, end_idx=end_idx)

        print("stego samples are made completely, stego algorithm HCM.")


def stego_make_eecs_batch(wav_files_path, mp3_files_path, frame_num="50",
                          embed=embedding_file_path, start_idx=None, end_idx=None):
    """
    make stego samples (EECS, batch)
    :param wav_files_path: path of wav audio files
    :param mp3_files_path: path of mp3 audio files
    :param frame_num: frame number of embedding messages
    :param embed: path of embedding file path
    :param start_idx: start index of audio files
    :param end_idx: end index of audio files
    :return:
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        stego_files_dir = fullfile(mp3_files_path, "EECS")
        if not os.path.exists(stego_files_dir):
            os.mkdir(stego_files_dir)
        bitrates = ["128", "192", "256", "320"]
        widths = ["2", "3", "4", "5", "6", "7"]
        heights = ["7"]
        embedding_rates = ["10"]
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for bitrate in bitrates:
            for width in widths:
                for height in heights:
                    for embedding_rate in embedding_rates:
                        folder_name = "EECS_B_" + bitrate + "_W_" + width + "_H_" + height + "_ER_" + embedding_rate
                        mp3_files_sub_path = fullfile(stego_files_dir, folder_name)
                        stego_make_eecs(wav_files_path, mp3_files_sub_path, bitrate=bitrate, width=width, height=height,
                                        embed=embed, frame_num=frame_num, embedding_rate=embedding_rate, start_idx=start_idx, end_idx=end_idx)

        print("stego samples are made completely, stego algorithm EECS.")


def stego_make_acs_batch(wav_files_path, mp3_files_path, embed=embedding_file_path, start_idx=None, end_idx=None):
    """
    make stego samples (ACS, batch)
    :param wav_files_path: path of wav audio files
    :param mp3_files_path: path of mp3 audio files
    :param embed: path of embedding file path
    :param start_idx: start index of audio files
    :param end_idx: end index of audio files
    :return:
    """
    if not os.path.exists(wav_files_path):
        print("The wav files path does not exist.")
    else:
        stego_files_dir = fullfile(mp3_files_path, "EECS")
        if not os.path.exists(stego_files_dir):
            os.mkdir(stego_files_dir)
        bitrates = ["128", "192", "320"]
        widths = ["4", "6", "8"]
        heights = ["7"]
        embedding_rates = ["10"]
        if not os.path.exists(mp3_files_path):
            os.mkdir(mp3_files_path)
        for bitrate in bitrates:
            for width in widths:
                for height in heights:
                    for embedding_rate in embedding_rates:
                        folder_name = "ACS_B_" + bitrate + "_W_" + width + "_H_" + height + "_ER_" + embedding_rate
                        mp3_files_sub_path = fullfile(stego_files_dir, folder_name)
                        stego_make_acs(wav_files_path, mp3_files_sub_path, bitrate=bitrate, width=width, height=height,
                                       embed=embed, embedding_rate=embedding_rate, start_idx=start_idx, end_idx=end_idx)

        print("stego samples are made completely, stego algorithm ACS.")


def cover_make(wav_files_path, mp3_files_path, bitrate, cover_types="lame", start_idx=None, end_idx=None):
    """
    make cover mp3 samples with specified cover type and bitrate
    :param wav_files_path: path of wav audio files
    :param mp3_files_path:path of mp3 audio files
    :param bitrate: bitrate
    :param cover_types: the type of cover, "lame" or "mp3stego", default is "lame"
    :param start_idx: the start index of audio files to be processed
    :param end_idx: the end index of audio files to be processed
    :return: NULL
    """
    if cover_types == "lame":
        cover_make_lame(wav_files_path, mp3_files_path, bitrate, start_idx=start_idx, end_idx=end_idx)
    elif cover_types == "mp3stego":
        cover_make_mp3stego(wav_files_path, mp3_files_path, bitrate, start_idx=start_idx, end_idx=end_idx)
    else:
        print("No cover type matches.")


def cover_make_batch(wav_files_path, mp3_files_path, start_idx=None, end_idx=None):
    """
    make cover mp3 samples
    :param wav_files_path: path of wav audio files
    :param mp3_files_path:path of mp3 audio files
    :param start_idx: the start index of audio files to be processed
    :param end_idx: the end index of audio files to be processed
    :return: NULL
        origin: cover_make(wav_files_path, mp3_files_path, bitrate, cover_types="lame", start_idx=None, end_idx=None):
    """
    bitrates = ["128", "192", "256", "320"]
    cover_types = ["lame", "mp3stego"]
    if not os.path.exists(mp3_files_path):
        os.mkdir(mp3_files_path)
    else:
        pass

    for cover_type in cover_types:
        for bitrate in bitrates:
            mp3_files_sub_path = fullfile(mp3_files_path, bitrate) if cover_type == "lame" else fullfile(mp3_files_path, "mp3stego_" + bitrate)
            cover_make(wav_files_path, mp3_files_sub_path, bitrate, cover_type, start_idx=start_idx, end_idx=end_idx)


def calibration(mp3_files_path, calibration_files_path, bitrate, start_idx=None, end_idx=None):
    """
    mp3 calibration via lame encoder  -> lame.exe -b 128 ***.mp3 c_***.mp3
    :param mp3_files_path: the mp3 files path
    :param calibration_files_path: the calibrated mp3 files path
    :param bitrate: bitrate
    :param start_idx: start index
    :param end_idx: end index
    :return:
    """
    if not os.path.exists(mp3_files_path):
        print("The mp3 files path does not exist.")
    else:
        mp3_files_list = get_files_list(file_dir=mp3_files_path, start_idx=start_idx, end_idx=end_idx)
        if not os.path.exists(calibration_files_path):
            os.mkdir(calibration_files_path)
        for mp3_file_path in mp3_files_list:
            mp3_file_name = get_file_name(mp3_file_path)
            calibrated_mp3_file_path = fullfile(calibration_files_path, mp3_file_name)
            if not os.path.exists(calibrated_mp3_file_path):
                command = "encode.exe -b " + bitrate + " " + mp3_file_path + " " + calibrated_mp3_file_path
                os.system(command)
            else:
                pass
        print("calibration with bitrate %s are completed." % bitrate)


if __name__ == "__main__":
    # wav_audio_files_path = "E:/Myself/2.database/mtap/wav"
    wav_audio_files_path = "E:/Myself/2.database/2.wav_cut/wav_10s"
    mp3_audio_cover_files_path = "E:/Myself/2.database/3.cover/cover_10s"
    mp3_audio_stego_files_path = "E:/Myself/2.database/4.stego"

    # mp3_audio_cover_files_path = fullfile(mp3_audio_files_path, "cover")
    # mp3_audio_stego_files_path = fullfile(mp3_audio_files_path, "stego")

    # cover_make_batch(wav_audio_files_path, mp3_audio_cover_files_path)
    stego_make_eecs_batch(wav_audio_files_path, mp3_audio_stego_files_path)
    # stego_make_acs_batch(wav_audio_files_path, mp3_audio_stego_files_path)
    # stego_make_mp3stego_batch(wav_audio_files_path, mp3_audio_stego_files_path)
