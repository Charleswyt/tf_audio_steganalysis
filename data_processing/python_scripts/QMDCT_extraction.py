#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2018.01.09
Finished on 2018.01.09
@author: Wang Yuntao
"""

import os
import sys

sys.path.append("tools")


def fullfile(file_dir, file_name):
    """
    fullfile as matlab
    :param file_dir: file dir
    :param file_name: file name
    :return: a full file path
    """
    full_file_path = os.path.join(file_dir, file_name)
    full_file_path = full_file_path.replace("\\", "/")

    return full_file_path
    

def get_file_type(file_path, sep="."):
    """
    get the type of file
    :param file_path: file path
    :param sep: separator
    :return: file type
    """
    if os.path.exists(file_path):
        file_type = file_path.split(sep=sep)[-1]
    else:
        file_type = None

    return file_type


def get_files_list(files_path, file_type=None):
    """
    :param files_path: path of MP3 files for move
    :param file_type: file type, default is None
    :return: Null
    """
    filename = os.listdir(files_path)
    files_list = []
    for file in filename:
        file_path = fullfile(files_path, file)
        if get_file_type(file_path) == file_type or file_type is None:
            files_list.append(file_path)

    return files_list


def qmdct_extract(files_path, frame_number=50, file_num=None, file_type="mp3"):
    """
    :param files_path: path of MP3 files for QMDCT extraction
    :param frame_number: number of frames for extraction, default is 50
    :param file_num: number of MP3 files for QMDCT extraction, default is None which means all files in the current path are extracted, default is None
    :param file_type: file type, default is "mp3"
    :return: Null
    """

    files_list = get_files_list(files_path, file_type=file_type)
    total_num = len(files_list)

    if file_num is None or file_num > total_num:
        file_num = total_num

    for i in range(0, file_num):
        mp3_file = os.path.join(files_path, files_list[i])

        if os.path.isfile(mp3_file):

            mp3_file_name = mp3_file.split("/")[-1]
            file_type = mp3_file_name.split(".")[-1]
            mp3_file_name = mp3_file_name.split(".mp3")[0]
            if file_type == "mp3":
                wav_file_name = mp3_file_name + ".wav"
                wav_file_path = os.path.join(files_path, wav_file_name)

                txt_file_name = mp3_file_name + ".txt"
                txt_file_path = os.path.join(files_path, txt_file_name)

                if os.path.exists(txt_file_path):
                    pass
                else:
                    command = "lame_qmdct.exe " + mp3_file + \
                        " -framenum " + str(frame_number) + " -startind 0 -coeffnum 576 --decode"
                    os.system(command)
                    os.remove(wav_file_path)
            else:
                pass


if __name__ == "__main__":
    params_num = len(sys.argv)
    if params_num == 2:
        args_files_path = sys.argv[1]
        qmdct_extract(args_files_path)
    elif params_num == 3:
        args_files_path = sys.argv[1]
        args_frame_number = sys.argv[2]
        qmdct_extract(args_files_path, args_frame_number)
    elif params_num == 4:
        args_files_path = sys.argv[1]
        args_frame_number = sys.argv[2]
        args_file_num = int(sys.argv[3])
        qmdct_extract(args_files_path, args_frame_number, args_file_num)
    else:
        print("Please input the command as the format of {python QMDCT_extractor.py \"files_path\" \"file_num (default is None)\"} ")
