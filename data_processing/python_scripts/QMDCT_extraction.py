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

def QMDCT_extract(files_path, file_num=None, file_type="mp3"):
    """
    :param files_path: path of MP3 files for QMDCT extraction
    :param file_num: number of MP3 files for QMDCT extraction, default is None which means all files in the current path are extracted
    :param file_type: file type, default is "mp3"
    :return: Null
    """
    list = os.listdir(files_path)
    total_num = len(list)

    if file_num is None or file_num > total_num:
        file_num = total_num

    for i in range(0, file_num):
        mp3_file = os.path.join(files_path, list[i])

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
                        " -framenum 50 -startind 0 -coeffnum 576 --decode"
                    os.system(command)
                    os.remove(wav_file_path)
            else:
                pass


if __name__ == "__main__":
    params_num = len(sys.argv)
    if params_num == 2:
        files_path = sys.argv[1]
        QMDCT_extract(files_path)
    elif params_num == 3:
        files_path = sys.argv[1]
        file_num = int(sys.argv[2])
        QMDCT_extract(files_path, file_num)
    else:
        print("Please input the command as the format of {python QMDCT_extractor.py \"files_path\" \"file_num (defalut is None)\"} ")
