import sys
import shutil
from utils import *

# global variables
cover_files_path = "E:/Myself/2.database/3.cover/cover_10s/192"
HCM_stego_files_path = "E:/Myself/2.database/4.stego/HCM/"
EECS_stego_files_path = "E:/Myself/2.database/4.stego/EECS/"
total_num = 10000


def dataset_blind_make(new_dataset_path, algorithm, bitrate, interval):
    """
    :param new_dataset_path: the path of new blind dataset
    :param algorithm: the steganographic algorithm
    :param bitrate: the bitrate of mixed MP3 audio
    :param interval: the interval of each parameters 
    """
    stego_payload_rates = ["01", "03", "05", "08", "10"]
    widths = ["2", "3", "4", "5", "6"]
    files_list = get_files_list(cover_files_path, "txt")
    if algorithm == "HCM":
        spr_index = 0
        for file_index in range(total_num):
            spr = stego_payload_rates[spr_index]
            file_name = get_file_name(files_list[file_index])
            old_file_path = fullfile(fullfile(HCM_stego_files_path, "HCM_B_" + str(bitrate) + "_ER_" + spr), file_name)
            new_file_path = fullfile(new_dataset_path, file_name)

            if (file_index + 1) % interval == 0 and spr_index < len(stego_payload_rates):
                spr_index += 1
            if (file_index + 1) % interval == 0 and spr_index == len(stego_payload_rates):
                spr_index = 0
            
            shutil.copy(old_file_path, new_file_path)
    
    if algorithm == "EECS":
        w_index = 0
        for file_index in range(total_num):
            w = widths[w_index]
            file_name = get_file_name(files_list[file_index])
            old_file_path = fullfile(fullfile(EECS_stego_files_path, "EECS_B_" + str(bitrate) + "_W_" + w + "_H_7_ER_10"), file_name)
            new_file_path = fullfile(new_dataset_path, file_name)

            if (file_index + 1) % interval == 0 and w_index < len(widths):
                w_index += 1
            if (file_index + 1) % interval == 0 and w_index == len(widths):
                w_index = 0
            
            shutil.copy(old_file_path, new_file_path)

    if algorithm == "all":
        spr_index, w_index, index = 0, 0, 0
        stego_dir = fullfile(EECS_stego_files_path, "EECS_B_" + str(bitrate) + "_W_2_H_7_ER_10")
        spr = stego_payload_rates[spr_index]

        for file_index in range(total_num):
            index = spr_index + w_index
            file_name = get_file_name(files_list[file_index])
            old_file_path = fullfile(fullfile(EECS_stego_files_path, stego_dir), file_name)
            new_file_path = fullfile(new_dataset_path, file_name)

            shutil.copy(old_file_path, new_file_path)
            
            if (file_index + 1) % interval == 0 and w_index + 1 < len(widths) and index + 1 < len(widths):
                w_index += 1
                w = widths[w_index]
                stego_dir = fullfile(EECS_stego_files_path, "EECS_B_" + str(bitrate) + "_W_" + w + "_H_7_ER_10")
            
            elif (file_index + 1) % interval == 0 and w_index + 1 == len(widths) and index + 1 == len(widths):
                stego_dir = fullfile(HCM_stego_files_path, "HCM_B_" + str(bitrate) + "_ER_" + spr)
                spr_index += 1

            elif (file_index + 1) % interval == 0 and spr_index < len(stego_payload_rates) and (len(widths) < index + 1 < len(widths) + len(stego_payload_rates)):
                spr = stego_payload_rates[spr_index]                
                stego_dir = fullfile(HCM_stego_files_path, "HCM_B_" + str(bitrate) + "_ER_" + spr)
                spr_index += 1

            elif (file_index + 1) % interval == 0 and spr_index == len(stego_payload_rates) and (index + 1 > len(widths) and index + 1 == len(widths) + len(stego_payload_rates)):
                w_index, spr_index = 0, 0
                w = widths[w_index]
                spr = stego_payload_rates[spr_index]
                stego_dir = fullfile(EECS_stego_files_path, "EECS_B_" + str(bitrate) + "_W_" + w + "_H_7_ER_10")

            else:
                pass
            

if __name__ == "__main__":
    args_params_num = len(sys.argv)
    args_algorithm = sys.argv[1]
    args_bitrate = sys.argv[2]
    args_interval = int(sys.argv[3])
    if algorithm == "HCM":
        args_new_dataset_path = "E:/Myself/2.database/blind_data/MIX_HCM_B_" + bitrate
    elif algorithm == "EECS":
        args_new_dataset_path = "E:/Myself/2.database/blind_data/MIX_EECS_B_" + bitrate
    else:
        args_new_dataset_path = "E:/Myself/2.database/blind_data/MIX_B_" + bitrate

    dataset_blind_make(args_new_dataset_path, args_algorithm, args_bitrate, args_interval)
