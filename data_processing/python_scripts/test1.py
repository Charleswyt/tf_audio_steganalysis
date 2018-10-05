import random
import sys
import os

sys.path.append("../../src")
sys.path.append("../secret_message/")

def message_random(secret_message_path, message_length=None):
    """
    generate temporary secret message file
    :param secret_message_path: path of origianl full secret message file
    :param message_length: length of secret message
    :return:
        temp_file_path: path of temporary secret message file
    """

    with open(secret_message_path, "r", encoding="utf-8-sig", errors='ignore') as file:
        content = file.read()
        content_list = list(content)
        random.shuffle(content_list)
        
        if message_length is None:
            content_length = len(content)
            message_length = int(content_length / 10)

        new_content_list = content_list[:message_length]

        file_name = secret_message_path.split("/")[-1]
        temp_file_path = secret_message_path.replace(file_name, "temp.txt")

        with open(temp_file_path, "w", encoding="utf-8-sig") as temp_file:
            new_content = "".join(new_content_list)
            temp_file.write(new_content)
    
    return temp_file_path


if __name__ == "__main__":
    message_random("../secret_message/stego_info_full.txt")