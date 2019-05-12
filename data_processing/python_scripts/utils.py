import os


def get_file_name(file_path, sep="/"):
    """
    get the name of file
    :param file_path: file path
    :param sep: separator
    :return: file name
    """
    if os.path.exists(file_path):
        file_path.replace("\\", "/")
        file_name = file_path.split(sep=sep)[-1]
    else:
        file_name = None
    return file_name



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


def get_files_list(file_dir, file_type="txt", start_idx=None, end_idx=None):
    """
    :param files_path: path of MP3 files for move
    :param file_type: file type, default is "txt"
    :return: Null
    """
    filename = os.listdir(file_dir)
    files_list = []
    for file in filename:
        file_path = fullfile(file_dir, file)
        if get_file_type(file_path) == file_type:
            files_list.append(file_path)
    
    files_list = files_list[start_idx:end_idx]

    return files_list