# -*- coding: utf-8 -*-

# global libraries
import sys
import shutil
from os import path, makedirs, walk, sep, scandir
from datetime import datetime
from dataclasses import dataclass
from typing import List


# local libraries


def get_path(input_path: str) -> str:
    """
    Return the absolut path if path relativ to the test_progressbar.py is passed in
    :param input_path:
    :return:
    """
    return path.abspath(path.join(path.dirname(sys.argv[0]), input_path))


def get_dir_path(input_path: str) -> str:
    """
    Return root path of relativ input path, checks if it is a directory
    :param input_path:
    :return:
    """
    output_path = get_path(input_path)
    path_items = output_path.split(sep)
    if "." in path_items[-1]:
        output_path = path.dirname(output_path)
    return output_path


def get_file_path(input_path: str) -> str:
    """
    Return root path of relativ input path, checks if it is a file
    :param input_path:
    :return:
    """
    output_path = get_path(input_path)
    if not path.isfile(output_path):
        raise FileNotFoundError(f"File: {output_path} does not exists!")
    return output_path


def make_path(input_path: str) -> str:
    """
    Create a direcotry if it not exists.
    :param input_path:
    :return: the path as string
    """
    output_path = get_path(input_path)
    if path.isfile(output_path):
        raise NotADirectoryError(f"Path: {output_path} is a File!")
    if path.ismount(output_path):
        raise IsADirectoryError(f"Directory: {output_path} is a mount!")
    if not path.isdir(output_path):
        makedirs(output_path)

    return output_path


def get_DiskSpace(input_path):
    """
    returns free diskspace of a path
    :param input_path:
    :return:
    """
    dir_name = get_dir_path(input_path)
    total, used, free = shutil.disk_usage(dir_name)
    free = round(free / 2 ** 20, 3)
    return free


def get_timestamp(input_path: str, create: bool = False) -> datetime:
    """
    returns datetime timestamp of path_item
    :param create:
    :param input_path:
    :return:
    """
    if create:
        return datetime.fromtimestamp(path.getctime(input_path))
    else:
        return datetime.fromtimestamp(path.getmtime(input_path))


@dataclass
class DirectoryItem:
    folder_list: list
    root_path: str
    file_list: list


def check_date(path_str: str,
               min_date: datetime = None,
               max_date: datetime = None) -> bool:
    """
    :param path_str:
    :param max_date:
    :param min_date:
    :return: return true if file age is within boundaries (min - max Date).
    """
    if not min_date:
        return True
    if not max_date:
        max_date = datetime.now()
    return min_date < get_timestamp(get_path(path_str)) < max_date


def check_string(input_str: str,
                 char_list: list = None,
                 type_any: bool = True) -> bool:
    if not char_list:
        return True
    if type_any:
        return any((char in input_str) for char in char_list)
    else:
        return all((char in input_str) for char in char_list)


def filter_path(path_str: str,
                min_date: datetime = None,
                max_date: datetime = None,
                char_list: list = None):
    return all((check_date(path_str, min_date, max_date),
                check_string(path_str, char_list)))


def crawl(root_path: str) -> List[DirectoryItem]:
    """
    Find all files in actual folder and all subfolder
    :param root_path: [str]
    :return: return a list of lists, each root path represents a row.
     first element of a row is path_items, second is the path, third is list of files
    """
    root_path = get_path(root_path)

    return [DirectoryItem(folder_list=root.split(sep),
                          root_path=root,
                          file_list=files)
            for root, dirs, files in walk(root_path)
            if files]


def get_dirs(root_path: str) -> List[DirectoryItem]:
    root_path = get_path(root_path)
    with scandir(root_path) as dir_entry_iterator:
        return [DirectoryItem(folder_list=dir_entry.path.split(sep),
                              root_path=dir_entry.path,
                              file_list=[])
                for dir_entry in dir_entry_iterator
                if dir_entry.is_dir()]


def get_files(root_path: str) -> DirectoryItem:
    root_path = get_path(root_path)
    with scandir(root_path) as dir_entry_iterator:
        files = [dir_entry.name
                 for dir_entry in dir_entry_iterator
                 if not dir_entry.name.startswith(".") and dir_entry.is_file()]
        return DirectoryItem(folder_list=root_path.split(sep),
                             root_path=root_path,
                             file_list=files)