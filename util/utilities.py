"""
Justin Liao
CS5330
Spring 2023
Final Project

This file contains utility functions for file management
"""

from _csv import writer

import constants as const


def build_absolute_path(target: str):
    """
    Builds an absolute path with the given name in reference to the file that is calling the function.
    If the string is empty, the directory of the file calling the function will be returned.
    :param target: The target location in which the path is being built towards
    :return: The absolute path to the referenced target.
    """

    if target == "":
        return const.PROJECT_DIRECTORY + '/'
    return const.PROJECT_DIRECTORY + '/' + target

def get_map_label(file=const.MAP_LABEL_FILE_PATH, is_reverse=False):
    map_label = {}
    print(file)
    with open(file) as file_data:
        for i, line in enumerate(file_data):
            if is_reverse:
                map_label[i] = line.strip()
            else:
                map_label[line.strip()] = i
    return map_label


def append_data(label: str, pts: list):
    from_labels = get_map_label()
    int_key = from_labels.get(label)
    if int_key is None:
        with open(const.MAP_LABEL_FILE_PATH, 'a') as file:
            file.writelines(label + "\n")
            file.close()
        int_key = len(from_labels)

    with open(const.DATABASE_PATH, 'a') as file:
        entry = [int_key] + pts
        w = writer(file)
        w.writerow(entry)
        file.close()
#
# def main():
#
#     print(build_absolute_path(c.NETWORK_DIRECTORY + c.NETWORK_MODEL))
#
# if __name__ == '__main__':
#     main()