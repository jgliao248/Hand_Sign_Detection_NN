
import os
import constants as c

def build_absolute_path(target: str):
    """
    Builds an absolute path with the given name in reference to the file that is calling the function.
    If the string is empty, the directory of the file calling the function will be returned.
    :param target: The target location in which the path is being built towards
    :return: The absolute path to the referenced target.
    """

    if target == "":
        return c.PROJECT_DIRECTORY + '/'
    return c.PROJECT_DIRECTORY + '/' + target


def main():

    print(build_absolute_path(c.NETWORK_DIRECTORY + c.NETWORK_MODEL))

if __name__ == '__main__':
    main()