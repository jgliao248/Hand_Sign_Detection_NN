import os

import cv2
import numpy as np
import pandas as pd

import constants as const

import torch
from torch.utils.data import Dataset

from LandmarkDetector import LandmarkDetector
from util import utilities


class LandmarkDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        try:
            df = pd.read_csv(csv_file)

            self.labels, self.data = self.process_df(df)

        except FileExistsError:
            print("Data file not found. Download the appropriate csv files")

        self.root_dir = root_dir
        self.transform = transform

    def process_df(self, df: pd.DataFrame):
        """
        Converts the given dataframe to a np array of n entries for 28 by 28 sized images.
        Returns the np array and the labels associated with the entries
        :param df:
        :return:
        """
        labels = df["label"]
        x = df.drop(["label"], axis=1)

        x1 = np.array(x, dtype="float32")

        # print(images)
        return labels, x1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        landmarks = torch.FloatTensor(self.data[idx])
        landmarks = landmarks.unsqueeze(0)

        label = self.labels[idx]

        if self.transform:
            landmarks = self.transform(landmarks)
            return landmarks, label

        return landmarks, label


def get_df_entry(lst, true_value):
    """
    A helper method that creates a df entry from the given lst data from LandmarkDetector.get_points() and the truth
    value corresponding to that data.
    :param lst: landmark data points
    :param true_value: letter in AS SCI corresponding to the data
    :return: df containing the data
    """

    my_dict = {"label": true_value}
    cords = ['x', 'y', 'z']

    for item in range(len(lst)):
        for cord in range(len(cords)):
            key = str(item) + "-" + cords[cord]
            my_dict[key] = lst[item][cord]

    # print(my_dict)
    df = pd.DataFrame(my_dict, index=[0])
    # print(df)
    return df


def create_df():
    """
    Creates ta blank df with the correct column names
    :return: blank df with the correct column names
    """
    keys = ["label"]
    cords = ['x', 'y', 'z']

    for item in range(21):
        for cord in range(3):
            key = str(item) + "-" + cords[cord]
            keys.append(key)

    # print(keys)

    df = pd.DataFrame(columns=keys)
    # print(df)
    return df


def create_csv(directory=const.RAW_DATA_DIR):
    """
    Generates a csv files of the images within the given directory. The data contains the hand landmarks
    of several American Sign Language alphabet and numbers.

    There will be 4 csv files that will be generated:
    - alphabet test
    - alphabet train
    - numeric test
    - numeric train
    :param directory: the directory path string for the images
    :return: none
    """

    detector = LandmarkDetector(True, 1, 0.2, 0.2)
    raw_data_path = utilities.build_absolute_path(const.DATA_DIRECTORY + directory)
    raw_data = os.listdir(raw_data_path)

    target_string = "hand5"  # test data sub set
    numerics = [str(x) for x in range(0, 10)]  # numerics

    train_alpha = create_df()
    test_alpha = create_df()
    train_numeric = create_df()
    test_numeric = create_df()

    for sub in raw_data:
        if len(sub) != 1 or sub == ".DS_Store":
            continue
        folder_path = raw_data_path + sub + "/"
        # print(sub)
        for img in os.listdir(folder_path):
            # build image path and load the image
            img_path = folder_path + img

            img = cv2.imread(img_path)
            lst = detector.get_points(img)
            if len(lst) == 0:
                continue
            print(img_path)
            print("sub: ", sub)
            print("ord: ", ord(sub))
            entry = get_df_entry(lst, ord(sub))

            # test set alpha
            if target_string in img_path:
                # test set numeric
                if sub in numerics:
                    print("test numeric")
                    test_numeric = pd.concat([test_numeric, entry], ignore_index=True)
                    continue
                print("test alpha")
                test_alpha = pd.concat([test_alpha, entry], ignore_index=True)
                continue

            # train set numeric
            if sub in numerics:
                print("train numeric")
                train_numeric = pd.concat([train_numeric, entry], ignore_index=True)
                continue
            # train set alpha
            print("train alpha")
            train_alpha = pd.concat([train_alpha, entry], ignore_index=True)

    # export to csv files
    train_alpha.to_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + const.ALPHA_TRAIN_FILE), index=False)
    test_alpha.to_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + const.ALPHA_TEST_FILE), index=False)
    train_numeric.to_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + const.NUMERIC_TRAIN_FILE), index=False)
    test_numeric.to_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + const.NUMERIC_TEST_FILE), index=False)

def find_csv():

    a_train_exists = os.path.exists(utilities.build_absolute_path(const.DATA_DIRECTORY + const.ALPHA_TRAIN_FILE))
    a_test_exists = os.path.exists(utilities.build_absolute_path(const.DATA_DIRECTORY + const.ALPHA_TEST_FILE))
    n_train_exists = os.path.exists(utilities.build_absolute_path(const.DATA_DIRECTORY + const.NUMERIC_TRAIN_FILE))
    n_test_exists = os.path.exists(utilities.build_absolute_path(const.DATA_DIRECTORY + const.NUMERIC_TEST_FILE))

    if not (a_train_exists and a_test_exists and n_train_exists and n_test_exists):
        print("Missing csv file(s)")
        create_csv()
        return
    print("All csv files exist")


# def main():
#     dataset = LandmarkDataset(utilities.build_absolute_path(const.DATA_DIRECTORY + const.ALPHA_TRAIN_FILE),
#                               utilities.build_absolute_path(const.DATA_DIRECTORY))
#
#     print(dataset.__getitem__(0))
#
# if __name__ == '__main__':
#     main()