import os
from _csv import writer

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

            self.labels, self.x_data, self.y_data, self.z_data = self.process_df(df)

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
        labels = np.array(labels, dtype="int")
        n = len(labels)

        df1 = df.drop(["label"], axis=1)

        c = len(df1.columns) // 3
        x_comp = df1.columns[::3]
        y_comp = df1.columns[1::3]
        z_comp = df1.columns[2::3]

        x = np.array(df1[x_comp], dtype="float32")
        y = np.array(df1[y_comp], dtype="float32")
        z = np.array(df1[z_comp], dtype="float32")


        #print(images)
        return labels, x, y, z

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = np.array([self.x_data[idx], self.y_data[idx], self.z_data[idx]])

        landmarks = torch.FloatTensor(entry)
        landmarks = landmarks

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

    map_label = get_map_label()

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
            # print("ord: ", ord(sub))
            label = map_label.get(sub)
            print("label: ", label)
            if (label == None):
                continue

            entry = get_df_entry(lst, map_label.get(sub))

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

def create_csv2(directory=const.RAW_DATA_DIR):
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

    train = create_df()
    test = create_df()

    map_label = get_map_label()

    for sub in raw_data:
        label = map_label.get(sub)
        if label == None:
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
            # print("ord: ", ord(sub))

            #print("label: ", label)


            entry = get_df_entry(lst, map_label.get(sub))

            # test set alpha
            if target_string in img_path:

                print("test")
                test = pd.concat([test, entry], ignore_index=True)
                continue

            train = pd.concat([train, entry], ignore_index=True)

    # export to csv files
    test.to_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + "test.csv"), index=False)
    train.to_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + "train.csv"), index=False)


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

#
# def main():
#     lst = [0, 2, 3, 4, 5]
#     label = "fuck"
#     append_data(label, lst)
#
#
#
#
# if __name__ == '__main__':
#     main()