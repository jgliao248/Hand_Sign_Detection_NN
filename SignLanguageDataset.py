import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import util.utilities as util
import constants as const

import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIRECTORY_PATH = util.build_absolute_path(const.DATA_DIRECTORY)
TRAIN_PATH = util.build_absolute_path(const.DATA_DIRECTORY + const.TRAIN_DATA_FILE)
TEST_PATH = util.build_absolute_path(const.DATA_DIRECTORY + const.TEST_DATA_FILE)


class SignLanguageDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.std = None
        self.mean = None
        try:
            df = pd.read_csv(csv_file)

            if len(df.columns) != 785:
                raise ValueError("Improper data given.")

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
        self.mean = np.average(x)

        self.std = np.std(x)
        x -= self.mean

        x /= self.std



        x1 = np.array(x, dtype="float32")


        n = len(df.index)
        images = x1.reshape(n, 28, 28)
        #print(images)
        return labels, images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = cv2.resize(self.data[idx], (224, 224))
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        img /= 255.

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
            return img, label

        return img, label

    def show_images(self):

        labels_map = create_labels_dict()
        plt.figure(figsize=(20, 20))
        for i in range(0, 50):
            plt.subplot(10, 5, i + 1)
            plt.axis("off")
            plt.imshow(self.data[i], cmap="gray_r")
            plt.title("Ground Truth: {}".format(labels_map.get(self.labels[i])))
        plt.show()

    def process_image(self, img):
        img = cv2.resize(img, (28, 28))
        cv2.imwrite("processed.png", img)
        img = img.flatten()

        img = img - self.mean

        img = img / self.std
        img = np.array(img, dtype="float32")

        n = len(img)
        img = img.reshape(28, 28)



        img = cv2.resize(img, (224, 224))
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        img /= 255.
        return img



def create_labels_dict():
    """
    Creates a dictionary for the sign language alphabet. Since J and Z both require movements,
    they are not included in the dictionary.
    :return: A dictionary for the labels and alphabet.
    """
    # range of A to Y
    keys = list(range(65, 90))
    # remove J
    keys.remove(74)

    values = [chr(x) for x in keys]
    keys = [x - 65 for x in keys]

    return dict(zip(keys, values))


def main():
    #print(create_labels_dict())
    dataset = SignLanguageDataset(TRAIN_PATH, DATA_DIRECTORY_PATH)
    #dataset.show_images()
    print(dataset[0])

if __name__ == '__main__':
    main()
