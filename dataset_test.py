import io
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):

    def __init__(self, csv_file, root_dir, transfrom=None):
        try:
            df = pd.read_csv(csv_file)

            if len(df.columns) != 785:
                raise ValueError("Improper data given.")

            self.label, self.data = self.process_df(df)


        except FileExistsError:
            print("Data file not found. Download the appropriate csv files")

        self.root_dir = root_dir
        self.transform = transfrom

    def process_df(self, df: pd.DataFrame):
        """
        Converts the given dataframe to a np array of n entries for 28 by 28 sized images.
        Returns the np array and the labels associated with the entries
        :param df:
        :return:
        """
        labels = df["label"]
        x = df.drop(["label"], axis=1)
        x1 = np.array(x)
        n = len(df.index)
        images = x1.reshape(n, 28, 28)
        return labels, images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]
        label = self.label[idx]

        if self.transform:
            img = self.transform(img)
            return img, label

        return img, label


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
    keys = [x for x in range(len(values))]

    return dict(zip(keys, values))


def main():
    face_dataset = TestDataset(csv_file='data/sign_mnist_train.csv',
                               root_dir='data/')

    label_map = create_labels_dict()
    print(label_map)

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample, label = face_dataset[i]
        print("label is ", label)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        plt.title("Ground Truth: {}".format(label_map.get(label)))
        plt.imshow(sample, cmap="gray_r")

        ax.axis('off')

        if i == 3:
            plt.show()
            break


if __name__ == '__main__':
    main()
