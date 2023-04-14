import os

import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


import SignLanguageDataset as data
from Network.network import MyNetwork

import constants as c
from util import utilities
import seaborn as sns

from torch import nn
import torch.nn.functional as F
import torch.optim as optim

PADDING = 30



image_path = "/Users/jgliao/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/CS 5330 - Computer Vision/Homework/Final Project/data/asl_dataset/y/hand1_y_bot_seg_1_cropped.jpeg"
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

            #self.mpDraw.draw_detection(img, self.results.multi_hand_landmarks)
                pts = self.get_bbox_coordinates(handLms, img.shape)


                cv2.rectangle(img, [pts[0], pts[1]], [pts[2], pts[3]], (255, 0, 255))


        return img

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist

    def crop(self, img):

        pass

    def export_hands(self, img):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                pts = self.get_bbox_coordinates(handLms, img.shape)
                width = pts[2] - pts[0]
                length = pts[3] - pts[1]
                l = length // 2
                if width >= length:
                    l = width // 2

                center_x, center_y = (pts[2] + pts[0]) // 2, (pts[3] + pts[1]) // 2

                cropped_image = img[(center_y - l):(center_y + l), (center_x - l):(center_x + l)]
                return cropped_image
        else:
            return None

    def get_points(self, img, handNo=0):

        lst = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id)
                #print(lm)
                lst.append((lm.x, lm.y, lm.z))


        return lst





    def get_bbox_coordinates(self, handLadmark, image_shape):
        """
        Get bounding box coordinates for a hand landmark.
        Args:
            handLadmark: A HandLandmark object.
            image_shape: A tuple of the form (height, width).
        Returns:
            A tuple of the form (xmin, ymin, xmax, ymax).
        """
        all_x, all_y = [], []  # store all x and y points in list
        for hnd in self.mpHands.HandLandmark:
            all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1]))  # multiply x by image width
            all_y.append(int(handLadmark.landmark[hnd].y * image_shape[0]))  # multiply y by image height

        return min(all_x) - PADDING, min(all_y) - PADDING, max(all_x) + PADDING, max(all_y) + PADDING  # return as (xmin, ymin, xmax, ymax)

    def get_df_entry(self, lst, true_value):

        my_dict = {"value": true_value}
        cords = ['x', 'y', 'z']

        for item in range(len(lst)):
            for cord in range(3):
                key = str(item) + "-" + cords[cord]
                my_dict[key] = lst[item][cord]

        #print(my_dict)
        df = pd.DataFrame(my_dict, index=[0])
        #print(df)
        return df


class FeaturesModel(nn.Module):
    def __init__(self):
        """
        The constructor of MyNetwork class.
        """
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(63, 2 * 63),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2 * 63, 36),
            nn.ReLU(),
            nn.Dropout(0.4)
        )


    def forward(self, x):
        """
        Computes a forward pass for the network
        :param x: the input to the neural network.
        :return:
        """
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x)

def create_df():
    keys = ["value"]
    cords = ['x', 'y', 'z']

    for item in range(21):
        for cord in range(3):
            key = str(item) + "-" + cords[cord]
            keys.append(key)

    #print(keys)

    df = pd.DataFrame(columns=keys)
    #print(df)
    return df


def main():
    detector = handDetector(True, 1, 0.2, 0.2)
    raw_data_path = utilities.build_absolute_path(c.DATA_DIRECTORY + c.RAW_DATA_DIR)
    raw_data = os.listdir(raw_data_path)
    target_string = "hand5"
    df = create_df()


    for sub in raw_data:
        if len(sub) != 1 or sub == ".DS_Store":
            continue
        folder_path = raw_data_path + sub + "/"
        #print(sub)
        for img in os.listdir(folder_path):
            if target_string in img:
                continue
            img_path = folder_path + img
            #print(folder_path + img)
            img = cv2.imread(img_path)
            lst = detector.get_points(img)
            if len(lst) == 0:
                continue

            entry = detector.get_df_entry(lst, ord(sub))
            df = pd.concat([df, entry], ignore_index=True)

    df.to_csv("data.csv", index=False)


def main2():
    csv = pd.read_csv("data.csv")


    # Plotting the number of data in each label as a countplot.
    plt.figure(figsize=(20, 20))
    sns.countplot(x="value", data=csv)
    plt.show()




if __name__ == '__main__':
    main()
    main2()