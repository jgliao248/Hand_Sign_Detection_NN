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





class LandmarkDetector():
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
        """
        This function attempts to find hands in the given image. It will return an image with a bounded box around it.
        If draw is true, the landmarks will also be drawn.
        :param img: the input image
        :param draw: if true, draw the landmarks.
        :return: img with bounded box and/or landmarks drawn if detected.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                # print(handLms)
                pts = self.get_bbox_coordinates(handLms, img.shape)

                cv2.rectangle(img, [pts[0], pts[1]], [pts[2], pts[3]], (255, 0, 255))

        return img


    def crop_hands(self, img):
        """
        With the given image, the function will create a cropped image of the hand with equal sides
        :param img: the given image
        :return: the cropped hand if detected. none if not
        """
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

    def get_points(self, img, handNo=0, is_flatten=False):
        """
        Given an image and the number of hands in the image, this function will check for a hand detection and return
        a list of landmark points. There are 21 landmark points from 0 to 20. The landmark points are normalized and
        stored in tuple with the respective indices refer to each landmark points
        :param img:
        :param handNo:
        :return:
        """

        lst = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        norm = 0
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                if id == 0:
                    ref_pt = lm

                #print(id)
                #print(lm)
                pt = [lm.x - ref_pt.x, lm.y - ref_pt.y, lm.z - ref_pt.z]

                norm = max(norm, abs(min(pt)), max(pt))

                lst = lst + pt
            lst = [x / norm for x in lst] # final step of normalization
        if is_flatten:
            return lst

        nested = []
        j = 0
        for i in range(len(lst)):
            if j == 0:
                pt = [lst[i]]
                j += 1
                continue
            pt.append(lst[i])

            if j == 2:
                nested.append(pt)
                j = 0
                continue

            j += 1

        return nested


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

        return min(all_x) - c.PADDING, min(all_y) - c.PADDING, max(all_x) + c.PADDING, max(all_y) + c.PADDING
        # return as (xmin, ymin, xmax, ymax)


