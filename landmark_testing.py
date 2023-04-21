import os
from threading import Thread

import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from LandmarkDetector import LandmarkDetector
from data.landmark_dataset import LandmarkDataset, get_map_label, append_data

import constants as const
from util import utilities
import tensorflow as tf

from sklearn.model_selection import train_test_split


model_path = const.PROJECT_DIRECTORY + "/" + const.NETWORK_DIRECTORY + "landmark_model.hdf5"
optimizer_path = const.PROJECT_DIRECTORY + "/" + const.NETWORK_DIRECTORY + "landmark_optim.pth"



def get_data():
    # df_train = pd.read_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + "train.csv"))
    # df_test = pd.read_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + "test.csv"))

    df_train = pd.read_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + const.NUMERIC_TRAIN_FILE))
    df_test = pd.read_csv(utilities.build_absolute_path(const.DATA_DIRECTORY + const.NUMERIC_TEST_FILE))

    X_train = np.array(df_train.drop(["label"], axis=1))
    y_train = np.array(df_train["label"])

    X_test = np.array(df_test.drop(["label"], axis=1))
    y_test = np.array(df_test["label"])

    return X_train, y_train, X_test, y_test

def train_model():

    to_labels = get_map_label(is_reverse=True)
    num_classes = len(to_labels)
    print(to_labels)
    print(num_classes)

    X_dataset = np.loadtxt(const.DATABASE_PATH, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 3) + 1)))
    y_dataset = np.loadtxt(const.DATABASE_PATH, delimiter=',', dtype='int32', usecols=(0))
    RANDOM_SEED = 3
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 3,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_path, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    # Model compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )


class UpdateModelThread(Thread):
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None

    # function executed in a new thread
    def run(self):
        # block for a moment
        train_model()
        # store data in an instance variable
        self.value = tf.keras.models.load_model(model_path)
        print("Updated Model")


def main():

    t = UpdateModelThread()

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = LandmarkDetector()
    label = ''
    key = ''
    interval = 1

    img_count = 0

    cv2.namedWindow("hand", cv2.WINDOW_AUTOSIZE)
    to_labels = get_map_label(is_reverse=True)

    # Loading the saved model
    model = tf.keras.models.load_model(model_path)

    is_collecting = False
    is_predicting = False

    while key != "q":
        success, img = cap.read()
        img_copy = img.copy()
        img_copy = detector.find_hands(img_copy)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img_copy, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        if is_predicting and success:
            vect = np.array(detector.get_points(img, is_flatten=True))
            if len(vect) != 0:
                predict_results = np.squeeze(model.predict(np.array([vect])))
                best_match_idx = np.argmax(predict_results)
                prediction_str = "Prediction is " + to_labels.get(best_match_idx) + " at " + str(round(predict_results[best_match_idx] * 100, 2)) + "%"
                cv2.putText(img_copy, prediction_str, (10, 110), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.putText(img_copy, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        if is_collecting:
            print(img_count)
            pts = detector.get_points(img, is_flatten=True)
            # check for empty values
            if len(pts) == 0 or pts is None:
                continue
            img_count += 1
            append_data(label, pts)
            if img_count % 2 == 1:
                img_copy = cv2.bitwise_not(img_copy)

            if img_count == const.TOTAL_IMAGES:
                is_collecting = False
                interval = 1
                t.run()
                model = t.value


        cv2.imshow("Image", img_copy)
        key = cv2.waitKey(interval)
        if key == 27:  # Esc key to stop
            break
        # if key == 104 and success:  # h key
        #     # vect = np.array(detector.get_points(img, is_flatten=True))
        #     # # print(vect)
        #     # # print(vect.shape)
        #     # predict_result = model.predict(np.array([vect]))
        #     # # print(np.squeeze(predict_result))
        #     # print("guess")
        #     # print(np.squeeze(predict_result))
        #     # print(to_labels.get(np.argmax(np.squeeze(predict_result))))
        if key == 104:  # h key
            is_predicting = not is_predicting
        if key == 32 and success:   # space button
            # toggle collecting
            is_collecting = not is_collecting

            if is_collecting:
                label = input("Enter sign for data collection: ")
                interval = 50
                img_count = 0





if __name__ == '__main__':
    main()
