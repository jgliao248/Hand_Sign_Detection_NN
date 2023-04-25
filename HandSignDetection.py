"""
Justin Liao
CS5330
Spring 2023
Final Project


This file contains the main driver for running the hand sign detection application.

"""

import time
from enum import Enum
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import constants as const
from LandmarkDetector import LandmarkDetector
from util.utilities import get_map_label, append_data



class Mode(Enum):
    """
    Enum to hold different operation modes within the application
    """
    PREDICTING = "PREDICTING"
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"


class UpdateModelThread(Thread):
    """
    A thread class to train in the background
    """

    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.model = None
        self.isDoneTraining = False

    # function executed in a new thread
    def run(self):
        # block for a moment
        train_model()
        # store data in an instance variable
        self.model = tf.keras.models.load_model(const.TENSOR_FLOW_MODEL)
        print("Updated Model")
        self.isDoneTraining = True


def create_model():
    """
    Creates a simple fully connected model with 1D input.
    :return: A simple fully connected model with 1D input.
    """

    to_labels = get_map_label(is_reverse=True)
    num_classes = len(to_labels)
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

    # Model compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model():
    """
    Trains the model from a loaded data set. The trained model is stored in an external file.
    :return: none
    """

    X_dataset = np.loadtxt(const.DATABASE_PATH, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 3) + 1)))
    y_dataset = np.loadtxt(const.DATABASE_PATH, delimiter=',', dtype='int32', usecols=0)

    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75,
                                                        random_state=const.RANDOM_SEED)

    model = create_model()

    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        const.TENSOR_FLOW_MODEL, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )


def get_new_entry_name(img):
    """
    Graphical UI to get user's label name.
    :param img: An image to write draw the UI on during the label name collection.
    :return: the label name
    """

    (width, height, _) = img.shape
    key = ""
    entry_name = ""
    text = ""

    pulse = True
    while key != 13:    # enter key
        img_copy = img.copy()
        if pulse:
            text = "Entry name: " + entry_name + "_"
        else:
            text = "Entry name: " + entry_name

        cv2.putText(img_copy, text, (width//2 - 50, height//2), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Input", img_copy)
        key = cv2.waitKey(100)
        pulse = not pulse
        if key == 13:   # enter press
            break
        if (key == 8 or key == 127) and len(entry_name) >= 0:    # backspace pressed
            entry_name = entry_name[:-1]
            continue
        if key == 27:   # esc pressed
            entry_name = ""
            break   # cancelled
        if key > -1:
            entry_name = entry_name + chr(key)

    cv2.destroyAllWindows()
    return entry_name



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
    model = tf.keras.models.load_model(const.TENSOR_FLOW_MODEL)

    mode = Mode.IDLE
    is_training = False

    while True:

        if not t.is_alive() and t.isDoneTraining:
            t.join()
            model = t.model
            mode = Mode.IDLE
            to_labels = get_map_label(is_reverse=True)
            t = UpdateModelThread()
            is_training = False

        success, img = cap.read()
        if not success:
            print("Please check camera privacy settings")
            return

        img_copy = img.copy()
        img_copy = detector.find_hands(img_copy)


        # frame rate text
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img_copy, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # mode text
        cv2.putText(img_copy, "MODE: " + mode.value, (10, 110), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # training text
        if is_training:
            cv2.putText(img_copy, "Model is Training", (10, 190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # prediction text
        if mode == Mode.PREDICTING and success:
            vect = np.array(detector.get_points(img, is_flatten=True))
            if len(vect) != 0:
                predict_results = np.squeeze(model.predict(np.array([vect])))
                best_match_idx = np.argmax(predict_results)
                prediction_str = "Prediction is \"" + to_labels.get(best_match_idx) + "\" at " + str(
                    round(predict_results[best_match_idx] * 100, 2)) + "%"
                cv2.putText(img_copy, prediction_str, (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # process data collection
        if mode == Mode.COLLECTING:
            pts = detector.get_points(img, is_flatten=True)
            if img_count % 2 == 1:
                img_copy = cv2.bitwise_not(img_copy)
            # check for empty values
            if len(pts) > 0 and pts is not None:
                img_count += 1
                append_data(label, pts)
                if img_count == const.TOTAL_IMAGES:
                    interval = 1
                    t.start()
                    mode = Mode.IDLE
                print(img_count)

        cv2.imshow("Image", img_copy)
        key = cv2.waitKey(interval)
        # print(key)
        #key = cv2.waitKey()
        if key == 27:  # Esc key to stop
            if is_training:
                continue
            break
        if key == 105:   # i pressed
            mode = Mode.IDLE
        if key == 104:  # h key

            mode = Mode.PREDICTING
        if key == 32 and success:  # space button
            if is_training:
                continue
            label = get_new_entry_name(img_copy)
            print(label)
            if label != "":
                mode = Mode.COLLECTING
                interval = 50
                img_count = 0
        if key == 115:  # s pressed
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # release resources
    cv2.destroyAllWindows()
    cap.release()



if __name__ == '__main__':
    main()
