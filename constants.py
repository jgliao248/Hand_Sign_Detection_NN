"""
Justin Liao
CS5220
Spring 2023
Final Project

This file contains constants relative to this program package
"""

import os

# project dir
PROJECT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# data related file names
DATA_DIRECTORY = "data/"
TRAIN_DATA_FILE = "sign_mnist_train.csv"
TEST_DATA_FILE = "sign_mnist_test.csv"


NETWORK_DIRECTORY = "network/"
NETWORK_MODEL = "model.pth"
NETWORK_OPTIMIZER = "optimizer.pth"

NETWORK_MODEL_PATH = PROJECT_DIRECTORY + "/" + NETWORK_DIRECTORY + NETWORK_MODEL
NETWORK_OPTIMIZER_PATH = PROJECT_DIRECTORY + "/" + NETWORK_DIRECTORY + NETWORK_OPTIMIZER

UTIL_DIRECTORY = "util/"

# network constants
LOG_INTERVAL = 10



