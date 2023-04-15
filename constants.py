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

# landmark based data files
RAW_DATA_DIR = "asl_dataset/"
ALPHA_TRAIN_FILE = "alpha_train_file.csv"
NUMERIC_TRAIN_FILE = "numeric_train_file.csv"
ALPHA_TEST_FILE = "alpha_test_file.csv"
NUMERIC_TEST_FILE = "numeric_test_file.csv"
MAP_LABEL_FILE_PATH = PROJECT_DIRECTORY + "/" + DATA_DIRECTORY + "map_label.txt"
DATABASE_PATH = PROJECT_DIRECTORY + "/" + DATA_DIRECTORY + "database.csv"

NETWORK_DIRECTORY = "network/"
NETWORK_MODEL = "model.pth"
NETWORK_OPTIMIZER = "optimizer.pth"

NETWORK_MODEL_PATH = PROJECT_DIRECTORY + "/" + NETWORK_DIRECTORY + NETWORK_MODEL
NETWORK_OPTIMIZER_PATH = PROJECT_DIRECTORY + "/" + NETWORK_DIRECTORY + NETWORK_OPTIMIZER

UTIL_DIRECTORY = "util/"

# network constants
LOG_INTERVAL = 10

PADDING = 30

# image
TOTAL_IMAGES = 50



