Justin Liao
CS5330
Spring 2023
Final Project

# Description:
My project is a simple hand gesture recognition. The app will utilize the webcam to find the user's right hand and
match the bets match for the displayed hand. The app allows for the user to input new data into the database by
collecting features of the hand. A neural network is used to categorize the specific hand features for that specific
hand gesture. This network is used for real time display of the actual hand gesture/predict new hand gestures.

# Presentation and Demo Link
The video for the presentation and demo are one video.
https://drive.google.com/drive/folders/18VzBjDvAQ0UgF018errFHAR5A194dKrq?usp=share_link

# Project Directory Structure
.
├── Final Report.docx
├── HandSignDetection.py
├── constants.py
├── data
    ├── database.csv
    ├── map_label.txt
├── network
    ├── InitialDesign.py
    └── landmark_model.hdf5
└── util
    └── utilities.py

# Libraries:

Install these libraries prior to running the program.

- opencv-python
    - used to create the main UI and get the input video feed
- mediapipe
    - used hand solution to get hand landmarks to extract the features needed for classification
    - note:
        apple-silicon requires the use of a third party version: pip install mediapipe-silicon
        Must be running Python 3.7 - 3.10 https://github.com/google/mediapipe/blob/master/docs/getting_started/troubleshooting.md#python-pip-install-failure
- tensorflow
    - used to crate the main dense neural network for classification of the hands signs.
    - note:
        apply-silicon version
        https://developer.apple.com/metal/tensorflow-plugin/
- pandas
    - used for database file manipulation
- numpy
    - core supporting library to package image data and vectors in multi-dimensional arrays
- scikit-learn
    - AI and ML tool; used to create training and testing sets for training and assessing the main model

# Running the program
Please make sure that the project structure matches that of the described structure noted earlier in the Readme file.
Upload to academic grading websites does not always retain the project's directory structure. To run the program, simply
run the HandSignDetection.py file in an IDE or terminal.

## Commands
### Add new hand sign
Keypress: space
Description: This command prompts the user to enter a new semantic meaning for a hand sign. The user can enter a new
entry and press enter when done. At the point of providing a name for the new entry, the user can cancel this command by
pressing the esc button. If the name value was confirmed with enter, the program will start the "collection" mode to
start collecting data points. There will be 50 data points that the program will collect. If a valid data point is being
collected, the program will alternate negative colors in the video feed output to notify the user. Move the hand around
to different locations within the video feed frame at slightly varying orientations. Do not over exaggerate the
orientation of the hand as it will affect the training as well as possibly overlap with another entry. Once the 50 data
points are collected, the program will start training the model with the new data. Please allow the program to train for
a few minutes. During this time, the previous model without the new entry will still work.

### Prediction
Keypress: h
Description: This command will put the program in "PREDICTION" mode which will use the trained model to give the best
match to the given hand sign if a hand is detected in the video feed. The program will display the best match with the
percentage confidence. For instance, if the American sign language "A" was given, the program will display "A at 96%" to
signify the semantic match.

### Idle mode
Keypress: i
Description: This command will put the program in "IDLE" mode which will not do anything besides show the landmarks
on the hand if detected.

# Notes:
- performance was not great with convolution based neural network. Perhaps, the low resolution made it hard to properly
train the model. The model gets trained within 1 or 2 epochs of train batch of 128 and test batch of 64.

- when using landmarks, it is difficult to find landmarks for certain letter photo examples.


# Resources
[13]	Vignesh, V. (2020, July 1). CNN - Pytorch - 96%. Kaggle. Retrieved April 24, 2023, from https://www.kaggle.com/code/vijaypro/cnn-pytorch-96
[14]	Thakur, A. (2019, April 28). American Sign Language Dataset. Kaggle. Retrieved April 24, 2023, from https://www.kaggle.com/datasets/ayuraj/asl-dataset
[15]	Kiselov, N. (n.d.). hand-gesture-recognition-mediapipe. GitHub. Retrieved April 24, 2023, from https://github.com/kinivi/hand-gesture-recognition-MediaPipe
