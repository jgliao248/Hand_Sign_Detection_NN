import cv2
import mediapipe as mp
import time
import numpy as np
import torch
from torch import optim

import SignLanguageDataset as data
from Network.network import MyNetwork

import constants as c

PADDING = 30
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

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        return lmlist

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



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    key = ''

    learning_rate = 1e-3
    momentum = 0.5

    maps_label = data.create_labels_dict()

    model_path = c.NETWORK_MODEL_PATH
    optimizer_path = c.NETWORK_OPTIMIZER_PATH

    model = MyNetwork()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum)

    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))


    train_data = data.SignLanguageDataset(data.TRAIN_PATH, data.DATA_DIRECTORY_PATH)
    print(train_data[0])
    mean = np.full((224, 224), train_data.mean)
    print(mean)
    #print(train_data.std)

    std = np.full((224, 224), np.mean(train_data.std))
    print(std)


    model.eval()

    cv2.namedWindow("hand", cv2.WINDOW_AUTOSIZE)

    while key != "q":
        success, img = cap.read()
        img_copy = img.copy()
        img_copy = detector.findHands(img_copy)
        lmlist = detector.findPosition(img_copy)

        """if len(lmlist) != 0:
            print(lmlist[4])"""

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img_copy)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to stop
            break
        if key == 32: # space key
            cv2.imwrite("test.png", detector.export_hands(img))
        if key == 104 and success: # h key
            img = cv2.cvtColor(detector.export_hands(img), cv2.COLOR_BGR2GRAY)
            img = train_data.process_image(img)

            print(img)


            with torch.no_grad():
                output = model(img.unsqueeze(0))
                print(maps_label.get(output.data.max(1, keepdim=True)[1][0].item()))




if __name__ == "__main__":
    main()
