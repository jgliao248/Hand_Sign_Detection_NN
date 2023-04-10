import cv2
import mediapipe as mp
import time
import numpy as np

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

    def export_hands(self, img):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                pts = self.get_bbox_coordinates(handLms, img.shape)
                cropped_image = img[pts[1]:pts[3], pts[0]:pts[2]]
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



if __name__ == "__main__":
    main()
