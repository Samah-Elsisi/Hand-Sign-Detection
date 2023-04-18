import cv2          #openCV library
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#open the camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)   #open the camera

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

while True :
    # a method to read the image captured by the camera and stores it in img variable
    # success is a boolean variable that stores true if the img is captured and stored successfully and stores false otherwise
    success, img = cap.read()

    # passes the image captured by the camera to capture the hand object from it
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]   #???????????????
        x, y, w, h = hand['bbox']   #boundingBOX of the hand img
        imgWhite = np.ones((imgSize ,imgSize ,3), np.uint8)*255
        imgCrop = img[y-offset: y+h+offset, x-offset: x + w + offset]
        # img shape is an property which is baiscially a list that contains the dimensions of that img
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)    #??????
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap , :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)


