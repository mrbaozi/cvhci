#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import imutils
import numpy as np

# lower = np.array([0, 48, 80], dtype = "uint8")
# upper = np.array([20, 255, 255], dtype = "uint8")

lower = np.array([10, 10, 10], dtype = "uint8")
upper = np.array([200, 200, 200], dtype = "uint8")

vc = cv2.VideoCapture(0)

while True:
    err, frame = vc.read()
    frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    cv2.imshow("images", np.hstack([frame, skin]))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
