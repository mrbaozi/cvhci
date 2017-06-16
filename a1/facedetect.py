#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import cv2
import imutils

def main():
    vc = cv2.VideoCapture(0)

    cascadePath = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    while True:
        err, frame = vc.read()
        frame = imutils.resize(frame, width = 400)

        # face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        annotatedFrame = np.copy(frame)
        for (_x, _y, _w, _h) in faces:
            (x, y, w, h) = (_x, _y, _w, _h)
            cv2.rectangle(annotatedFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # detect skin only if face found
        skin = np.copy(frame)
        if len(faces) > 0:
            # we work in hsv color space
            workframe = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            # workframe[0] = cv2.equalizeHist(workframe[0])
            # workframe[1] = cv2.equalizeHist(workframe[1])
            # workframe[2] = cv2.equalizeHist(workframe[2])
            workframe = cv2.merge(workframe)

            roi = workframe[y+h/2:y+h, x+w/4:x+w-w/4]

            # sample skin color from face region
            roiMean = np.mean(np.mean(roi, axis=0), axis=0)
            lower = np.array(roiMean - np.array([7, 40, 60]), dtype="uint8")
            upper = np.array(roiMean + np.array([7, 40, 60]), dtype="uint8")

            # print(lower, upper)

            skinMask = cv2.inRange(workframe, lower, upper)

            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
            # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
            # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

            skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        cv2.imshow("images", np.hstack([cv2.flip(annotatedFrame, 1), cv2.flip(skin, 1)]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
