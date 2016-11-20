import argparse

import cv2

from src.lib import sign_detector


def detect(image):
    detector = sign_detector.SignDetector()

    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvMask = detector.detectRedInHsv(hsvImage)

    outputImage = cv2.medianBlur(hsvMask, 3)
    circles = detector.getSignCircles(outputImage)
    outputImage = detector.highlightCircles(circles, image)

    cv2.imshow('image', image)
    cv2.imshow('hsvMask', hsvMask)
    cv2.imshow('output', outputImage)

###############################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the image", required=True)
args = vars(ap.parse_args())

image = cv2.imread(args['file'])

detect(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
