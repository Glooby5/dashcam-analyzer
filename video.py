import argparse
import sign_detector
import cv2
import time


def detect(filename):
    detector = sign_detector.SignDetector()
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsvMask = detector.detectRedInHsv(hsvImage)

        if cv2.countNonZero(hsvMask) > 1200:
            outputImage = cv2.medianBlur(hsvMask, 5)
            circles = detector.getSignCircles(outputImage)

            if circles is None:
                print("none")
                continue

            frame = detector.highlightCircles(circles, frame)

            cv2.imshow('output', frame)
            cv2.waitKey(0)


###############################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the video", required=True)
args = vars(ap.parse_args())

detect(args['file'])

cv2.destroyAllWindows()
