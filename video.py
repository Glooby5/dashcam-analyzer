import argparse
import sign_detector
import cv2
import time

RECTANGLE_WIDTH = 10


def detect(filename):
    detector = sign_detector.SignDetector()
    cap = cv2.VideoCapture(filename)

    name = filename.split("\\")[-1].split('.')[0]
    frameCount = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        frameCount += 1

        if not frameCount % 100:
            currentTime = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            print(str(currentTime / 1000))
            print('time:' + str(time.time() - start))

        if not frameCount % 2 and (frameCount % 30):
            continue

        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsvMask = detector.detectRedInHsv(hsvImage)
        hsvMask = cv2.medianBlur(hsvMask, 3)

        if cv2.countNonZero(hsvMask) > 1400:
            outputImage = cv2.medianBlur(hsvMask, 5)
            circles = detector.getSignCircles(outputImage)

            if circles is None:
                if not frameCount % 30:
                    cv2.imwrite("data/negatives/" + name + "-" + str(frameCount) + ".jpg", frame)
                continue

            # frame = detector.highlightCircles(circles, frame)
            # for (x, y, r) in circles:
            #     sign = frame[y - r - RECTANGLE_WIDTH:y + r + RECTANGLE_WIDTH, x - r - RECTANGLE_WIDTH: x + r + RECTANGLE_WIDTH]
            #     # cv2.imshow('output', sign)
            #     cv2.imwrite("data/positives/" + name + "-" + str(frameCount) + ".jpg", sign)
            #     # cv2.imwrite("data/positives/originals/" + name + "-" + str(frameCount) + ".jpg", sign)


            # cv2.waitKey(0)
    print('total:' + str(time.time() - start))

###############################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the video", required=True)
args = vars(ap.parse_args())

# listFile = open('data/video/list.txt', 'r')

with open(args['file'], "r") as ins:
    for line in ins.readlines():
        print(line)
        detect(line.rstrip())


# detect(args['file'])

cv2.destroyAllWindows()
