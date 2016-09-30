import numpy as np
import cv2
import argparse


def detectRedInRgb(image):
    lower = np.array([15, 15, 100], dtype=np.uint8)
    upper = np.array([90, 70, 255], dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)

    return mask


def detectRedInHsv(image):
    lower = np.array([170, 100, 100], dtype=np.uint8)
    upper = np.array([190, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)

    return mask

###############################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--file", help = "path to the image")
args = vars(ap.parse_args())

imageFileName = args['file']

if not imageFileName:
    print("chybi obrazek")
    exit(1)


# image = cv2.imread(imageFileName)
#
# rgbMask = detectRedInRgb(image)
#
#
#
# hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsvMask = detectRedInHsv(hsvImage)
#
#
# print(cv2.countNonZero(hsvMask))

# output = cv2.bitwise_and(image, image, mask = mask)
# cv2.imshow('image', image)
# cv2.imshow('mask', rgbMask)
# cv2.imshow('hsvMask', hsvMask)
# cv2.imshow('output', np.hstack([image, output]))

# cv2.waitKey(0)
# cv2.destroyAllWindows()

###############################################################################


cap = cv2.VideoCapture(imageFileName)

counter = 0
images = 0

while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        break

    images = images + 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame = frame[0:height, int(width / 2):width]

    if images % 2:
        continue

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsvMask = detectRedInHsv(hsvImage)

    if cv2.countNonZero(hsvMask) > 1200:
        image = cv2.medianBlur(hsvMask, 5)
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=20, minRadius=22, maxRadius=50)

        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            cv2.imwrite(imageFileName + '-' + str(counter) + '.png', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter += 1


cap.release()
cv2.destroyAllWindows()


