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


cap = cv2.CaptureFromFile(imageFileName)




# cap = cv2.VideoCapture(imageFileName)
print(cap.isOpened())
while(cap.isOpened()):
    print('frame')
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# cap = cv2.VideoCapture(imageFileName)
#
# print(cap)
#
# while (cap.isOpened()):
#     print("fdsfs")
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
