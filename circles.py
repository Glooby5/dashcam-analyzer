import numpy as np
import cv2
import argparse


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

image = cv2.imread(imageFileName)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsvMask = detectRedInHsv(hsvImage)

# image = cv2.medianBlur(image,5)
output = image.copy() #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# image = cv2.bitwise_and(image, image, mask = hsvMask)
# image = cv2.cvtColor(hsvMask, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(hsvMask, 5)
cv2.imshow('segment', image)
circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,50,
                            param1=50,param2=20,minRadius=0,maxRadius=0)

print(circles)
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)



cv2.imshow('detected circles', output)



cv2.waitKey(0)
cv2.destroyAllWindows()