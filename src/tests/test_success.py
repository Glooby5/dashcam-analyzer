import cv2
import argparse
import os

MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
args = vars(ap.parse_args())

video = cv2.VideoCapture(args['video'])
cascade = cv2.CascadeClassifier(args['file'])
count = 0
all = 0

for img in os.listdir('falsepositive'):
    image = cv2.imread('falsepositive/' + img)

    if image is None or image is False:
        continue

    all += 1

    sign_hits = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20))

    for (x, y, w, h) in sign_hits:
        # if w > MAX_WIDTH or h > MAX_WIDTH:
        #     continue
        #
        # if w < 0 or h < 0:
        #     continue

        count += 1
        print(str(count))
        # continue

print("all: " + str(all))
print(str(count))
print(str(100 * count / all) + " %")
cv2.destroyAllWindows()
