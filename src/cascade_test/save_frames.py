import cv2
import argparse
from datetime import datetime
import os
MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video", required=True)
ap.add_argument("-o", "--output", help="path to the output", required=True)
args = vars(ap.parse_args())

video = cv2.VideoCapture(args['video'])
frame_number = 0

directory = args['output']

if not os.path.exists(directory):
    os.makedirs(directory)

while True:
    ret, image = video.read()

    if ret is False:
        break

    frame_number += 1

    if frame_number % 40:
        continue

    cv2.imwrite(directory + '/frame' + str(datetime.now()) + '-' + str(frame_number) + '.jpg', image)
