import cv2
import argparse
from analyzer import video_processor
from analyzer import smart_sign

MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
args = vars(ap.parse_args())

video_processor = video_processor.VideoProcessor(args['file'], args['video'])
k = None

while 1:
    frame = video_processor.get_next()

    if frame is False:
        break

    if frame is None:
        continue

    if len(frame.signs) == 0:
        continue

    for sign in frame.signs:
        cv2.imshow("sign", sign.image)

        sign.crop_sign()
        classify_sign = smart_sign.SmartSign(sign, video_processor.knn_model)
        classify_sign.classify()

        print("value: " + str(classify_sign.value))

        k = cv2.waitKey(0)

    if k == 27:
        break
