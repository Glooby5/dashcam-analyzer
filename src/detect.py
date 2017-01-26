import cv2
import argparse
from analyzer import video_processor
from analyzer import smart_sign
import time

MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
args = vars(ap.parse_args())

video_processor = video_processor.VideoProcessor(args['file'], args['video'])
k = None

while True:
    frame = video_processor.get_next()

    if frame is False:
        break

    frame_image = video_processor.actual_frame.image

    if frame is None:
        continue

    for sign in frame.signs:

        [x, y, w, h] = sign.get_position()

        cv2.rectangle(frame_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if sign.value is not None:
            cv2.putText(frame_image, str(sign.value), (x + 5, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (46, 255, 0), 3)

    cv2.imshow("frame_image", frame_image)
    k = cv2.waitKey(1)

    if frame and len(frame.signs):
        time.sleep(0.05)

    if k == 27:
        break
