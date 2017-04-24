import cv2
import argparse
from analyzer import video_processor
from analyzer import type_sign
import os

MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
args = vars(ap.parse_args())

video_processor = video_processor.VideoProcessor(args['file'], args['video'])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280,720))
k = None

while True:
    frame = video_processor.get_next()

    if frame is False:
        break

    frame_image = video_processor.actual_frame.image
    copy_frame_image = frame_image.copy()

    if frame is None:
        continue

    for sign in frame.signs:

        [x, y, w, h] = sign.get_position()

        cv2.rectangle(copy_frame_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if sign.value is not None:
            cv2.putText(copy_frame_image, str(sign.value), (x + 5, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (46, 255, 0), 3)
            cv2.putText(copy_frame_image, "rychlost", (x + -50, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 50, 255), 3)

        if isinstance(sign, type_sign.TypeSign):
            cv2.putText(copy_frame_image, sign.mapping[sign.type], (x + -50, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255), 3)

        cv2.imshow("copy_frame_image", copy_frame_image)
        key = cv2.waitKey(0)
        # key = 0

        print(key)

        if key == 119:
            out.release()

        if key == 120:
            continue

        cv2.rectangle(frame_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if sign.value is not None:
            cv2.putText(frame_image, str(sign.value), (x + 5, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (46, 255, 0), 3)

            if key != 121:
                cv2.putText(frame_image, "Rychlostni omezeni", (x - 50, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 50, 255), 3)

        if isinstance(sign, type_sign.TypeSign):
            if key != 121:
                cv2.putText(frame_image, sign.mapping[sign.type], (x - 50, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255), 3)

    out.write(frame_image)




    # if frame.number % 10 == 0:
    cv2.imshow("frame_image", frame_image)

    k = cv2.waitKey(1)
    # cv2.waitKey(0)


print("end")
out.release()
cv2.destroyAllWindows()
