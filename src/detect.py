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
k = None

while True:
    frame = video_processor.get_next()


    if frame is False:
        break

    frame_image = video_processor.actual_frame.image

    if frame is None:
        continue

    # if frame.number % 5:
    #     continue

    for sign in frame.signs:

        [x, y, w, h] = sign.get_position()

        cv2.rectangle(frame_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        continue

        if sign.value is not None:
            cv2.putText(frame_image, str(sign.value), (x + 0, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (46, 255, 0), 3)
            cv2.putText(frame_image, "Rychlostni omezeni", (x + -120, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 50, 255), 3)
            print(str(frame.number) + ': ' + str(sign.value))

            # directory = 'export2/rychlost'
            #
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            #
            # cv2.imwrite(directory + '/sign' + args['video'].split('/')[-1] + str(frame.number) + '.jpg',
            #             sign.sign.original)

        if isinstance(sign, type_sign.TypeSign) and w > 60:
            cv2.putText(frame_image, sign.mapping[sign.type], (x -230, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 255), 3)

            # directory = 'export2/' + sign.mapping[sign.type]
            #
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            #
            # cv2.imwrite(directory + '/sign' + args['video'].split('/')[-1] + str(frame.number) + '.jpg', sign.sign.original)

        # k = cv2.waitKey(0)

    if len(frame.signs):
        directory = 'frame_export_2'

        if not os.path.exists(directory):
            os.makedirs(directory)

        cv2.imwrite(directory + '/frame' + str(frame.number) + '.jpg', frame_image)

    # if frame.number % 10 == 0:
    cv2.imshow("frame_image", frame_image)
    k = cv2.waitKey(1)

    # cv2.waitKey(0)
cv2.destroyAllWindows()
