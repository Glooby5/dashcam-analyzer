import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the image", required=True)
ap.add_argument("-t", "--time", help="time in ms", required=True, type=int)
args = vars(ap.parse_args())

frameSelector = float(args['time'])
cap = cv2.VideoCapture(args['file'])

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameSelector)      # Go to the 1 sec. position
    ret, frame = cap.read()

    if frame is None:
        break

    cv2.imshow("Frame", frame)           # Displays the frame on screen
    key = cv2.waitKey(0)

    name = args['file'].split("\\")[-1]
    name = name[:8]
    print(name)

    if key == ord('s'):
        cv2.imwrite("data/frames/frame" + name + "-" + str(frameSelector) + ".jpg", frame)
    elif key == 2424832:
        frameSelector -= 1
    elif key == 2555904:
        frameSelector += 1
    elif key == 2621440:
        frameSelector -= 29
    elif key == 2490368:
        frameSelector += 29
    else:
        break
