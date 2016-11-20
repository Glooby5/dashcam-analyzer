import cv2
import argparse

MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
args = vars(ap.parse_args())

cascade = cv2.CascadeClassifier(args['file'])

cap = cv2.VideoCapture(args['video'])
frameCount = 0

while 1:
    ret, frame = cap.read()

    if frameCount % 2:
        frameCount += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    watches = cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20))

    # add this
    for (x, y, w, h) in watches:
        if w > MAX_WIDTH or h > MAX_WIDTH:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('img', frame)
    k = cv2.waitKey(30) & 0xff
    frameCount += 1

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
