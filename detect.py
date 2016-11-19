import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
args = vars(ap.parse_args())

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# this is the cascade we just made. Call what you want
watch_cascade = cv2.CascadeClassifier(args['file'])

cap = cv2.VideoCapture(args['video'])
frameCount = 0

while 1:
    if frameCount % 5:
        continue

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # add this
    # image, reject levels level weights.
    watches = watch_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20))

    # add this
    for (x, y, w, h) in watches:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()