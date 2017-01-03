import cv2
from src.analyzer.frame import Frame

MAX_WIDTH = 150


class VideoProcessor:
    def __init__(self, cascade_file_name, video_file_name):
        self.video = cv2.VideoCapture(video_file_name)
        self.cascade = cv2.CascadeClassifier(cascade_file_name)
        self.frame_number = 0

    def get_next(self):
        ret, image = self.video.read()

        if self.frame_number % 2:
            self.frame_number += 1
            yield None

        frame = Frame(image)

        sign_hits = self.cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20))

        for (x, y, w, h) in sign_hits:
            if w > MAX_WIDTH or h > MAX_WIDTH:
                continue

            if w < 0 or h < 0:
                continue

            frame.add_sign([x, y, w, h])

            yield frame

"""
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

        if w < 0 or h < 0:
            continue

        #print(str(x) + ':' + str(y) + ' - ' + str(w) + ':' + str(h))
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        #ßcv2.imshow('img', frame[y:y + h, x:x + h])
    #/cv2.imshow('test', frame)
        cv2.imwrite('out/sign' + str(frameCount) + '.jpg', frame[y:y + h, x:x + h])

    k = cv2.waitKey(30) & 0xff
    frameCount += 1

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
"""
