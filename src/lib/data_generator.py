import time

import cv2

from src.lib import sign_detector


class DataGenerator:
    def __init__(self, filename, positivesPath, negativesPath):
        self.detector = sign_detector.SignDetector()
        self.cap = cv2.VideoCapture(filename)
        self.name = filename.split("\\")[-1].split('.')[0]
        self.positivesPath = positivesPath
        self.negativesPath = negativesPath

        self.positives = 0
        self.negatives = 0
        self.frameCount = 0
        self.start = 0

    def positives(self):
        return self.positives

    def negatives(self):
        return self.negatives

    def generate(self):
        self.start = time.time()

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if frame is None:
                break

            self._processFrame(frame)

        return time.time() - self.start

    def _processFrame(self, frame):
        self.frameCount += 1

        if not self.frameCount % 100:
            self._printStatistics()

        if not self.frameCount % 2 and (self.frameCount % 30):
            return

        self._findSigns(frame)

    def _findSigns(self, frame):
        signs = self.detector.getSignsFromFrame(frame)

        if signs is None and not self.frameCount % 30:
            cv2.imwrite(self.negativesPath + "\\" + self.name + "-" + str(self.frameCount) + ".jpg", frame)
            self.negatives += 1

        if signs is None:
            return

        for sign in signs:
            cv2.imwrite(self.positivesPath + "\\" + self.name + "-" + str(self.frameCount) + ".jpg", sign)
            self.positives += 1

    def _printStatistics(self):
        currentTime = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))

        print('Video time: ', str(currentTime / 1000))
        print('Run time: ' + str(time.time() - self.start))
