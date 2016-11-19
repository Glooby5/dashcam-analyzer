import argparse
import sign_detector
import cv2
import time
import os


def create_pos_n_neg():
    for file_type in ['data/negatives']:

        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type + '/' + img + ' 1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)
            elif file_type == 'data/negatives':
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)


def resizeImages(path):
    for imageFileName in os.listdir(path):
        image = cv2.imread(path + '/' + imageFileName)

        try:
            height, width = image.shape[:2]

            if height < 720:
                continue

            if width < 1280:
                continue

        except Exception:
            continue

        resized = cv2.resize(image, (1280, 720))

        cv2.imwrite(path + '/' + imageFileName, resized)

        # line = path + '/' + imageFileName + ' 1 0 0 ' + str(width) + ' ' + str(height) + '\n'
        #
        # with open('info.dat', 'a') as f:
        #     f.write(line)


def resizePositiveImages(path):
    for imageFileName in os.listdir(path):
        image = cv2.imread(path + '/' + imageFileName)

        try:
            height, width = image.shape[:2]

            if height < 50:
                continue

        except Exception:
            continue

        resized = cv2.resize(image, (50, 50))

        cv2.imwrite(path + '/' + imageFileName, resized)

        # line = path + '/' + imageFileName + ' 1 0 0 ' + str(width) + ' ' + str(height) + '\n'
        #
        # with open('info.dat', 'a') as f:
        #     f.write(line)


def createPositivesInfo(path):
    for imageFileName in os.listdir(path):
        image = cv2.imread(path + '/' + imageFileName)

        height, width = image.shape[:2]

        line = path + '/' + imageFileName + ' 1 0 0 ' + str(width) + ' ' + str(height) + '\n'

        with open('info.dat', 'a') as f:
            f.write(line)

# create_pos_n_neg()
resizeImages('data/negatives')
# resizePositiveImages('data/positives')
# createPositivesInfo('data/positives')

# opencv_createsamples -info info.dat -nuum 445 -w 50 -h 50 -vec samples.vec -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.1
# opencv_traincascade -data training -vec samples.vec -bg bg.txt -numStages 20 -numPos 440 -numNeg 4000 -w 40 -h 40 -featureType LBP
# >opencv_traincascade -data training2 -vec samples.vec -bg bg.txt -numStages 5 -numPos 420 -numNeg 800 -w 50 -h 50 -featureType LBP -maxFa
# lseAlarmRate 0.3

# opencv_traincascaded -data training5 -vec samples.vec -bg bg.txt -numStages 7 -numPos 420 -numNeg 420 -w 50 -h 50 -featureType LBP -maxFalseAlarmRate 0.5
