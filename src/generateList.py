import cv2
import os


def createNegativesInfo(path, outputFilename):
    for img in os.listdir(path):
        line = path + '/' + img + '\n'
        with open(outputFilename, 'a') as f:
            f.write(line)


def createPositivesInfo(path, outputFilename):
    for imageFileName in os.listdir(path):
        image = cv2.imread(path + '/' + imageFileName)

        height, width = image.shape[:2]
        line = path + '/' + imageFileName + ' 1 0 0 ' + str(width) + ' ' + str(height) + os.linesep

        with open(outputFilename, 'a') as f:
            f.write(line)


def resizeImages(path, newWidth, newHeight):
    for imageFileName in os.listdir(path):
        image = cv2.imread(path + '/' + imageFileName)

        try:
            height, width = image.shape[:2]
        except Exception:
            continue

        if width < newWidth:
            continue

        if height < newHeight:
            continue

        resized = cv2.resize(image, (newWidth, newHeight))
        cv2.imwrite(path + '/' + imageFileName, resized)


resizeImages('data/negatives', 1280, 720)
createPositivesInfo('data/positives', 'info.dat')
createNegativesInfo('data/negatives', 'bg.txt')
