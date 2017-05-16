import argparse
import cv2
import numpy as np
import sys
import os.path
import operator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from sign import Sign
from smart_sign import SmartSign
from crop_sign import CropSign

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to dataset file", required=True)
args = vars(ap.parse_args())

samples = np.loadtxt('general-samples.data', np.float32)
responses = np.loadtxt('general-responses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

all = 0
correct = 0
incorrect = 0
unknown = 0

classes = {
    30: 0,
    40: 0,
    50: 0,
    60: 0,
    70: 0,
    80: 0,
    90: 0,
    100: 0,
    130: 0,
}

classes_correct = {
    30: 0,
    40: 0,
    50: 0,
    60: 0,
    70: 0,
    80: 0,
    90: 0,
    100: 0,
    130: 0,
}


with open(args['dataset'], "r") as ins:
    prev = None

    for line in ins.readlines():
        dir_class = line.rstrip().split(':')[0]
        path = line.rstrip().split(':')[1]

        image = cv2.imread(path)

        if image is None:
            continue

        sign = Sign(image, None)
        sign.crop_sign()
        # cropped_sign = CropSign(image)

        cv2.imshow("crop_sign", sign.image)
        classify_sign = SmartSign(sign, model)
        value = classify_sign.classify()

        all += 1

        if value == int(dir_class):
            correct += 1
            classes_correct[int(dir_class)] += 1
        else:
            incorrect += 1
            # cv2.waitKey(0)

        classes[int(dir_class)] += 1

        if prev != dir_class:
            print(dir_class + ": 0,")
            prev = dir_class


acurracy = 100 * correct / all

print(str(acurracy) + ":" + str(all) + ":" + str(correct) + ":" + str(incorrect)+ ":" + str(unknown))

items = sorted(classes.items(), key=operator.itemgetter(0))

for i, class_item in items:
    if class_item:
        acurracy = 100 * classes_correct[i] / class_item

        print(str(i) + " & " + str(class_item) + " & " + str(classes_correct[i]) + " & " + "{0:.2f}".format(acurracy) + "\,\%  \\\\")

# 91.42857142857143:105:96:9:0

