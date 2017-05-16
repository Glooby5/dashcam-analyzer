import cv2
import numpy as np
import sys
import os.path
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from sign import Sign
from smart_sign import SmartSign


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--signs", help="path to signs file", required=True)
args = vars(ap.parse_args())

samples = np.loadtxt('symbol-classification-samples.data', np.float32)
responses = np.loadtxt('symbol-classification-responses.data', np.float32)
responses = responses.reshape((responses.size, 1))

knn_model = cv2.ml.KNearest_create()
knn_model.train(samples, cv2.ml.ROW_SAMPLE, responses)

count = 0

with open(args['signs'], "r") as ins:
    for line in ins.readlines():
        path = line.rstrip()

        image = cv2.imread(path)

        if image is None:
            continue

        cv2.imshow("sign", image)


        sign = Sign(image, None)
        sign.crop_sign()

        cv2.imshow("crop_sign", sign.image)

        classify_sign = SmartSign(sign, knn_model)
        blobs = classify_sign.get_blobs()

        # count = 0

        for blob in blobs:
            cv2.imshow("blob", blob.image)
            value = blob.get_value()

            print(str(value))

            directory = 'dataset/' + str(value)

            if not os.path.exists(directory):
                os.makedirs(directory)

            cv2.imwrite(directory + '/symbol' + str(count) + '.jpg', blob.image)

            count += 1
            # cv2.waitKey(0)

        # cv2.waitKey(0)

cv2.destroyAllWindows()
