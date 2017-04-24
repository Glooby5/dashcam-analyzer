import argparse
import cv2
import numpy as np
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) + '/analyzer')
import threshold_sign
import crop_sign

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to dataset file", required=True)
args = vars(ap.parse_args())

samples = np.loadtxt('sign-classification-samples.data', np.float32)
responses = np.loadtxt('sign-classification-responses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

all = 0
correct = 0
incorrect = 0
unknown = 0

mapping = {
    1: "chodci",
    2: "kolo",
    3: "nakladak",
    4: "nakladak_rozmer",
    5: "predjizdeni",
    6: "rychlost",
    7: "stani",
    8: "vlevo",
    9: "vpravo",
    10: "zastaveni"
}

mapping_write = {v: k for k, v in mapping.items()}

with open(args['dataset'], "r") as ins:
    for line in ins.readlines():
        print("Reading: " + line)

        dir_class = line.rstrip().split(':')[0]
        path = line.rstrip().split(':')[1]

        image = cv2.imread(path)
        cv2.imshow("Fsdf", image)

        if image is None:
            continue

        sign = threshold_sign.ThresholdSign(image)
        roi_small = sign.calculate_features()

        cropped_sign = crop_sign.CropSign(image)
        th_sign = threshold_sign.ThresholdSign(cropped_sign.image)

        roi_small = th_sign.calculate_features()

        retval, results, neigh_resp, dists = model.findNearest(roi_small, k=2)

        print("nalezeno  : " + str(retval) + " : " + str(neigh_resp) + " : " + str(dists))

        cv2.waitKey(0)

        all += 1

        if dists[0][0] > 12000000:
            unknown += 1
            # cv2.waitKey(0)

        if retval == mapping_write[dir_class]:
            correct += 1
        else:
            incorrect += 1

acurracy = 100 * correct / all

print(str(acurracy) + ":" + str(all) + ":" + str(correct) + ":" + str(incorrect)+ ":" + str(unknown))
