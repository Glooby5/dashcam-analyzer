import argparse
import cv2
import numpy as np
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) + '/analyzer')
import threshold_sign

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to dataset file", required=True)
args = vars(ap.parse_args())

responses = []
samples = np.empty((0, 900))

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
        dir_class = line.rstrip().split(':')[0]
        path = line.rstrip().split(':')[1]

        image = cv2.imread(path)

        if image is None:
            continue

        sign = threshold_sign.ThresholdSign(image)

        sample = sign.calculate_features()
        samples = np.append(samples, sample, 0)
        responses.append(mapping_write[dir_class])

print("training complete")

np.savetxt('sign-classification-samples.data', samples)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
np.savetxt('sign-classification-responses.data', responses)
