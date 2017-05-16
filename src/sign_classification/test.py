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
    10: "zastaveni",
    11: "vjezd",
    12: "zuzeni",
    13: "hmotnost"
}

mapping_write = {v: k for k, v in mapping.items()}

mapping_final = {
    1: "Zákaz vstupu chodců",
    2: "Zákaz vjezdu jízdních kol",
    3: "Zákaz vjezdu nákladních automobilů",
    4: "Omezení rozměrů nákladních automobilů",
    5: "Zákaz předjíždění",
    6: "Nejvyšší povolená rychlost",
    7: "Zákaz stání",
    8: "Zákaz odbočování vlevo",
    9: "Zákaz odbočování vpravo",
    10: "Zákaz zastavení",
    11: "Zákaz vjezdu",
    12: "Omezení rozměrů",
    13: "Nejvyšší povolená hmotnost"
}



classes = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0
}

classes_correct = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0
}

default = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0
}


confusion_matrix = default.copy()

for i in range(1, 14):
    confusion_matrix[i] = default.copy()




with open(args['dataset'], "r") as ins:
    for line in ins.readlines():
        # print("Reading: " + line)

        dir_class = line.rstrip().split(':')[0]
        path = line.rstrip().split(':')[1]

        image = cv2.imread(path)
        # cv2.imshow("Fsdf", image)

        if image is None:
            continue

        sign = threshold_sign.ThresholdSign(image)
        roi_small = sign.calculate_features()

        cropped_sign = crop_sign.CropSign(image)
        cv2.imshow("1", cropped_sign.image)
        # cv2.waitKey(0)


        th_sign = threshold_sign.ThresholdSign(cropped_sign.image)

        roi_small = th_sign.calculate_features()

        retval, results, neigh_resp, dists = model.findNearest(roi_small, k=1)

        # print("nalezeno  : " + str(retval) + " : " + str(neigh_resp) + " : " + str(dists))

        # if dists[0][0] < 9000000:
        confusion_matrix[mapping_write[dir_class]][retval] += 1

        all += 1

        if dists[0][0] > 12000000:
            unknown += 1
            # cv2.waitKey(0)

        if retval == mapping_write[dir_class]:
            correct += 1
            classes_correct[mapping_write[dir_class]] += 1
        else:
            incorrect += 1

        classes[mapping_write[dir_class]] += 1

acurracy = 100 * correct / all

print(str(acurracy) + ":" + str(all) + ":" + str(correct) + ":" + str(incorrect)+ ":" + str(unknown))

for i, class_item in classes.items():
    if class_item:
        acurracy = 100 * classes_correct[i] / class_item

        print(str(i) + " & " + mapping_final[i] + " & " + str(class_item) + " & " + str(classes_correct[i]) + " & " + "{0:.2f}".format(acurracy) + "\,\%  \\\\")

print(unknown)
print(classes)
print(classes_correct)

line = " & "

for i in range(1, 14):
    line += str(i)

    if i < 13:
        line += " & "

# print(line)

for i in range(1, 14):

    line = str(i) + " & "

    for x, value in confusion_matrix[i].items():
        if value > 0:
            line += "\\textbf{" + str(value) + "}"
        else:
            line += str(value)

        if x < 13:
            line += " & "

    print(line + " \\\\")

#
# 88.04347826086956:184:162:22:21
# 1: 4, 100.0 %
# 2: 15, 100.0 %
# 3: 25, 100.0 %
# 5: 15, 100.0 %
# 6: 59, 100.0 %
# 7: 6, 33.333333333333336 %
# 8: 14, 85.71428571428571 %
# 9: 8, 50.0 %
# 10: 20, 65.0 %
# 12: 8, 75.0 %
# 13: 10, 70.0 %
# {1: 4, 2: 15, 3: 25, 4: 0, 5: 15, 6: 59, 7: 6, 8: 14, 9: 8, 10: 20, 11: 0, 12: 8, 13: 10}
# {1: 4, 2: 15, 3: 25, 4: 0, 5: 15, 6: 59, 7: 2, 8: 12, 9: 4, 10: 13, 11: 0, 12: 6, 13: 7}
