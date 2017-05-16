import cv2
import argparse
import matplotlib.pyplot as plt
MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to the video", required=True)
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
args = vars(ap.parse_args())

cascade = cv2.CascadeClassifier(args['file'])


def is_in_any_sign(signs_positions, position):
    [x, y, w, h] = position

    for sign in signs_positions:
        if x < sign['x1'] and y < sign['y1'] and (x + w) > sign['x2'] and (y + h) > sign['y2']:
            return True

    return False


def is_position_in_sign_hits(sign_hits, position):
    for (x, y, w, h) in sign_hits:
        x2 = x + w
        y2 = y + h

        if position['x1'] > x and position['y1'] > y and position['x2'] < x2 and position['y2'] < y2:
            return True

    return False


signs_sum = 0
frames_sum = 0
negative_sum = 0

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0


values_x = []
values_y = []

values_x.append(0)
values_y.append(0)


with open(args['dataset'], "r") as ins:

    for line in ins.readlines():
        # print("Reading: " + line)

        frames_sum += 1

        line = line.rstrip()
        split = line.split('|')

        if len(split) == 1:
            negative_sum += 1

        image_file = split[0]
        signs_positions = []

        if len(split) == 2:
            signs = split[1].split(',')

            for sign in signs:
                if sign == "":
                    continue

                positions = sign.split('-')

                start = positions[0]
                end = positions[1]

                startSplit = start.split(':')
                endSplit = end.split(':')

                signs_positions.append({
                    'x1': int(startSplit[0]),
                    'y1': int(startSplit[1]),
                    'x2': int(endSplit[0]),
                    'y2': int(endSplit[1]),
                })

                signs_sum += 1

        image = cv2.imread(image_file)

        sign_hits = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=1, minSize=(15, 15))

        for (x, y, w, h) in sign_hits:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for sign in signs_positions:
            if is_position_in_sign_hits(sign_hits, sign):  # Pokud tuto pozici detektor zaznamenal, tak TP + 1
                true_positive += 1
            else:  # Pokud tuto pozici detektor NEzaznamenal, tak FN + 1
                false_negative += 1


        for (x, y, w, h) in sign_hits:
            if is_in_any_sign(signs_positions, [x, y, w, h]) is False:  # Detektor něco detekoval, ale na framu nic takového není
                false_positive += 1

        if len(sign_hits) == 0 and len(signs_positions) == 0: # Na framu nic není a detwktor taky řekl, že tam nic není
            true_negative += 1


        if len(sign_hits) > 0:
            x = true_positive / 92
            y = false_positive / 1570
            print("{0:.4f}".format(x) + ";" + "{0:.4f}".format(y))
            values_x.append(x)
            values_y.append(y)


print("TP: " + str(true_positive))
print("FP: " + str(false_positive))
print("FN: " + str(false_negative))
print("TN: " + str(true_negative))
print("Signs SUM: " + str(signs_sum))
print("Frames SUM: " + str(frames_sum))
print("Negative SUM: " + str(negative_sum))

print("TP: " + "{0:.2f} %".format(100 * true_positive / signs_sum))
print("FP: " + "{0:.2f} %".format(100 * false_positive / frames_sum))
print("FN: " + "{0:.2f} % ".format(100 * false_negative / signs_sum))
print("TN: " + "{0:.2f} %".format(100 * true_negative / negative_sum))


values_x.append(1)
values_y.append(1)


plt.plot(values_y, values_x)
plt.show()

cv2.destroyAllWindows()

