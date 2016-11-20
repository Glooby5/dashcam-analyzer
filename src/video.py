import argparse
import os

import cv2

from src.lib import data_generator

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the video", required=True)
ap.add_argument("-p", "--positives", help="path to positives folder", default="data/positives")
ap.add_argument("-n", "--negatives", help="path to negatives folder", default="data/negatives")
args = vars(ap.parse_args())


with open(args['file'], "r") as ins:
    for line in ins.readlines():
        print("Reading: " + line)

        generator = data_generator.DataGenerator(line.rstrip(), args['positives'], args['negatives'])
        totalTime = generator.generate()

        print("Positives: " + str(generator.positives))
        print("Negatives: " + str(generator.negatives))
        print('Total time:' + str(totalTime) + os.linesep)

cv2.destroyAllWindows()
