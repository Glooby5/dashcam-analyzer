import argparse
import cv2
import numpy as np
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) + '/analyzer')
import threshold_sign
import crop_sign
import sign
import smart_sign


samples = np.loadtxt('general-samples.data', np.float32)
responses = np.loadtxt('general-responses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

image = cv2.imread("sign02017_04_13_16_53_16.mp42785.jpg")

sign = sign.Sign(image, None)
sign.crop_sign()

cv2.imshow("sign", sign.image)
cv2.waitKey(0)

classify_sign = smart_sign.SmartSign(sign, model)
print(classify_sign.classify())

