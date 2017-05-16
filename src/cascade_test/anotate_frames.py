import os
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", help="path to the directory", required=True)
args = vars(ap.parse_args())

directory = args['directory']


drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
image = None

positions_x = []
positions_y = []
positions_x2 = []
positions_y2 = []

cascade = cv2.CascadeClassifier('../cascade-6.xml')

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        positions_x.append(x)
        positions_y.append(y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(image, (ix, iy), (x, y),(0,255, 0), -1)
        else:
            cv2.circle(image,(x, y), 5, (0, 0, 255), -1)

        positions_x2.append(x)
        positions_y2.append(y)

        cv2.imshow('image', image)


cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

with open('frames' + directory + '.txt', 'w') as f:
    for frame in os.listdir(directory):
        print("Reading: " + frame)
        if "DS" in frame:
            continue

        positions_x = []
        positions_y = []
        positions_x2 = []
        positions_y2 = []

        fullpath = os.path.abspath(directory + '/' + frame)

        image = cv2.imread(directory + '/' + frame)

        if image is None:
            continue

        sign_hits = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=1, minSize=(20, 20))

        for (x, y, w, h) in sign_hits:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('image', image)
        k = cv2.waitKey(0) & 0xFF

        signs = ""

        if k == ord('m'):
            mode = not mode
        elif k == ord('d'):
            for i, x in enumerate(positions_x):
                signs += str(x) + ":" + str(positions_y[i]) + "-" + str(positions_x2[i]) + ":" + str(positions_y2[i]) + ","
        elif k == 27:
            break

        if signs != "":
            line = fullpath + '|' + signs + '\n'
        else:
            line = fullpath + '\n'

        print(line)
        f.write(line)

cv2.destroyAllWindows()
