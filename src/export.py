import cv2
import argparse
from analyzer import video_processor
from collections import deque
from parser import srt_parser
import csv
import time
from analyzer import smart_sign
import os

MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
ap.add_argument("-d", "--data", help="path to the srt file", required=True)
args = vars(ap.parse_args())

video_processor = video_processor.VideoProcessor(args['file'], args['video'])
signs = []
frames = deque(maxlen=15)

srt_records = []

file = open(args['data'], 'r')
parserObjet = srt_parser.SrtParser(file)

items = parserObjet.parse()

for srt_record in items:
    srt_records.append(srt_record)

csvfile = open('export.csv', 'w')
spamwriter = csv.DictWriter(csvfile, fieldnames=['filename', 'type', 'value', 'latitude', 'longtitude'])
spamwriter.writeheader()


mapping = {
            1: "Zákaz vstupu chodců",
            2: "Zákaz vjezdu jízdních kol",
            3: "Zákaz vjezdu nákladních automobilů",
            4: "Z. v. voz. nebo souprav, jejichž délka přesahuje vyznačenou mez",
            5: "Zákaz předjíždění",
            6: "Nejvyšší povolená rychlost",
            7: "Zákaz stání",
            8: "Zákaz odbočování vlevo",
            9: "Zákaz odbočování vpravo",
            10: "Zákaz zastavení",
            11: "Zákaz vjezdu všech vozidel(v obou směrech)",
            12: "Z. v. voz., jejichž šířka přesahuje vyznačenou mez",
            13: "Zákaz vjezdu. voz., jejichž hmot. přesahuje vyznačenou mez"
        }


def export_sign():
    if len(frames) < 10:
        return

    #chtělo by to všechno posunout dopředu abych mohl porvnávat i zpětně,mít 9 framů a kontrolovat to na otm uprostřed
    if len(frames[9].signs) == 0 and len(frames[8].signs) > 0:

        if has_next_n():
            return

        exported = []

        for i, sign in enumerate(frames[8].signs):
            srt_record = get_srt_record(frame.time)
            type = None

            if isinstance(sign, smart_sign.SmartSign):
                type = 6

                if sign.value is None:
                    previous_sign = get_value_recursive(7)

                    if previous_sign is not None:
                        sign = previous_sign

            else:
                type = sign.type

            if type in exported:
                continue

            directory = 'exporting/'

            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = '/sign' + str(i) + args['video'].split('/')[-1] + str(frame.number) + '.jpg'
            cv2.imwrite(directory + filename, sign.sign.original)

            type = mapping[type]
            value = {'filename': filename, 'type': type, 'value': sign.value, 'latitude': srt_record.latitude.replace(',', '.'), 'longtitude': srt_record.longtitude.replace(',', '.')}
            print(value)
            spamwriter.writerow(value)
            exported.append(type)


def get_value_recursive(index):
    while index >= 0:
        sign = get_sign_from_frame(frames[index], "rychlost")

        if sign is not None and sign.value:
            return sign

        index -= 1

    return None


def get_sign_from_frame(frame, type):
    for i, sign in enumerate(frame.signs):
        sign_type = None

        if isinstance(sign, smart_sign.SmartSign):
            sign_type = "rychlost"
        else:
            sign_type = sign.mapping[sign.type]

        if sign_type == type:
            return sign

    return None



def get_srt_record(time):
    i = 0

    for srt_record in srt_records:
        if time > srt_record.time and i + 1 < len(srt_records) and time < srt_records[i + 1].time:
            return srt_record

        i += 1

    return False


def has_next_n():
    # for frame in frames[2:]:
    for frame in [val for i, val in enumerate(frames) if i > 9]:

        for sign in frame.signs:
            # if sign.value is not None:
            # if frame.fake is False:
            return True

    return False

start = time.time()

while True:
    frame = video_processor.get_next()

    if frame is False:
        break

    for sign in frame.signs:

        [x, y, w, h] = sign.get_position()

        if sign.value is not None:
            print(str(frame.number) + ': ' + str(sign.value))

    export_sign()

    frames.append(frame)

print("elapsed" + str(time.time() - start))

csvfile.close()