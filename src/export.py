import cv2
import argparse
from analyzer import video_processor
from collections import deque
from parser import srt_parser
import csv

MAX_WIDTH = 150

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the cascade", required=True)
ap.add_argument("-v", "--video", help="path to the video", required=True)
ap.add_argument("-d", "--data", help="path to the srt file", required=True)
args = vars(ap.parse_args())

video_processor = video_processor.VideoProcessor(args['file'], args['video'])
signs = []
frames = deque(maxlen=10)

srt_records = []

file = open(args['data'], 'r')
parserObjet = srt_parser.SrtParser(file)

items = parserObjet.parse()

for srt_record in items:
    srt_records.append(srt_record)

csvfile = open('eggs.csv', 'w')
spamwriter = csv.DictWriter(csvfile, fieldnames=['value', 'latitude', 'longtitude'])
spamwriter.writeheader()


def export_sign():
    if len(frames) < 2:
        return

    if len(frames[1].signs) == 0 and len(frames[0].signs) > 0 and frames[0].signs[0].value is not None and (len(frames[2].signs) == 0 or (len(frames[2].signs) and frames[2].signs[0].value is None)):
        sign = frames[0].signs[0]

        if has_next_n():
            return

        print(str(sign.value))
        srt_record = get_srt_record(frame.time)

        if srt_record:
            print(srt_record.latitude)

        value = {'value': sign.value, 'latitude': srt_record.latitude.replace(',', '.'), 'longtitude': srt_record.longtitude.replace(',', '.')}

        spamwriter.writerow(value)


def get_srt_record(time):
    i = 0

    for srt_record in srt_records:
        if time > srt_record.time and i + 1 < len(srt_records) and time < srt_records[i + 1].time:
            return srt_record

        i += 1

    return False


def has_next_n():
    # for frame in frames[2:]:
    for frame in [val for i, val in enumerate(frames) if i > 2]:
        for sign in frame.signs:
            if sign.value is not None:
                return True

    return False


while True:
    frame = video_processor.get_next()

    export_sign()

    if frame is False:
        break

    frames.append(frame)

csvfile.close()