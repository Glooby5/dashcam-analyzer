from .record import Record
import re


class SrtParser:

    def __init__(self, file):
        self.file = file

    def parse(self):
        line_count = 0
        record_number = None
        time = None
        gps = None

        for line in self.file:
            if line_count == 0:
                m = re.search('\d+', line)
                record_number = m.group(0)

            elif line_count == 1:
                time = re.findall(r"\d{2}:\d{2}:\d{2},\d{3}", line)

            elif line_count == 3:
                # print(line)
                gps = re.findall(r"\d+\,\d+", line)
                # print(gps)

            line_count += 1

            if line != "\n":
                continue

            # print("check")
            # print(gps)
            if self.check_values(record_number, time, gps) is False:
                line_count = 0
                # print("next")
                continue

            print(gps)
            # print("save")
            record = Record()
            record.start = time[0]
            record.latitude = gps[0]
            record.longtitude = gps[1]
            record.set_time(record.start)
            line_count = 0

            yield record

        # print(gps)
        record = Record()
        record.start = time[0]
        record.latitude = gps[0]
        record.longtitude = gps[1]
        record.set_time(record.start)

        return record

    def check_values(self, record_number, time, gps):
        if record_number is None:
            return False

        # print("record_number")

        if len(time) != 2:
            return False

        # print("time")
        # print(gps)
        # print("gps2")

        if len(gps) != 2:
            return False

        # print("gps")

        return True
