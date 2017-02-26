from .record import Record
import re


class SrtParser:

    def parse(self, file):
        line_count = 0
        record_number = None
        time = None
        gps = None

        for line in file:
            if line_count == 0:
                m = re.search('\d+', line)
                record_number = m.group(0)

            elif line_count == 1:
                time = re.findall(r"\d{2}:\d{2}:\d{2},\d{3}", line)

            elif line_count == 3:
                gps = re.findall(r"\d+,\d+", line)

            line_count += 1

            if line != "\n":
                continue

            if self.check_values(record_number, time, gps) is False:
                line_count = 0
                continue

            record = Record()
            record.start = time[0]
            record.latitude = gps[0]
            record.longitude = gps[1]

            yield record

    def check_values(self, record_number, time, gps):
        if record_number is None:
            return False

        if len(time) != 2:
            return False

        if len(gps) != 2:
            return False

        return True
