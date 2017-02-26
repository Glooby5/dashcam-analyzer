class Record:

    def __init__(self):
        self.start = None
        self.time = None
        self.latitude = None
        self.longtitude = None

    def set_time(self, start):
        values = start.split(':')
        time = 0

        time += int(values[0]) * 3600
        time += int(values[1]) * 60
        time += float(values[2].replace(',', '.'))

        self.time = time
