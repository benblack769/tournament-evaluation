import sys
from collections import defaultdict
import os

fname = sys.argv[1]
fold_name = fname.split(".")[0]+"_split_data"
os.mkdir(fold_name)

class default_file:
    def __init__(self):
        self.name = None
        self.file = None
        self.num_lines = 0
    def update(self, name, line):
        if self.name is None:
            self.name = name
            self.file = open(os.path.join(fold_name, name+".csv"), mode='w')
            self.file.write(line1)
        if self.num_lines < 10000:
            self.file.write(line)
        self.num_lines += 1

events = defaultdict(default_file)
with open(fname, mode='r') as csv_file:
    line1 = csv_file.readline()
    assert line1.startswith("Event") or line1.startswith("event")
    for line in csv_file:
        event = line.split(",")[0]
        events[event].update(event,line)
