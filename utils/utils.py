import sys
import time
from itertools import tee

def find_argv(name, default = None):
    param_name = "--" + name
    if len(sys.argv) < 2:
        return default
    params = sys.argv[1:]
    p = list(filter(lambda e: e.startswith(param_name), params))
    if len(p) == 0:
        return default
    p = p[0]
    val = p.split("=")[1]
    return val

#From itertools recipes https://docs.python.org/3/library/itertools.html
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Timer:
    def __init__(self, start = True):
        self.start_time = None
        self.end_time = None
        if start:
            self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def get(self):
        return self.end_time - self.start_time

class TimerBox:
    def __init__(self):
        self.timers = {}

    def start(self, timer_name):
        self.timers[timer_name] = Timer()

    def stop(self, timer_name):
        self.timers[timer_name].stop()

    def get(self, timer_name):
        return self.timers[timer_name].get()

timers = TimerBox()

