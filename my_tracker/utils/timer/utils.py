from collections import deque
import time


class FrameRateCounter:
    """
    Class to count the frame rate of a process based on the time stamps of the frames.
    """
    def __init__(self, window=5):
        self.timestamps = deque()
        self.window = window

    def step(self):
        self.timestamps.append(time.time())

    def value(self):
        now = time.time()
        while len(self.timestamps) > 0 and (now - self.timestamps[0] > self.window):
            self.timestamps.popleft()
        return len(self.timestamps) / self.window


class Timer:
    """
    Class to measure the time taken by a process.
    """
    def __init__(self):
        self.t = time.time()

    def start(self):
        self.t = time.time()

    def elapsed(self):
        return time.time() - self.t

    def fetch_restart(self):
        diff = self.elapsed()
        self.start()
        return diff

    def print_restart(self, callname=""):
        print("Call ({}) took {:.5f} seconds.".format(
            callname, time.time() - self.t))
        self.t = time.time()
