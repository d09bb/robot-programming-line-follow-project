"""Interpolate measured odometry at the camera exposure time."""
from collections import deque
import math


class PoseHistory:
    def __init__(self):
        self.samples = deque(maxlen=300)

    def add(self, stamp, pose):
        if self.samples and stamp < self.samples[-1][0]:
            self.samples.clear()
        if self.samples and stamp == self.samples[-1][0]:
            self.samples.pop()
        self.samples.append((stamp, pose))

    def at(self, stamp):
        for (a, p), (b, q) in zip(self.samples, list(self.samples)[1:]):
            if a <= stamp <= b:
                ratio = (stamp-a)/(b-a)
                turn = math.atan2(math.sin(q[2]-p[2]), math.cos(q[2]-p[2]))
                return (p[0]+ratio*(q[0]-p[0]), p[1]+ratio*(q[1]-p[1]), p[2]+ratio*turn)
        if self.samples and stamp == self.samples[-1][0]:
            return self.samples[-1][1]
        return None
