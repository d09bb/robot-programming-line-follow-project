"""Bounded odometry-compensated approach after a terminal bar is confirmed."""
import math
import numpy as np
from .vision import terminal_target


class TerminalApproach:
    def __init__(self, max_speed=.08):
        self.max_speed = min(.08,max_speed)
        self.state = 'idle'
        self.goal = None
        self.direction = None
        self.started = None
        self.timeout = None
        self.remaining = None
        self.reason = ''

    def begin(self, mask, pose, now, progress):
        measured = terminal_target(mask)
        if measured is None or progress.blocked:
            return False
        point,direction = measured
        c,s = math.cos(pose[2]),math.sin(pose[2])
        rotation = np.array([[c,-s],[s,c]])
        goal = np.array(pose[:2])+rotation@point
        if progress.reverses(np.vstack([pose[:2],goal])):
            return False
        self.goal = goal
        self.direction = rotation@direction
        self.started = now
        self.remaining = float(point@direction)
        self.timeout = max(8.,self.remaining/.04+4.)
        self.state = 'approaching'
        self.reason = ''
        return True

    def command(self, pose, now, progress):
        progress.record(pose)
        delta = self.goal-np.array(pose[:2])
        self.remaining = float(delta@self.direction)
        normal = np.array([-self.direction[1],self.direction[0]])
        lateral = float(delta@normal)
        if self.state == 'failed':
            return 0.,0.
        if (progress.blocked or progress.reverses(np.vstack([pose[:2],self.goal]))):
            progress.blocked = True
            self.reason = 'route_reversal_stop'
        elif now-self.started>self.timeout or now<self.started:
            self.reason = 'terminal_approach_timeout'
        elif abs(lateral)>.12:
            self.reason = 'terminal_approach_off_path'
        if self.reason:
            self.state = 'failed'
            return 0.,0.
        # Stop the axle within 1.5 cm of the bar, not when it first appears.
        if self.remaining<=.015:
            self.state = 'complete'
            return 0.,0.
        target = self.goal-self.direction*max(0.,self.remaining-.08)
        local = target-np.array(pose[:2])
        c,s = math.cos(pose[2]),math.sin(pose[2])
        forward,left = c*local[0]+s*local[1],-s*local[0]+c*local[1]
        if forward<-.02:
            self.state = 'failed'
            self.reason = 'terminal_target_behind'
            return 0.,0.
        curvature = 2.*left/max(float(local@local),.0025)
        speed = min(self.max_speed,max(.015,.8*(self.remaining-.015)),.6/max(abs(curvature),.01))
        return speed,speed*curvature
