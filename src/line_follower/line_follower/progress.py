"""Persistent directed progress, independent of the replaceable camera cache."""
import math
import numpy as np


class RouteProgress:
    def __init__(self):
        self.points = []
        self.arc = []
        self.blocked = False
        self.phase = 'following'
        self.entry_heading = None
        self.entry_arc = None

    def record(self, pose):
        point = np.array(pose[:2],dtype=float)
        distance = 0. if not self.points else float(np.linalg.norm(point-self.points[-1]))
        if not self.points or distance>=.015:
            self.points.append(point)
            self.arc.append((self.arc[-1] if self.arc else 0.)+distance)
            while len(self.points)>1200:
                self.points.pop(0)
                self.arc.pop(0)
        if self.phase == 'entering' and self.entry_heading is not None:
            angle = math.atan2(math.sin(pose[2]-self.entry_heading),math.cos(pose[2]-self.entry_heading))
            if abs(angle)>.5:
                self.phase = 'turning'

    def observe(self, pose, round_path, straight_exit):
        if round_path and self.phase == 'following':
            self.phase = 'entering'
            self.entry_heading = pose[2]
            self.entry_arc = self.arc[-1] if self.arc else 0.
        elif self.phase == 'turning' and straight_exit:
            self.phase = 'exiting'
        elif self.phase == 'exiting' and not round_path:
            self.phase = 'following'
            self.entry_heading = self.entry_arc = None

    def reverses(self, route):
        """Reject >=8 cm of reverse overlap with already travelled geometry.

        Ignore the most recent 25 cm so a normal bend is not a reversal.
        A perpendicular crossing and a distinct parallel return arm are valid.
        """
        if len(self.points)<3 or len(route)<2:
            return False
        old = np.array(self.points)
        ends = np.asarray(self.arc[1:])
        keep = ends < self.arc[-1]-.25
        starts = old[:-1][keep]
        edges = np.diff(old,axis=0)[keep]
        if not len(starts):
            return False
        lengths2 = np.sum(edges*edges,axis=1)
        cumulative = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        if cumulative[-1]<.08:
            return False
        arc = np.arange(0.,cumulative[-1],.02)
        samples = np.column_stack([np.interp(arc,cumulative,route[:,i]) for i in (0,1)])
        if len(samples)<5:
            return False
        delta = samples[:,None,:]-starts[None,:,:]
        t = np.clip(np.sum(delta*edges[None,:,:],axis=2)/np.maximum(lengths2,1e-10),0.,1.)
        distance = np.linalg.norm(delta-t[:,:,None]*edges[None,:,:],axis=2)
        direction = np.diff(samples,axis=0)
        direction /= np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-9)
        old_direction = edges/np.maximum(np.sqrt(lengths2)[:,None],1e-9)
        opposite = direction@old_direction.T < -.7
        matches = np.any((distance[:-1]<.025)&opposite,axis=1)
        consecutive = 0
        for hit in matches:
            consecutive = consecutive+1 if hit else 0
            if consecutive>=4:
                return True
        return False
