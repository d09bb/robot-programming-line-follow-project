"""Pure image processing; no ROS or simulator state dependencies."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Observation:
    error: float
    heading: float
    confidence: float
    terminal: bool
    mask: np.ndarray
    points: list
    preview: float = 0.
    bend: float = 0.


def _terminal_row(mask):
    """A nearby transverse bar ends the connected incoming line.

    Look across the whole image for continuation connected to that bar;
    an off-centre or curved continuation is still a crossing, not a finish.
    """
    h,w = mask.shape
    for row in range(int(h*.97),int(h*.84)-1,-1):
        active = np.flatnonzero(mask[row])
        groups = np.split(active,np.where(np.diff(active)>1)[0]+1)
        wide = [group for group in groups if len(group)>w*.42]
        if not wide:
            continue
        group = max(wide,key=len)
        _,labels = cv2.connectedComponents(mask,8)
        component = labels[row,int(group[len(group)//2])]
        connected = labels==component
        top = row
        while top>int(h*.54) and np.count_nonzero(connected[top-1])>w*.42:
            top -= 1
        # Ignore the bar's thickness and short thinning/aliasing hooks.
        beyond = connected[:max(0,top-int(h*.06))]
        return (top+row)/2 if np.count_nonzero(beyond)<max(8,int(h*.02)) else None
    return None


def terminal_bar(mask):
    return _terminal_row(mask) is not None


def terminal_target(mask):
    """Intersect the observed incoming stem with the terminal bar on the floor."""
    row = _terminal_row(mask)
    if row is None:
        return None
    h,w = mask.shape
    focal = w/(2*np.tan(1.085595/2))
    points = []
    previous = w/2
    for y in range(h-6,int(row)+8,-4):
        active = np.flatnonzero(mask[y])
        groups = np.split(active,np.where(np.diff(active)>1)[0]+1)
        groups = [g for g in groups if 3<=len(g)<=w*.25]
        if not groups:
            continue
        group = min(groups,key=lambda g:abs(float(g.mean())-previous))
        previous = float(group.mean())
        depth = .103*focal/(y-h/2)
        points.append((.076+depth,(w/2-previous)*depth/focal))
    if len(points)<3:
        return None
    points = np.array(points)
    slope,offset = np.polyfit(points[:,0],points[:,1],1)
    forward = .076+.103*focal/(row-h/2)
    left = slope*forward+offset
    if not .15<forward<.7 or abs(left)>.15 or abs(slope)>.7:
        return None
    return np.array([forward,left]),np.array([1.,slope])/np.hypot(1.,slope)


def observe(bgr, threshold=65):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = np.uint8(gray < threshold) * 255
    mask[:int(h * .54)] = 0  # horizon and remote track segments
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    points = []
    previous = w / 2
    broad = []
    # Start nearest the robot; follow the same branch toward the horizon.
    for y in np.linspace(h * .97, h * .57, 21).astype(int):
        active = np.flatnonzero(np.count_nonzero(mask[y-2:y+3], axis=0) >= 3)
        groups = np.split(active, np.where(np.diff(active) > 1)[0] + 1)
        # Perspective makes nearby oblique bends much wider than straight lines.
        width_limit = .85 if y >= h * .78 else .32
        candidates = [g for g in groups if 3 <= len(g) <= w * width_limit]
        broad.append(any(len(g) > w * .42 for g in groups))
        if not candidates:
            continue
        group = min(candidates, key=lambda g: abs(float(g.mean()) - previous))
        center = float(group.mean())
        if points and abs(center - previous) > w * .22:
            continue
        points.append((center, int(y)))
        previous = center
    if len(points) < 2:
        return Observation(0., 0., 0., False, mask, points)
    # Require actual near-field support; distant segments alone must not steer.
    local = [(x, y) for x, y in points if y >= h * .80]
    if len(local) < 2:
        return Observation(0., 0., 0., False, mask, points)
    near = np.mean([x for x, _ in local[:2]])
    far = np.mean([x for x, _ in points[-3:]])
    # Image right is negative ROS yaw (clockwise).
    error = (w / 2 - near) / (w / 2)
    # Fit only the nearby tangent. Remote bends are a speed preview, not yaw.
    tangent = local[:3]
    slope = np.polyfit([y / h for _, y in tangent],
                       [x / (w / 2) for x, _ in tangent], 1)[0]
    heading = float(np.clip(slope * .10, -.35, .35)) if abs(slope) > 1e-6 else 0.
    # Look ahead only within the near field. A tight bend can leave the bottom
    # centered until it is too late to turn; remote track still cannot steer.
    bend = float(np.clip((near - np.mean([x for x, _ in local[-2:]])) /
                         (w / 2), -1., 1.))
    preview = max(abs(near - far) / (w / 2),
                  (max(x for x, _ in points) - min(x for x, _ in points)) / (w / 2))
    # A transverse near bend can be too wide for row-center candidates.
    # Use it for braking before it reaches the steering lookahead.
    if any(broad[:14]):
        preview = max(preview, .8)
    terminal = any(broad[:7]) and terminal_bar(mask)
    return Observation(float(error), float(heading), max(.30, len(points)/21.), terminal, mask, points, float(preview), bend)


class Controller:
    def __init__(self, max_speed=.26, kp=1.6, kd=.06, heading_gain=.65):
        self.max_speed, self.kp, self.kd, self.heading_gain = max_speed, kp, kd, heading_gain
        self.previous = None
        self.filtered = 0.
        self.speed = 0.
        self.yaw = 0.

    def command(self, obs, dt):
        if obs.confidence < .23:
            self.previous = None
            self.filtered = 0.
            self.speed = 0.
            self.yaw = 0.
            return 0., 0.
        # Bound the control interval after image gaps and filter in real time.
        dt = float(np.clip(dt, .01, .1))
        # Aim at the continuation of the near line, rather than its bottom
        # intercept. In a bend these can lie on opposite sides of the camera.
        bend = getattr(obs, 'bend', 0.)
        aim = obs.error + bend
        derivative = 0. if self.previous is None else (aim - self.previous) / dt
        self.previous = aim
        alpha = dt / (.10 + dt)
        self.filtered += alpha * (derivative - self.filtered)
        yaw = float(np.clip(self.kp*aim + self.kd*self.filtered +
                            self.heading_gain*obs.heading, -1.2, 1.2))
        # Bound changes between frames to avoid abrupt left/right oscillation.
        self.yaw += float(np.clip(yaw - self.yaw, -2.5 * dt, 2.5 * dt))
        yaw = self.yaw
        preview = getattr(obs, 'preview', 0.)
        target = self.max_speed / (1. + 1.3*abs(yaw) + 1.2*preview + 3.*abs(bend))
        if preview >= .65:
            target = min(target, .08)
        # Crawl through a sharp bend so a static image cannot latch the robot
        # into an endless pivot. Confidence loss still stops immediately.
        if abs(bend) > .45 or abs(obs.error) > .55:
            target = min(target, .035)
        target *= min(1., obs.confidence/.45)
        # Accelerate gently; reductions take effect immediately for bends/loss.
        self.speed = min(target, self.speed + .35 * dt)
        return self.speed, yaw
