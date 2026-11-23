"""Bounded corner recovery from observed line geometry and measured motion."""
import math
import numpy as np
from .path import ground_centerline, path_turns, projected_support, single_near_line


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CornerRecovery:
    def __init__(self):
        self.state = 'idle'
        self.reason = ''
        self.candidate = None
        self.stable = 0
        self.corner = None
        self.heading = None
        self.direction = 0
        self.confirmed = None
        self.started = None
        self.start_pose = None
        self.previous_pose = None
        self.travel = self.rotation = self.yaw = 0.
        self.consumed = None

    def observe(self, local, pose, now):
        """Require three agreeing observations of an incoming and outgoing arm."""
        if self.state == 'corner_recovery' or local is None or len(local) < 8:
            self.stable = 0
            self.candidate = None
            return
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
        if lengths[-1] < .18:
            self.stable = 0
            self.candidate = None
            return
        near = local[np.searchsorted(lengths, .06)] - local[0]
        far = local[-1] - local[np.searchsorted(lengths, lengths[-1]-.06)]
        near /= max(np.linalg.norm(near), 1e-9)
        far /= max(np.linalg.norm(far), 1e-9)
        turn = wrap(math.atan2(far[1],far[0])-math.atan2(near[1],near[0]))
        # A round/omega path can bend both ways in one image. Its final
        # tangent is not the exit heading of a single right-angle corner.
        turns = path_turns(local)
        opposite = np.sum(np.abs(turns[turns*(1 if turn > 0 else -1) < 0]))
        if opposite > math.radians(30):
            self.stable = 0
            self.candidate = None
            # Do not retain a previous single-corner hypothesis for this path.
            if self.state == 'corner_armed':
                self.state = 'idle'
                self.reason = 'round_path_not_single_corner'
                self.corner = self.confirmed = None
            return
        if not math.radians(45) <= abs(turn) <= math.radians(120) or near[0] < .7:
            self.stable = 0
            self.candidate = None
            return
        # Intersection of the fitted arms estimates where the axle must turn.
        matrix = np.column_stack([near, -far])
        along, _ = np.linalg.solve(matrix, local[-1]-local[0])
        corner = local[0]+along*near
        if not .15 <= corner[0] <= .65 or abs(corner[1]) > .18:
            self.stable = 0
            self.candidate = None
            return
        c,s = math.cos(pose[2]),math.sin(pose[2])
        corner = corner @ np.array([[c,s],[-s,c]]) + pose[:2]
        heading = wrap(pose[2]+math.atan2(far[1],far[0]))
        direction = 1 if turn > 0 else -1
        if self.consumed is not None and np.linalg.norm(corner-self.consumed) < .25:
            return
        if (self.candidate is not None and np.linalg.norm(corner-self.candidate[0]) < .06
                and abs(wrap(heading-self.candidate[1])) < .2
                and direction == self.candidate[2]):
            self.stable += 1
        else:
            self.stable = 1
        self.candidate = (corner, heading, direction)
        if self.stable >= 3:
            self.corner, self.heading, self.direction = self.candidate
            self.confirmed = now
            self.state = 'corner_armed'
            self.reason = ''

    def begin(self, pose, now, stop_reason):
        visual_conflict = stop_reason == 'path_visual_conflict'
        if self.state != 'corner_armed' or stop_reason not in (
                'path_missing_or_expired', 'path_end', 'target_behind', 'path_visual_conflict'):
            return False
        remaining = self.direction*wrap(self.heading-pose[2])
        if (now-self.confirmed > 6. or now < self.confirmed
                or np.linalg.norm(np.array(pose[:2])-self.corner) > (.25 if visual_conflict else .20)
                or (visual_conflict and not math.radians(5) <= remaining <= math.radians(30))
                or not -.05 <= remaining <= math.radians(110)):
            self.cancel('corner_evidence_unusable')
            return False
        self.state = 'corner_recovery'
        self.reason = ''
        self.started = now
        self.start_pose = np.array(pose)
        self.previous_pose = np.array(pose)
        self.travel = self.rotation = self.yaw = 0.
        self.consumed = self.corner.copy()
        return True

    def begin_before_path_end(self, tracker, pose, now):
        if (self.state != 'corner_armed' or tracker.round_tracking
                or tracker.progress.blocked or tracker.stop_reason
                or tracker.forward_limit_reason != 'short_cached_corner'
                or tracker.update_reason != 'observation_too_short'):
            return False
        remaining = self.direction*wrap(self.heading-pose[2])
        if not math.radians(5) <= remaining <= math.radians(20):
            return False
        if np.linalg.norm(np.array(pose[:2])-self.corner) > .15:
            return False
        return self.begin(pose, now, 'path_end')

    def command(self, pose, now, dt):
        if self.state != 'corner_recovery':
            return 0., 0.
        current = np.array(pose)
        self.travel += float(np.linalg.norm(current[:2]-self.previous_pose[:2]))
        self.rotation += abs(wrap(current[2]-self.previous_pose[2]))
        self.previous_pose = current
        if (now < self.started or now-self.started >= 3.
                or self.travel >= .12 or self.rotation >= math.radians(110)):
            self.cancel('corner_recovery_limit')
            return 0., 0.
        remaining = self.direction*wrap(self.heading-pose[2])
        if remaining <= -.12:
            self.cancel('corner_heading_limit')
            return 0., 0.
        # Pivot at the axle. Never move forward blindly or infer turn direction
        # from the sign of a noisy last steering command.
        desired = self.direction*min(.6, max(.2, .9*max(remaining,0.)))
        step = 3.*float(np.clip(dt,.01,.1))
        self.yaw += float(np.clip(desired-self.yaw,-step,step))
        return 0., self.yaw

    def cancel(self, reason):
        if self.corner is not None:
            self.consumed = self.corner.copy()
        self.state = 'idle'
        self.reason = reason
        self.corner = self.confirmed = self.candidate = None
        self.stable = 0
        self.yaw = 0.


class ConflictRecovery:
    """One bounded stationary look toward a valid remembered exit tangent."""
    def __init__(self):
        self.state = 'idle'
        self.reason = ''
        self.heading = None
        self.started = None
        self.origin = None
        self.previous_heading = None
        self.rotation = 0.
        self.yaw = 0.
        self.consumed = None
        self.consumed_arc = 0.
        self.support_frames = 0
        self.support_stamp = None
        self.kind = 'visual_conflict'
        self.hold_end = False
        self.remaining_at_start = None
        self.exit_reference = None
        self.start_heading = None
        self.view_candidate = None
        self.view_frames = 0
        self.view_adjusted = False

    def begin(self, tracker, pose, now):
        if tracker.progress.blocked:
            self.cancel('route_reversal_stop')
            return False
        if self.state=='conflict_recovery':
            return True
        position = np.array(pose[:2])
        travelled = tracker.progress.arc[-1] if tracker.progress.arc else 0.
        # Nearby arms may be several metres apart along the route.
        repeated = (self.consumed is not None and np.linalg.norm(position-self.consumed)<.25
                    and travelled-self.consumed_arc<.50)
        end_request = tracker.round_tracking and tracker.stop_reason=='path_end'
        if (tracker.progress.blocked or (not tracker.conflict_latched and not end_request)
                or tracker.points is None or len(tracker.points)<3
                or tracker.updated is None or not 0.<=now-tracker.updated<=tracker.memory_seconds
                or repeated):
            return False
        points = tracker.points
        edges = np.diff(points,axis=0)
        fraction = np.clip(np.sum((position-points[:-1])*edges,axis=1)
                           /np.maximum(np.sum(edges*edges,axis=1),1e-10),0.,1.)
        projections = points[:-1]+fraction[:,None]*edges
        index = int(np.argmin(np.linalg.norm(projections-position,axis=1)))
        if np.linalg.norm(projections[index]-position)>.08:
            return False
        ahead = np.vstack([projections[index],points[index+1:]])
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ahead,axis=0),axis=1))]
        # Classify by the remaining geometry, not which stop happened first.
        # A visual-conflict latch with 9--30 cm left needs the same bounded
        # view change as path_end; it must not fall between two modes.
        short_exit = tracker.round_tracking and .015<=arc[-1]<.30
        if not tracker.conflict_latched and not short_exit:
            return False
        if short_exit:
            # A round exit can run out beneath the camera. Its last observed
            # tangent supplies only a direction to look, never a blind drive.
            tail = max(0,len(points)-6)
            tangent = points[-1]-points[tail]
            if (np.linalg.norm(points[-1]-position)>.35
                    or np.linalg.norm(tangent)<.025
                    or np.any(now>tracker.deadlines()[index:])):
                return False
        else:
            last = min(len(points),index+int(np.searchsorted(arc,.25))+1)
            if arc[-1]<.30 or np.any(now>tracker.deadlines()[index:last]):
                return False
            ends = np.column_stack([np.interp([.12,.24],arc,ahead[:,i]) for i in (0,1)])
            tangent = ends[1]-ends[0]
            if np.linalg.norm(tangent)<.08:
                return False
        error = wrap(math.atan2(tangent[1],tangent[0])-pose[2])
        minimum_error = 8 if short_exit else 12
        # A nearly exhausted round exit can point almost straight ahead of
        # the rover while the outgoing line is still outside the camera.
        # A confirmed conflict can use that small but directed angle to look.
        if short_exit and tracker.conflict_latched and arc[-1] <= .10:
            minimum_error = 4
        if not math.radians(minimum_error)<=abs(error)<=math.radians(80):
            return False
        turn = math.copysign(math.pi/3,error) if short_exit else float(np.clip(error,-math.pi/3,math.pi/3))
        self.heading = wrap(pose[2]+turn)
        self.started = now
        self.origin = position.copy()
        self.consumed = position.copy()
        self.consumed_arc = travelled
        self.previous_heading = pose[2]
        self.start_heading = pose[2]
        self.view_candidate = None
        self.view_frames = 0
        self.view_adjusted = False
        self.rotation = self.yaw = 0.
        self.support_frames = 0
        self.support_stamp = None
        self.state = 'conflict_recovery'
        self.kind = ('path_end' if end_request else 'short_exit') if short_exit else 'visual_conflict'
        self.hold_end = bool(short_exit)
        self.remaining_at_start = float(arc[-1])
        self.exit_reference = (points[-1].copy(), tangent/np.linalg.norm(tangent),
                               float(tracker.deadlines()[index:].min())) if short_exit else None
        self.reason = ''
        return True

    def current_exit_reference(self, now):
        if (self.state == 'conflict_recovery' and self.exit_reference is not None
                and self.started <= now < self.started+3.5 and now <= self.exit_reference[2]):
            return self.exit_reference[:2]
        return None

    def aim_at_visible_exit(self, tracker, mask, pose, now):
        """Adjust the stationary look once when a forward exit stays visible."""
        reference = self.current_exit_reference(now)
        if (reference is None or self.view_adjusted or tracker.progress.blocked
                or tracker.progress.phase != 'turning'):
            return
        candidate = None
        if single_near_line(mask):
            local = ground_centerline(mask)
            if len(local) >= 5 and local[0, 0] <= .40:
                c, s = math.cos(pose[2]), math.sin(pose[2])
                world = local @ np.array([[c, s], [-s, c]]) + pose[:2]
                nearby = world[np.linalg.norm(world-world[0], axis=1) <= .10]
                end, outgoing = reference
                gap = world[0]-end
                normal = np.array([-outgoing[1], outgoing[0]])
                if (len(nearby) >= 5 and .03 <= gap@outgoing <= .25
                        and abs(gap@normal) <= .10 and not tracker.progress.reverses(world)):
                    direction = nearby[-1]-nearby[0]
                    heading = math.atan2(direction[1], direction[0])
                    turn = abs(wrap(heading-pose[2]))
                    travelled = self.rotation + abs(wrap(pose[2]-self.previous_heading))
                    if (np.linalg.norm(direction) >= .07
                            and abs(wrap(heading-self.start_heading)) <= math.pi/3
                            and travelled+turn <= math.pi/3
                            and now-self.started+turn/.35 < 3.5
                            and turn+.10 < abs(wrap(self.heading-pose[2]))):
                        candidate = (now, world[0], heading)
        if candidate is None:
            self.view_candidate = None
            self.view_frames = 0
            return
        previous = self.view_candidate
        if previous is not None and now <= previous[0]:
            return
        if (previous is not None and now-previous[0] <= .2
                and np.linalg.norm(candidate[1]-previous[1]) <= .035
                and abs(wrap(candidate[2]-previous[2])) <= .15):
            self.view_frames += 1
        else:
            self.view_frames = 1
        self.view_candidate = candidate
        if self.view_frames >= 3:
            self.heading = candidate[2]
            self.view_adjusted = True

    def command(self, pose, now, dt, *, pause=False):
        if self.state!='conflict_recovery':
            return 0.,0.
        self.rotation += abs(wrap(pose[2]-self.previous_heading))
        self.previous_heading = pose[2]
        remaining = wrap(self.heading-pose[2])
        if (not 0.<=now-self.started<3.5 or self.rotation>=math.pi/3
                or np.linalg.norm(np.array(pose[:2])-self.origin)>.02):
            self.cancel('conflict_recovery_limit')
            return 0.,0.
        if abs(remaining)<.05:
            self.cancel('exit_heading_reached')
            return 0.,0.
        if pause:
            self.yaw = 0.
            return 0.,0.
        desired = math.copysign(min(.35,max(.1,.9*abs(remaining))),remaining)
        self.yaw += float(np.clip(desired-self.yaw,-3.*min(.1,max(.01,dt)),3.*min(.1,max(.01,dt))))
        return 0.,self.yaw

    def revalidate(self, tracker, mask, pose, now):
        # Looking at an old incoming arm is insufficient. The remembered,
        # non-reversing exit itself must now be visible and match the camera.
        if (self.hold_end or self.started is None or self.rotation<.1 or not 0.<=now-self.started<=4.
                or self.reason not in ('','exit_heading_reached')
                or not tracker.conflict_latched or tracker.progress.blocked
                or tracker.points is None or tracker.updated is None
                or now-tracker.updated>tracker.memory_seconds
                or np.any(now>tracker.deadlines())
                or np.linalg.norm(np.array(pose[:2])-self.origin)>.02
                or tracker.progress.reverses(tracker.points)):
            self.support_frames = 0
            return False
        support = projected_support(tracker.points,pose,mask,min_samples=12)
        if support is None or support<.8:
            self.support_frames = 0
            return False
        if self.support_stamp is not None and now<=self.support_stamp:
            return False
        self.support_frames = (self.support_frames+1 if self.support_stamp is not None
                               and now-self.support_stamp<=.2 else 1)
        self.support_stamp = now
        if self.support_frames<5:
            return False
        # Preserve all geometry, observation times and deadlines. Rotation
        # alone cannot authorize an expired path or renew an unseen segment.
        tracker.reset_conflict()
        tracker.update_reason = 'visible_exit_revalidated'
        self.cancel('visible_exit_revalidated')
        return True

    def cancel(self, reason):
        self.state = 'idle'
        self.exit_reference = None
        self.reason = reason
        self.yaw = 0.
        self.support_frames = 0
        self.support_stamp = None
        if reason not in ('conflict_recovery_limit','exit_heading_reached'):
            self.hold_end = False
