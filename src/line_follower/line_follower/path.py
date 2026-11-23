"""Short camera-derived ground path, compensated by measured ego motion.

Camera calibration is from the supplied Waffle Pi SDF: height 0.093 m
plus the 0.010 m resting model origin, forward offset 0.076 m. No map queries.
"""
import math
import numpy as np
from .progress import RouteProgress


class GroundPath:
    def __init__(self, max_speed=.18, straight_speed=None):
        self.max_speed = max_speed
        self.straight_speed = max_speed if straight_speed is None else max(max_speed,straight_speed)
        self.hill_until = -math.inf
        self.straight_speed_active = False
        self.progress = RouteProgress()
        self.points = None
        self.updated = None
        self.point_deadline = None
        self.point_seen = None
        self.seen_points = None
        self.memory_seconds = 3.
        self.update_reason = 'waiting_for_observation'
        self.observed_length = 0.
        self.path_confidence = 0.
        self.round_tracking = False
        self.round_exit_anchor = None
        self.alignment_offset = None
        self.alignment_frames = 0
        self.alignment_stamp = None
        self.unverified_since = None
        self.unverified_stamp = None
        self.unverified_frames = 0
        self.unverified_arc = None
        self.unverified_heading = None
        self.conflict_since = None
        self.conflict_stamp = None
        self.conflict_frames = 0
        self.conflict_latched = False
        self.visible_support = None
        self.splice_error = ''
        self.splice_offset = None
        self.splice_frames = 0
        self.splice_stamp = None
        self.speed = 0.
        self.yaw = 0.
        self.target_yaw = 0.
        self.forward_limit_reason = ''
        self.visible_straight_steering = False
        self.stop_reason = 'waiting_for_path'
        self.reacquire_frames = 0
        self.reacquire_anchor = None
        self.reacquire_stamp = None
        self.reacquire_direction = None
        self.lookahead = .09
        self.target_curvature = 0.
        self.straight_alignment_frames = 0
        self.straight_alignment_stamp = None
        self.straight_alignment_anchor = None
        self.straight_alignment_direction = None

    @staticmethod
    def project(points, shape):
        h, w = shape[:2]
        focal = w / (2. * math.tan(1.085595 / 2.))
        result = []
        for x, y in points:
            if y <= h / 2 + 5:
                continue
            depth = .103 * focal / (y - h / 2)
            forward = .076 + depth
            left = (w / 2 - x) * depth / focal
            if forward <= .85:
                result.append((forward, left))
        return np.asarray(result, dtype=float).reshape(-1, 2)

    def update(self, pixels, shape, pose, now, mask=None, smooth_start=False):
        prior = None
        if mask is not None and self.points is not None:
            c,s = math.cos(pose[2]),math.sin(pose[2])
            cached = (self.points-np.array(pose[:2])) @ np.array([[c,-s],[s,c]])
            visible = ((cached[:,0] >= .315) & (cached[:,0] <= .85)
                       & (np.abs(cached[:,1]) < (cached[:,0]-.076)*math.tan(1.085595/2)))
            indexes = np.flatnonzero(visible)
            if len(indexes):
                prior = cached[indexes[0]:]
        local = self.project(pixels, shape) if mask is None else ground_centerline(mask,prior)
        self.observed_length = (float(np.linalg.norm(np.diff(local,axis=0),axis=1).sum())
                                if len(local)>1 else 0.)
        self.path_confidence = min(1.,self.observed_length/.20)
        if len(local) < 2 or (mask is not None and self.observed_length < .09):
            self.reset_straight_alignment()
            self.update_reason = 'observation_too_short'
            self.path_confidence = 0.
            return
        x, y, angle = pose
        c, s = math.cos(angle), math.sin(angle)
        world = local @ np.array([[c, s], [-s, c]]) + (x, y)
        if self.progress.reverses(world):
            # Seeing a passed arm is allowed; replacing the active route with
            # that arm in reverse is not. Stop only if the commanded route does it.
            return self.reject('observation_returns_on_passed_path',world,pose,now,mask)
        old_seen = self.observed_at()
        old_deadlines = self.deadlines()
        # Keep the path hidden below the camera; replace only its visible tail.
        if self.points is None:
            self.reset_straight_alignment()
            arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(world,axis=0),axis=1))]
            tangent_index = min(len(world)-1,int(np.searchsorted(arc,.05)))
            tangent = world[tangent_index] - world[0]
            length = np.linalg.norm(tangent)
            if length < .001:
                return
            tangent /= length
            # At startup, extend the visible near tangent back to the axle.
            back = max(0., float(np.dot(world[0] - (x, y), tangent)))
            prefix = world[0] - tangent * (back + .1)
            initial = np.vstack([prefix,world])
            if smooth_start:
                # A stopped rover is not already on the extrapolated line.
                # Start at the axle with its current heading, joining the
                # visible tangent gradually instead of demanding a hard turn
                # toward a parallel line 5--10 cm away at a 9 cm lookahead.
                position = np.array(pose[:2])
                distance = float(np.linalg.norm(world[0]-position))
                u = np.linspace(0.,1.,max(8,int(distance/.01)))[:,None]
                incoming = distance*np.array([c,s])
                outgoing = distance*tangent
                connector = ((2*u**3-3*u**2+1)*position
                             +(u**3-2*u**2+u)*incoming
                             +(-2*u**3+3*u**2)*world[0]
                             +(u**3-u**2)*outgoing)
                initial = np.vstack([connector[:-1],world])
            if self.progress.reverses(initial):
                return self.reject('observation_returns_on_passed_path',world,pose,now,mask)
            self.points = initial
            retained_prefix = retained_tail = np.empty(0)
            prefix_deadlines = tail_deadlines = np.empty(0)
        else:
            distances = np.linalg.norm(self.points-world[0],axis=1)
            # Select the nearest compatible directed arm, rather than rejecting
            # the whole observation when the closest arm runs the other way.
            observed_direction = world[min(5,len(world)-1)]-world[0]
            following = np.minimum(np.arange(len(self.points))+5,len(self.points)-1)
            cached_directions = self.points[following]-self.points
            short = np.linalg.norm(cached_directions,axis=1) < 1e-6
            previous = np.maximum(np.arange(len(self.points))-5,0)
            cached_directions[short] = (self.points-self.points[previous])[short]
            norms = np.linalg.norm(cached_directions,axis=1)*np.linalg.norm(observed_direction)
            alignment = (cached_directions@observed_direction)/np.maximum(norms,1e-9)
            round_path = self.round_tracking
            compatible = ((norms < 1e-8) | (alignment >= (.7 if round_path else -.3)))
            index = int(np.argmin(np.where(compatible,distances,np.inf)))
            if round_path:
                distances_all = np.linalg.norm(world[:,None,:]-self.points[None,:,:],axis=2)
                matches = np.argmin(distances_all,axis=1)
                overlap = distances_all[np.arange(len(world)),matches] < .04
                cached_length = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
                progress = cached_length[matches[overlap]]
                if len(progress)>=5 and np.max(np.maximum.accumulate(progress)-progress) > .10:
                    return self.reject('observation_reverses_path_order',world,pose,now,mask)
            tolerance = .015 if round_path else .15
            if round_path and .015 < distances[index] <= .06 and single_near_line(mask):
                offset = world[0]-self.points[index]
                if (self.alignment_offset is not None and self.alignment_stamp is not None
                        and 0. <= now-self.alignment_stamp <= .2
                        and np.linalg.norm(offset-self.alignment_offset) <= .015):
                    if self.alignment_stamp is None or now>self.alignment_stamp:
                        self.alignment_frames += 1
                else:
                    self.alignment_frames = 1
                self.alignment_offset = offset
                self.alignment_stamp = now
                if self.alignment_frames >= 3:
                    tolerance = .06
            else:
                self.alignment_frames = 0
                self.alignment_offset = None
                self.alignment_stamp = None
            bridge = False
            if not compatible[index] or distances[index] > tolerance:
                # At a round-bend exit, the remembered arc may end underneath
                # the camera while the straight exit first appears 30 cm ahead.
                # Bridge only a collinear, forward continuation of that endpoint.
                tail_index = max(0,len(self.points)-6)
                tangent = self.points[-1]-self.points[tail_index]
                tangent /= max(np.linalg.norm(tangent),1e-9)
                normal = np.array([-tangent[1],tangent[0]])
                gap = world[0]-self.points[-1]
                near = world[np.linalg.norm(world-world[0],axis=1)<=.15]
                if len(near) >= 3:
                    observed = near[-1]-near[0]
                    length = np.linalg.norm(observed)
                    observed_normal = np.array([-observed[1],observed[0]])/max(length,1e-9)
                    straight = np.max(np.abs((near-near[0])@observed_normal)) <= .005
                    heading_match = observed@tangent/max(length,1e-9)
                    tail_distance = np.linalg.norm(self.points[-1]-np.array(pose[:2]))
                    bridge = (local[0,0] <= .4 and length >= .1 and straight
                              and tail_distance <= .15
                              and 0. <= gap@tangent <= .4 and abs(gap@normal) <= .025
                              and heading_match >= .94)
                    # The camera first sees the outgoing straight while the
                    # rover is still following the final arc underneath it.
                    # Allow a short directed join while that arc is valid.
                    if not bridge and (round_path and not self.progress.blocked
                            and not self.conflict_latched and now <= old_deadlines.min()
                            and .15 < tail_distance <= .25
                            and local[0,0] <= .36 and length >= .12 and straight
                            and .03 <= gap@tangent <= .15 and abs(gap@normal) <= .035
                            and heading_match >= .82 and tangent@[c,s] >= .94
                            and gap@observed >= .9*np.linalg.norm(gap)*length
                            and single_near_line(mask)
                            and not self.progress.reverses(np.vstack([self.points[-1:],world]))):
                        bridge = True
                if not bridge:
                    return self.reject('observation_disconnected_or_reversed',world,pose,now,mask)
                index = len(self.points)-1
            end_distances = np.linalg.norm(self.points-world[-1],axis=1)
            end_index = int(np.argmin(end_distances))
            tail = np.empty((0,2))
            if index <= end_index < len(self.points)-1 and end_distances[end_index] < .08:
                outgoing = world[-1]-world[max(0,len(world)-5)]
                cached = self.points[min(end_index+5,len(self.points)-1)]-self.points[end_index]
                norm = np.linalg.norm(outgoing)*np.linalg.norm(cached)
                # Do not append stale points that lead back along the new arm.
                if norm > 1e-8 and outgoing@cached/norm >= -.3:
                    tail = self.points[end_index+1:]
            prefix = self.points.copy() if bridge else self.points[:index].copy()
            prefix,straight_aligned = self.align_straight_prefix(prefix,world,pose,now,mask)
            near = world[np.linalg.norm(world-world[0],axis=1) <= .15]
            if not straight_aligned and len(near) >= 3 and len(prefix):
                tangent = near[-1]-near[0]
                length = np.linalg.norm(tangent)
                if length > .02:
                    normal = np.array([-tangent[1],tangent[0]])/length
                    residual = np.max(np.abs((near-near[0])@normal))
                    # Only a straight continuation may shift the blind path;
                    # fitting the tangent of an upcoming bend would turn early.
                    cached = np.vstack([prefix, self.points[index]])
                    cached_residual = np.max(np.abs((cached-cached[-1])@normal))
                    # A straight outgoing arm must not translate the hidden
                    # incoming arm and corner as if they were one straight line.
                    if length >= .1 and residual <= .004 and cached_residual <= .01:
                        correction = float(np.dot(world[0]-self.points[index],normal))
                        gain,limit = (.2,.005) if self.round_tracking else (.6,.02)
                        prefix += normal*float(np.clip(gain*correction,-limit,limit))
                        # Spread the remaining observed straight-line offset
                        # over the last 15 cm instead of leaving a lateral step.
                        if not smooth_join(prefix,world):
                            prefix = blend_lateral(prefix,world[0],normal,.15)
            self.splice_error = ''
            if len(prefix) and not smooth_join(prefix,world):
                incoming = prefix[-1]-prefix[max(0,len(prefix)-5)]
                outgoing = world[min(4,len(world)-1)]-world[0]
                norm = np.linalg.norm(incoming)*np.linalg.norm(outgoing)
                if norm>.0001 and incoming@outgoing/norm>=.8:
                    tangent = outgoing/np.linalg.norm(outgoing)
                    normal = np.array([-tangent[1],tangent[0]])
                    offset = float((world[0]-prefix[-1])@normal)
                    consistent = (self.splice_offset is not None and self.splice_stamp is not None
                                  and 0.<now-self.splice_stamp<=.2
                                  and abs(offset-self.splice_offset)<=.025)
                    if self.splice_stamp!=now:
                        self.splice_frames = self.splice_frames+1 if consistent else 1
                    self.splice_offset = offset
                    self.splice_stamp = now
                    # A stable single observed arm can correct a larger drift,
                    # but spread it over 20 cm instead of inserting a step.
                    limit = .08 if self.splice_frames>=3 and single_near_line(mask) else .04
                    if abs(offset)<=limit:
                        prefix = blend_lateral(prefix,world[0],normal,.20)
            if len(prefix) and not smooth_join(prefix,world):
                self.splice_error = 'prefix'
                return self.reject('observation_bad_splice',world,pose,now,mask)
            if len(tail) and not smooth_join(world,tail):
                outgoing = world[-1]-world[max(0,len(world)-5)]
                incoming = tail[min(4,len(tail)-1)]-tail[0]
                norm = np.linalg.norm(outgoing)*np.linalg.norm(incoming)
                offset = world[-1]-tail[0]
                if norm>.0001 and outgoing@incoming/norm>=.8 and np.linalg.norm(offset)<=.04:
                    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(tail,axis=0),axis=1))]
                    weights = np.clip(1.-arc/.15,0.,1.)
                    weights = weights*weights*(3.-2.*weights)
                    tail = tail+weights[:,None]*offset
                if not smooth_join(world,tail):
                    self.splice_error = 'tail_dropped'
                    tail = np.empty((0,2))
            prefix_deadlines = old_deadlines[:len(prefix)]
            tail_deadlines = old_deadlines[end_index+1:] if len(tail) else np.empty(0)
            retained_prefix = old_seen[:len(prefix)]
            retained_tail = old_seen[end_index+1:] if len(tail) else np.empty(0)
            self.points = np.vstack([prefix, world, tail])
        if not self.conflict_latched:
            self.reset_conflict()
        self.updated = now
        self.update_reason = 'accepted'
        # Tight round bends can remain entirely inside the camera blind area
        # for longer than a right-angle bend. Spend a bounded extra interval
        # only when the observed path actually contains that much turning.
        turns = path_turns(self.points)
        turning = float(np.sum(np.abs(turns)))
        local_turns = path_turns(local)
        positive = float(np.sum(local_turns[local_turns>0.]))
        negative = float(-np.sum(local_turns[local_turns<0.]))
        # A clipped line hooks along the image edge during thinning. Do not
        # classify that terminal hook as a second physical bend.
        focal = shape[1]/(2.*math.tan(1.085595/2))
        end_u = shape[1]/2-local[-1,1]*focal/max(local[-1,0]-.076,.01)
        compound = (self.observed_length>=.4 and .1*shape[1]<end_u<.9*shape[1]
                    and min(positive,negative)>math.radians(40)
                    and positive+negative>math.radians(120))
        if abs(float(np.sum(local_turns))) > math.radians(150) or compound:
            self.round_tracking = True
            self.round_exit_anchor = None
        elif self.round_tracking and turning < .4 and self.observed_length > .25:
            if self.round_exit_anchor is None:
                self.round_exit_anchor = np.array(pose[:2])
            elif (np.linalg.norm(np.array(pose[:2])-self.round_exit_anchor) > .8
                  and single_near_line(mask)):
                self.round_tracking = False
                self.round_exit_anchor = None
        # A compound bend can require more than eight seconds at the yaw and
        # forward-speed limits. Budget for both travel and turning, measured
        # only from the accepted geometry. Rejected/blank frames cannot renew it.
        distance = float(np.linalg.norm(np.diff(self.points,axis=0),axis=1).sum())
        travel_seconds = distance/max(self.max_speed,.01)
        self.memory_seconds = (min(12., max(3., travel_seconds+1., turning/.5+1.))
                               if turning > math.radians(60) else 3.)
        # Keep actual observation times separate from inferred validity.
        # Retained geometry was not seen again when a fragment was accepted.
        fresh_count = len(self.points)-len(retained_prefix)-len(retained_tail)
        self.point_seen = np.r_[retained_prefix,np.full(fresh_count,now),retained_tail]
        # A directed, connected observation of >=15 cm can anchor the blind
        # incoming prefix. A shorter fragment only verifies its own geometry.
        # Retained outgoing geometry may share that anchor only after the
        # directed overlap and splice checks above have accepted the join.
        if self.observed_length>=.15:
            prefix_deadlines = np.full(len(retained_prefix),now+self.memory_seconds)
            tail_deadlines = np.full(len(retained_tail),now+self.memory_seconds)
        self.point_deadline = np.r_[prefix_deadlines,
                                    np.full(fresh_count,now+self.memory_seconds),tail_deadlines]
        self.seen_points = self.points
        self.progress.observe(pose,self.round_tracking,
                              self.observed_length>.25 and abs(float(np.sum(local_turns)))<.4)
        return local

    def reset_straight_alignment(self):
        self.straight_alignment_frames = 0
        self.straight_alignment_stamp = None
        self.straight_alignment_anchor = None
        self.straight_alignment_direction = None

    def align_straight_prefix(self, prefix, world, pose, now, mask):
        """Gently align a small blind-path error to a persistent straight line."""
        near = world[np.linalg.norm(world-world[0],axis=1)<=.25]
        if (mask is None or self.round_tracking or len(prefix)<8
                or len(near)<10 or not single_near_line(mask)):
            self.reset_straight_alignment()
            return prefix,False
        center = near.mean(axis=0)
        _,_,axes = np.linalg.svd(near-center,full_matrices=False)
        tangent = axes[0]
        if tangent@(near[-1]-near[0])<0.:
            tangent = -tangent
        normal = np.array([-tangent[1],tangent[0]])
        correction = (center-prefix)@normal
        align_prefix = self.memory_seconds <= 3.
        if (np.linalg.norm(near[-1]-near[0])<.20
                or np.max(np.abs((near-center)@normal))>.003
                or abs((center-np.array(pose[:2]))@normal)>.025):
            self.reset_straight_alignment()
            return prefix,False
        if align_prefix:
            cached_center = prefix.mean(axis=0)
            _,_,cached_axes = np.linalg.svd(prefix-cached_center,full_matrices=False)
            cached_tangent = cached_axes[0]
            if cached_tangent@(prefix[-1]-prefix[0])<0.:
                cached_tangent = -cached_tangent
            cached_normal = np.array([-cached_tangent[1],cached_tangent[0]])
            if (np.linalg.norm(prefix[-1]-prefix[0])<.20
                    or np.max(np.abs((prefix-cached_center)@cached_normal))>.008
                    or cached_tangent@tangent<.995
                    or np.max(np.abs(correction))>.025):
                self.reset_straight_alignment()
                return prefix,False
        if self.straight_alignment_stamp is not None and now<=self.straight_alignment_stamp:
            return prefix,align_prefix
        stable = (self.straight_alignment_stamp is not None
                  and now-self.straight_alignment_stamp<=.2
                  and self.straight_alignment_direction@tangent>=.999
                  and abs((center-self.straight_alignment_anchor)@normal)<=.005)
        self.straight_alignment_frames = self.straight_alignment_frames+1 if stable else 1
        self.straight_alignment_stamp = now
        self.straight_alignment_anchor = center
        self.straight_alignment_direction = tangent
        if not align_prefix:
            return prefix,False
        if self.straight_alignment_frames<3:
            return prefix,True
        # Never translate a whole blind path by a 2 cm frame-to-frame step.
        # A least-squares tangent over 20+ cm also suppresses 5 mm grid steps.
        return prefix+np.clip(.25*correction,-.002,.002)[:,None]*normal,True

    def observed_at(self):
        """Per-point observation times; support restored/externally supplied legacy paths."""
        if self.points is None:
            return np.empty(0)
        if self.seen_points is self.points and self.point_seen is not None:
            return self.point_seen
        stamp = self.updated if self.updated is not None else -math.inf
        return np.full(len(self.points),stamp)

    def deadlines(self):
        if self.seen_points is self.points and self.point_deadline is not None:
            return self.point_deadline
        return self.observed_at()+self.memory_seconds

    def command(self, pose, now, dt, *, allow_straight_boost=True, odom_speed=None):
        self.progress.record(pose)
        if self.progress.blocked:
            return self.stop('route_reversal_stop')
        if self.conflict_latched:
            return self.stop('path_visual_conflict')
        if self.points is None or self.updated is None or now-self.updated>self.memory_seconds:
            return self.stop('path_missing_or_expired')
        position = np.array(pose[:2])
        starts, ends = self.points[:-1], self.points[1:]
        edges = ends - starts
        lengths2 = np.sum(edges * edges, axis=1)
        t = np.clip(np.sum((position-starts)*edges, axis=1) /
                    np.maximum(lengths2, 1e-10), 0., 1.)
        projections = starts + t[:, None] * edges
        index = int(np.argmin(np.linalg.norm(projections-position, axis=1)))
        nearest = projections[index]
        seen = self.observed_at()
        deadlines = self.deadlines()
        # Check the segment under the axle and the actual steering lookahead.
        # A newly observed distant arm must not authorize an expired bend here.
        upcoming = np.vstack([nearest,self.points[index+1:]])
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(upcoming,axis=0),axis=1))]
        self.lookahead = .09
        if (allow_straight_boost and self.round_tracking and self.progress.phase == 'exiting'
                and not self.conflict_frames and not self.reacquire_frames
                and self.update_reason == 'accepted' and self.path_confidence >= .75
                and 0. <= now-self.updated <= .12 and arc[-1] >= .14
                and np.linalg.norm(nearest-position) <= .025):
            self.lookahead = self.round_exit_lookahead(
                upcoming, arc, pose[2], now, deadlines[index:])
        last = min(len(self.points),index+int(np.searchsorted(arc,self.lookahead))+1)
        if np.any(now>deadlines[index:last]):
            return self.stop('path_missing_or_expired')
        horizon = min(len(upcoming),int(np.searchsorted(arc,.30))+1)
        if self.progress.reverses(upcoming[:horizon]):
            self.progress.blocked = True
            return self.stop('route_reversal_stop')
        if np.linalg.norm(nearest-position) > .16:
            return self.stop('path_too_far')
        # A short lookahead follows the bend at the axle instead of cutting
        # toward a point 30--45 cm ahead in the camera image.
        remaining = self.lookahead
        target = nearest
        for end in self.points[index+1:]:
            distance = np.linalg.norm(end-target)
            if distance >= remaining:
                target = target + (end-target) * remaining / distance
                remaining = 0.
                break
            remaining -= distance
            target = end
        if remaining > .04:
            return self.stop('path_end')
        delta = target-position
        c, s = math.cos(pose[2]), math.sin(pose[2])
        forward, left = c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]
        if forward < -.02:
            return self.stop('target_behind')
        curvature = 2.*left/max(float(delta@delta), .0025)
        self.visible_straight_steering = False
        if (self.straight_alignment_frames >= 3 and self.straight_alignment_stamp == now
                and self.updated == now and not self.round_tracking
                and not self.conflict_frames and not self.reacquire_frames
                and allow_straight_boost and arc[-1] >= .20):
            # The axle lookahead lies inside the camera blind area. Use the
            # stable visible line for steering only when the cached path
            # agrees all the way to the farther straight steering target.
            last_straight = int(np.searchsorted(arc, .20))+1
            tangent = self.straight_alignment_direction
            normal = np.array([-tangent[1], tangent[0]])
            cached = upcoming[:last_straight]
            if self.memory_seconds > 3.:
                # A previous bend can make the whole cache long-lived. Check
                # the next 20 cm separately, averaging over the grid steps.
                samples = np.column_stack([
                    np.interp(np.linspace(0., .20, 4), arc, axis) for axis in upcoming.T
                ])
                steps = np.diff(samples, axis=0)
                lateral_limit, alignment = .022, .94
            else:
                steps = np.diff(cached, axis=0)
                lateral_limit, alignment = .012, .90
            if (not np.any(now > deadlines[index:index+last_straight])
                    and np.max(np.abs((cached-self.straight_alignment_anchor)@normal)) <= lateral_limit
                    and np.all(steps@tangent >= alignment*np.linalg.norm(steps, axis=1))):
                offset = float((self.straight_alignment_anchor-position)@normal)
                delta = .20*tangent+offset*normal
                curvature = 2.*(c*delta[1]-s*delta[0])/float(delta@delta)
                self.visible_straight_steering = True
        self.target_curvature = curvature
        ceiling = self.max_speed
        if allow_straight_boost:
            ceiling = self.straight_speed_ceiling(
                upcoming, arc, pose, now, curvature, deadlines[index:], odom_speed)
        if now <= self.hill_until:
            ceiling = min(ceiling, .32)
        corner_speed = ceiling
        if (allow_straight_boost and not self.visible_straight_steering
                and (self.round_tracking or self.memory_seconds > 3.)
                and abs(curvature) <= .65 and abs(self.yaw) <= .20):
            measured_speed = max(self.speed, odom_speed if odom_speed is not None else 0.)
            corner_speed = self.preview_corner_speed(upcoming, arc, now, deadlines[index:],
                                                     measured_speed, ceiling)
        self.straight_speed_active = corner_speed > self.max_speed
        preview_braking = corner_speed < ceiling
        ceiling = corner_speed
        speed = min(ceiling, math.sqrt(.06/max(abs(curvature), .01)),
                    .85/max(abs(curvature), .01))
        if abs(math.atan2(left, forward)) > .7:
            speed = min(speed, .05)
        # A corner can look locally straight as the cached route runs out.
        # Keep the approach slow until the camera can extend the route again.
        short_cached_corner = (allow_straight_boost
                               and (self.round_tracking or self.memory_seconds > 3.)
                               and now-self.updated >= .35 and arc[-1] <= .20)
        if short_cached_corner:
            speed = min(speed, .10)
        if self.conflict_frames:
            speed = min(speed,.06)
        if (preview_braking or short_cached_corner or self.conflict_frames) and self.speed > speed:
            # Scale yaw with braking to preserve the prior turning radius.
            self.yaw *= speed/self.speed
        dt = float(np.clip(dt, .01, .1))
        self.speed = min(speed, self.speed+.25*dt)
        desired = float(np.clip(self.speed*curvature, -.85, .85))
        self.target_yaw = desired
        self.forward_limit_reason = 'short_cached_corner' if short_cached_corner else ''
        if preview_braking and not self.forward_limit_reason:
            self.forward_limit_reason = 'preview_corner'
        self.yaw += float(np.clip(desired-self.yaw, -3.*dt, 3.*dt))
        # On a nearly straight path, a small bounded yaw transient must not
        # turn every sign change into a full forward stop. With the existing
        # 3 rad/s² slew, this bounds excess heading to 0.005 rad (~0.3 degrees)
        # if the target stays fixed. Tight bends and conflict braking retain
        # the exact turning-radius guard.
        yaw_error = abs(self.yaw-desired)
        gentle_correction = (self.memory_seconds<=3.
                             and not self.round_tracking and not self.conflict_frames
                             and abs(curvature)<=1.
                             and max(abs(self.yaw),abs(desired))<=.2
                             and yaw_error*yaw_error/(2.*3.)<=.005)
        if abs(curvature) > .1 and not gentle_correction:
            available = max(0., self.yaw/curvature)
            if self.speed > available:
                self.forward_limit_reason = 'turn_radius'
                # Fresh tracking can carry a little motion through a small
                # steering sign change while leaving the yaw slew bound intact.
                if (available == 0. and allow_straight_boost
                        and self.updated == now and self.update_reason == 'accepted'
                        and self.path_confidence >= .75 and not self.round_tracking
                        and not self.conflict_frames and not self.reacquire_frames
                        and abs(curvature) <= .6 and abs(self.yaw) <= .06
                        and yaw_error*yaw_error/(2.*3.) <= .005):
                    available = min(.04, .0025/max(dt, yaw_error/3.))
                    self.forward_limit_reason = 'small_yaw_transition'
                self.speed = min(self.speed, available)
        # Prune passed points, retaining the segment beneath the axle.
        prune = max(0,index-1)
        self.points = self.points[prune:]
        self.point_seen = seen[prune:]
        self.point_deadline = deadlines[prune:]
        self.seen_points = self.points
        self.stop_reason = ''
        return self.speed, self.yaw

    @staticmethod
    def preview_corner_speed(upcoming, arc, now, deadlines, measured_speed, ceiling):
        """Propagate each visible turn's speed limit back to the axle."""
        valid = np.flatnonzero(now > deadlines)
        horizon = min(.50, arc[max(0, valid[0]-1)] if valid.size else arc[-1])
        samples = np.arange(0., horizon+1e-9, .02)
        if len(samples) < 7:
            return ceiling
        points = np.column_stack([np.interp(samples, arc, axis) for axis in upcoming.T])
        # Compare adjacent 6 cm chords to suppress pixel-grid heading jitter.
        chords = points[3:]-points[:-3]
        headings = np.arctan2(chords[:, 1], chords[:, 0])
        difference = headings[3:]-headings[:-3]
        turns = np.abs(np.arctan2(np.sin(difference), np.cos(difference)))
        bends = turns >= .25
        if not np.any(bends):
            return ceiling
        curvature = turns[bends]/.06
        corner_limits = np.minimum(np.sqrt(.06/curvature), .85/curvature)
        corner_limits = np.where(turns[bends] >= .75, np.minimum(corner_limits, .085),
                                 corner_limits)
        usable = np.maximum(0., samples[:-6][bends]-.05-.24*measured_speed)
        return min(ceiling, float(np.min(np.sqrt(corner_limits**2+.20*usable))))

    def straight_speed_ceiling(self, upcoming, arc, pose, now, curvature, deadlines, odom_speed):
        """Limit straight boost by the distance available before a bend."""
        if (self.straight_speed<=self.max_speed or self.round_tracking
                or self.conflict_frames or self.reacquire_frames
                or self.straight_alignment_frames<5
                or self.memory_seconds>3. or self.updated!=now
                or self.update_reason!='accepted' or self.path_confidence<.75
                or abs(curvature)>.5 or abs(self.yaw)>.15
                or np.linalg.norm(upcoming[0]-np.array(pose[:2]))>.02):
            return self.max_speed

        cap = min(self.straight_speed, max(self.max_speed, .20))
        # Higher speeds need a longer run of stable observations and enough
        # current camera coverage. Cached geometry alone may hide a bend.
        high_cap = min(self.straight_speed, max(self.max_speed, .23))
        high_preview = .40 + .75*(high_cap-self.max_speed)
        high_last = int(np.searchsorted(arc, high_preview))+1
        if (high_cap > cap and self.straight_alignment_frames >= 8
                and min(arc[-1], self.observed_length) >= high_preview+.05
                and np.all(now <= deadlines[:high_last])):
            cap = high_cap
        if (self.straight_speed > .25 and cap >= .23
                and self.straight_alignment_frames >= 12 and odom_speed is not None
                and 0. <= odom_speed <= self.straight_speed+.005
                and abs(curvature) <= .25 and abs(self.yaw) <= .08):
            # The reference profile budgets .4 m/s² for straight-line braking.
            # Limit boost to the speed that can brake within the verified view.
            horizon = min(arc[-1], self.observed_length)-.05
            expired = np.flatnonzero(now > deadlines)
            if expired.size:
                horizon = min(horizon, arc[max(0, expired[0]-1)])
            if horizon >= .40:
                bend = self.first_preview_bend(upcoming, arc, horizon)
                usable = (horizon if bend is None else bend)-self.lookahead-.07
                delay = .24
                brake = .4
                limit = math.sqrt(max(0., (brake*delay)**2+.085**2+2*brake*usable))-brake*delay
                measured_budget = .24*odom_speed+max(0., odom_speed**2-.085**2)/(2*brake)
                if limit > .25 and usable >= measured_budget:
                    return min(self.straight_speed, .45, limit)
        if (cap >= .23 and self.straight_speed > cap and cap < .25 and self.straight_alignment_frames >= 12
                and odom_speed is not None and 0. <= odom_speed <= .255
                and abs(curvature) <= .25 and abs(self.yaw) <= .08):
            trial_cap = min(self.straight_speed, .25)
            braking_speed = max(trial_cap, odom_speed)
            # Logged responses took up to 0.236 s. Budget that full interval
            # as delay, then use the existing provisional deceleration model.
            trial_preview = (self.lookahead + .05 + .02 + .24*braking_speed
                             + (braking_speed**2-.09**2)/(2*.10))
            if min(arc[-1], self.observed_length) >= trial_preview+.05:
                last = int(np.searchsorted(arc, trial_preview))+1
                if np.all(now <= deadlines[:last]) and self.first_preview_bend(
                        upcoming, arc, trial_preview) is None:
                    return trial_cap
        preview = .40 + .75*(cap-self.max_speed)
        if arc[-1] < preview+.05:
            return self.max_speed
        last = min(len(deadlines),int(np.searchsorted(arc,preview))+1)
        if np.any(now>deadlines[:last]):
            return self.max_speed
        bend_distance = self.first_preview_bend(upcoming, arc, preview)
        if bend_distance is None:
            return cap

        # Budget the steering lookahead, 5 cm of path uncertainty, one
        # sample interval, and command delay. Use provisional planning values
        # for deceleration and corner speed until odometry is measured.
        usable = max(0., bend_distance-self.lookahead-.05-.02-cap*.1)
        approach = math.sqrt(.09**2 + 2*.10*usable)
        return max(self.max_speed, min(cap, approach))

    @staticmethod
    def round_exit_lookahead(upcoming, arc, heading, now, deadlines):
        """Lengthen steering gently once the path ahead is a valid exit straight."""
        preview = .14
        last = int(np.searchsorted(arc, preview))+1
        if np.any(now > deadlines[:last]):
            return .09
        distances = np.linspace(0., preview, 8)
        points = np.column_stack([np.interp(distances, arc, axis) for axis in upcoming.T])
        displacement = points[-1]-points[0]
        tangent = displacement/max(np.linalg.norm(displacement), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        error = math.acos(float(np.clip(tangent@[math.cos(heading),math.sin(heading)], -1., 1.)))
        deviations = np.max(np.abs((points-points[0])@normal))
        # Limit the extension by the departure from a straight and by the
        # remaining heading correction. A bend keeps the normal 9 cm target.
        if np.any(np.diff(points, axis=0)@tangent < .018):
            return .09
        scale = max(0., min(1., 1.-deviations/.009, 1.-error/math.radians(18)))
        return .09 + .04*scale

    @staticmethod
    def first_preview_bend(upcoming, arc, preview):
        # Sample by arc length so uneven camera point spacing cannot shift
        # the bend test. Use the first 12 cm to establish the straight heading.
        distances = np.linspace(0., preview, math.ceil(preview/.02)+1)
        points = np.column_stack([np.interp(distances, arc, axis) for axis in upcoming.T])
        prefix = int(np.searchsorted(distances, .12))
        edges = np.diff(points, axis=0)
        headings = np.arctan2(edges[:, 1], edges[:, 0])
        tangent = points[prefix]-points[0]
        heading = math.atan2(tangent[1], tangent[0])
        heading_error = np.abs(np.arctan2(np.sin(headings-heading), np.cos(headings-heading)))
        lateral_error = np.abs((points-points[0]) @ np.array([-math.sin(heading), math.cos(heading)]))
        bends = np.flatnonzero((heading_error > .15) | (lateral_error[1:] > .008))
        return float(distances[bends[0]]) if bends.size else None

    def stop(self, reason):
        self.straight_speed_active = False
        self.target_curvature = 0.
        self.target_yaw = 0.
        self.forward_limit_reason = ''
        self.speed = self.yaw = 0.
        self.visible_straight_steering = False
        self.stop_reason = reason
        return 0., 0.

    def reacquire(self, mask, pose, now, near_supported, pending=False, *, exit_reference=None):
        """Rebuild a stopped path only from five stable near-line observations."""
        multiple = (self.conflict_latched or self.progress.blocked) and not single_near_line(mask)
        if (not near_supported or (not self.stop_reason and not pending)
                or (multiple and (exit_reference is None or self.progress.blocked))):
            self.reacquire_frames = 0
            self.reacquire_anchor = None
            return False
        local = ground_centerline(mask)
        if len(local) < 5 or local[0,0] > .4:
            self.reacquire_frames = 0
            self.reacquire_anchor = None
            return False
        # Fit the first 5 cm, excluding distant turns and crossing branches.
        tangent_points = local[np.linalg.norm(local-local[0],axis=1) <= .05]
        if len(tangent_points) < 3:
            self.reacquire_frames = 0
            return False
        tangent = tangent_points[-1]-tangent_points[0]
        length = np.linalg.norm(tangent)
        if length < .015 or tangent[0]/length < .7:
            self.reacquire_frames = 0
            return False
        tangent /= length
        lateral = local[0,1]-local[0,0]*tangent[1]/tangent[0]
        if abs(lateral) > .12:
            self.reacquire_frames = 0
            return False
        c,s = math.cos(pose[2]),math.sin(pose[2])
        anchor = np.array(pose[:2])+np.array([-s*lateral,c*lateral])
        direction = np.array([c*tangent[0]-s*tangent[1],s*tangent[0]+c*tangent[1]])
        if multiple:
            end, outgoing = exit_reference
            world = local @ np.array([[c,s],[-s,c]])+pose[:2]
            gap = world[0]-end
            normal = np.array([-outgoing[1],outgoing[0]])
            if (not self.round_tracking or direction@outgoing < .94
                    or not 0. <= gap@outgoing <= .32 or abs(gap@normal) > .08
                    or self.progress.reverses(world)):
                self.reacquire_frames = 0
                self.reacquire_anchor = None
                return False
        if pending and not self.pending_reacquire_supported(local,pose,anchor,direction,mask):
            self.reacquire_frames = 0
            self.reacquire_anchor = None
            return False
        if self.reacquire_stamp is not None and now<=self.reacquire_stamp:
            return False
        if (self.reacquire_anchor is not None and self.reacquire_stamp is not None
                and now-self.reacquire_stamp<=.2
                and np.linalg.norm(anchor-self.reacquire_anchor) <= .025
                and self.reacquire_direction is not None
                and direction@self.reacquire_direction>=.98):
            self.reacquire_frames += 1
        else:
            self.reacquire_frames = 1
        self.reacquire_anchor = anchor
        self.reacquire_stamp = now
        self.reacquire_direction = direction
        if self.reacquire_frames < 5:
            return False
        previous = self.points
        self.points = None
        accepted = self.update([],mask.shape,pose,now,mask,smooth_start=True)
        if accepted is None:
            self.points = previous
        self.reacquire_frames = 0
        self.reacquire_anchor = None
        if accepted is not None:
            self.reset_conflict()
            self.progress.blocked = False
        return accepted is not None

    def pending_reacquire_supported(self, local, pose, anchor, direction, mask):
        """Only align a pending conflict to the same nearby straight arm."""
        if (not self.conflict_frames or self.progress.blocked or self.points is None
                or len(self.points)<2
                or not single_near_line(mask)
                or np.linalg.norm(anchor-np.array(pose[:2]))>.04):
            return False
        near = local[np.linalg.norm(local-local[0],axis=1)<=.15]
        if len(near)<5 or np.linalg.norm(near[-1]-near[0])<.1:
            return False
        tangent = near[-1]-near[0]
        normal = np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
        if np.max(np.abs((near-near[0])@normal))>.005:
            return False
        position = np.array(pose[:2])
        edges = np.diff(self.points,axis=0)
        fraction = np.clip(np.sum((position-self.points[:-1])*edges,axis=1)
                           /np.maximum(np.sum(edges*edges,axis=1),1e-10),0.,1.)
        projections = self.points[:-1]+fraction[:,None]*edges
        index = int(np.argmin(np.linalg.norm(projections-position,axis=1)))
        if np.linalg.norm(projections[index]-position)>.04:
            return False
        if np.linalg.norm(anchor-projections[index])>.04:
            return False
        ahead = np.vstack([projections[index],self.points[index+1:]])
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ahead,axis=0),axis=1))]
        if arc[-1]<.08:
            return False
        samples = np.column_stack([np.interp(np.linspace(0.,.08,5),arc,ahead[:,i]) for i in (0,1)])
        if np.max(np.abs((samples-samples[0])@np.array([-direction[1],direction[0]])))>.012:
            return False
        tangent = samples[-1]-samples[0]
        return float(tangent@direction/max(np.linalg.norm(tangent),1e-9))>=.98


    def reset_conflict(self):
        self.unverified_since = self.unverified_stamp = None
        self.unverified_frames = 0
        self.unverified_arc = self.unverified_heading = None
        self.conflict_since = self.conflict_stamp = None
        self.conflict_frames = 0
        self.conflict_latched = False
        self.visible_support = None

    def reject(self, reason, world, pose, now, mask):
        self.reset_straight_alignment()
        self.update_reason = reason
        self.path_confidence = 0.
        if mask is None or self.points is None or self.conflict_latched:
            return
        relative = world[0]-np.array(pose[:2])
        forward = math.cos(pose[2])*relative[0]+math.sin(pose[2])*relative[1]
        if self.observed_length < .15:
            return
        self.visible_support = projected_support(self.points,pose,mask)
        if reason == 'observation_returns_on_passed_path' and self.visible_support is None:
            # An already passed arm is not evidence against an invisible exit.
            # Reject its adoption, but do not turn missing visibility into zero
            # support via the near-line fallback below. Visible contradictions
            # and commanded route reversals still use their existing guards.
            self.conflict_since = self.conflict_stamp = None
            self.conflict_frames = 0
            return
        nearest = float(np.min(np.linalg.norm(self.points-world[0],axis=1)))
        if self.visible_support is None and nearest>.12:
            if self.unverified_stamp is None or now>self.unverified_stamp:
                if self.unverified_since is None:
                    self.unverified_since = now
                    self.unverified_arc = self.progress.arc[-1] if self.progress.arc else 0.
                    self.unverified_heading = pose[2]
                self.unverified_stamp = now
                self.unverified_frames += 1
            # A distant line cannot contradict geometry outside the camera.
            # Keep the count for diagnostics, but wait for visible evidence;
            # distance travelled or heading change alone is not that evidence.
            # The original cache timestamp/expiry and route-direction checks
            # still bound motion while the expected bend is under the camera.
            if forward>.4:
                self.conflict_since = self.conflict_stamp = None
                self.conflict_frames = 0
                return
        if forward>.6:
            return
        if self.visible_support is None and forward<=.4 and nearest>.20:
            self.visible_support = 0.
        if self.visible_support is None:
            return
        if self.visible_support >= .25:
            if not self.conflict_latched:
                self.reset_conflict()
            return
        if self.conflict_stamp is not None and now<=self.conflict_stamp:
            return
        if self.conflict_stamp is not None and now-self.conflict_stamp>.25:
            self.conflict_since = None
            self.conflict_frames = 0
        if self.conflict_since is None:
            self.conflict_since = now
        self.conflict_stamp = now
        self.conflict_frames += 1
        if self.conflict_frames>=3 and now-self.conflict_since>=.15:
            self.conflict_latched = True


def ground_centerline(mask, prior=None):
    """Trace a thin line in ground coordinates, including transverse bends."""
    import cv2
    h, w = mask.shape
    focal = w/(2.*math.tan(1.085595/2.))
    # Five millimetre ground cells; the camera's blind area stays unobserved.
    forward = .18 + np.arange(145)*.005
    left = -.6 + np.arange(241)*.005
    xx, yy = np.meshgrid(forward, left, indexing='ij')
    u = (w/2 - yy*focal/(xx-.076)).astype(np.float32)
    v = (h/2 + .103*focal/(xx-.076)).astype(np.float32)
    binary = cv2.remap(mask, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    # Zhang--Suen thinning, using core OpenCV/numpy only.
    thin = np.pad(binary > 0, 1).astype(np.uint8)
    for _ in range(40):
        changed = False
        for phase in (0, 1):
            p = [thin[:-2,1:-1], thin[:-2,2:], thin[1:-1,2:], thin[2:,2:],
                 thin[2:,1:-1], thin[2:,:-2], thin[1:-1,:-2], thin[:-2,:-2]]
            neighbors = sum(p)
            transitions = sum((p[i] == 0) & (p[(i+1)%8] == 1) for i in range(8))
            if phase == 0:
                preserve = (p[0]*p[2]*p[4] == 0) & (p[2]*p[4]*p[6] == 0)
            else:
                preserve = (p[0]*p[2]*p[6] == 0) & (p[0]*p[4]*p[6] == 0)
            remove = ((thin[1:-1,1:-1] == 1) & (neighbors >= 2) &
                      (neighbors <= 6) & (transitions == 1) & preserve)
            if np.any(remove):
                thin[1:-1,1:-1][remove] = 0
                changed = True
        if not changed:
            break
    skeleton = thin[1:-1,1:-1].copy()
    from .tracing import join_fragments
    skeleton = join_fragments(skeleton)
    # Ground remapping can leave a tiny detached fragment at the image edge.
    # It must not win the nearest-seed search over a long visible return arm.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton,8)
    sizes = stats[:,cv2.CC_STAT_AREA]
    largest = int(np.max(sizes[1:])) if count > 1 else 0
    keep = sizes >= max(12,min(30,largest*.25))
    keep[0] = False
    skeleton = np.uint8(keep[labels])
    rows, cols = np.nonzero(skeleton)
    if len(rows) < 3:
        return np.empty((0,2))
    distance = forward[rows]**2 + left[cols]**2
    start_index = int(np.argmin(distance))
    seed_direction = None
    if prior is not None and len(prior) >= 2:
        candidates = np.column_stack([forward[rows],left[cols]])
        prior_distance = np.linalg.norm(candidates-prior[0],axis=1)
        match = int(np.argmin(prior_distance))
        if prior_distance[match] <= .08:
            start_index = match
            seed_direction = prior[min(5,len(prior)-1)]-prior[0]
    start = (int(rows[start_index]), int(cols[start_index]))
    if abs(left[start[1]]) > .22:
        return np.empty((0,2))
    from .tracing import trace
    prior_cells = None if prior is None else (prior-np.array([.18,-.6]))/.005
    cells = trace(skeleton,start,seed_direction,prior_cells)
    route = [(forward[row],left[col]) for row,col in cells]
    # Downsample while retaining the curve geometry.
    return np.asarray(route[::2], dtype=float).reshape(-1,2)


def path_turns(points):
    """Heading changes over 6 cm chords, suppressing skeleton stair steps."""
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0),axis=1))]
    if lengths[-1] < .12:
        return np.empty(0)
    samples = np.arange(0., lengths[-1]+1e-9, .06)
    sampled = np.column_stack([np.interp(samples,lengths,points[:,i]) for i in (0,1)])
    edges = np.diff(sampled,axis=0)
    headings = np.unwrap(np.arctan2(edges[:,1],edges[:,0]))
    turns = np.diff(headings)
    return turns[np.abs(turns) > .08]


def single_near_line(mask):
    """Only release return-arm protection when nearby parallel lines are gone."""
    if mask is None:
        return True
    h,w = mask.shape
    supported = 0
    for row in np.linspace(.8*h,.95*h,5).astype(int):
        active = np.flatnonzero(mask[row])
        groups = np.split(active,np.where(np.diff(active)>1)[0]+1)
        count = sum(3 <= len(group) <= .3*w for group in groups)
        if count>1:
            return False
        supported += count==1
    return supported>=3


def smooth_join(before, after):
    """Long joins must follow both neighboring tangents; short corners remain."""
    gap = after[0]-before[-1]
    length = np.linalg.norm(gap)
    if length <= .025:
        return True
    tangents = (before[-1]-before[max(0,len(before)-5)],
                after[min(4,len(after)-1)]-after[0])
    for tangent in tangents:
        norm = np.linalg.norm(tangent)
        if norm>=.025 and gap@tangent/(length*norm) < math.cos(math.radians(50)):
            return False
    return True


def projected_support(points, pose, mask, min_samples=6):
    """Fraction of camera-visible cached path supported by current black pixels."""
    import cv2
    h,w = mask.shape
    c,s = math.cos(pose[2]),math.sin(pose[2])
    local = (points-np.array(pose[:2])) @ np.array([[c,-s],[s,c]])
    length = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(local,axis=0),axis=1))]
    if length[-1] < .08:
        return None
    arc = np.arange(0.,length[-1],.015)
    samples = np.column_stack([np.interp(arc,length,local[:,i]) for i in (0,1)])
    focal = w/(2.*math.tan(1.085595/2))
    depth = samples[:,0]-.076
    good = depth>.05
    samples,depth = samples[good],depth[good]
    u = w/2-samples[:,1]*focal/depth
    v = h/2+.103*focal/depth
    valid = (u>=12)&(u<w-12)&(v>=h*.55)&(v<h-6)&(samples[:,0]<=.85)
    if np.count_nonzero(valid)<min_samples:
        return None
    expanded = cv2.dilate(mask,np.ones((25,25),np.uint8))
    return float(np.mean(expanded[np.rint(v[valid]).astype(int),np.rint(u[valid]).astype(int)]>0))


def blend_lateral(prefix, target, normal, distance):
    """Smooth a supported lateral correction without moving a hidden corner."""
    result = prefix.copy()
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(result,axis=0),axis=1))]
    weights = np.clip(1.-(arc[-1]-arc)/distance,0.,1.)
    weights = weights*weights*(3.-2.*weights)
    correction = float((target-result[-1])@normal)
    result += weights[:,None]*normal*correction
    return result

