"""Camera-derived ground path with odometry and sensor watchdogs."""
import time
import math
import json
import hashlib
from pathlib import Path
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from .vision import observe
from .path import GroundPath, single_near_line
from .timing import PoseHistory
from .recovery import CornerRecovery, ConflictRecovery
from .finish import TerminalApproach


class Follower(Node):
    def __init__(self):
        super().__init__('line_follower')
        for key, value in {'max_speed': .18, 'straight_speed': .45, 'threshold': 65,
                           'capture_dir': '', 'finish_distance': 12.0}.items():
            self.declare_parameter(key, value)
        self.ground_path = GroundPath(self.get_parameter('max_speed').value,
                                      self.get_parameter('straight_speed').value)
        self.corner_recovery = CornerRecovery()
        self.conflict_recovery = ConflictRecovery()
        self.pose_history = PoseHistory()
        self.pending_image = None
        self.last_control_stamp = None
        self.last_control_wall = None
        self.pose = None
        self.last_odom = None
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug = self.create_publisher(Image, '/line_follower/debug', qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.image, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self.imu, qos_profile_sensor_data)
        self.last_image = None
        self.tracking_state = None
        self.capture_session = str(time.time_ns())
        source = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
        self.controller_revision = hashlib.sha256(source).hexdigest()[:12]
        self.distance = 0.
        self.position = None
        self.done = False
        self.terminal_frames = 0
        self.terminal_confirmed_frames = 0
        self.finish = TerminalApproach(self.ground_path.max_speed)
        self.frames = 0
        self.create_timer(.05, self.watchdog)
        self.get_logger().info(f'Controller active; waiting for camera. revision={self.controller_revision}')

    def send(self, speed=0., yaw=0.):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(speed), float(yaw)
        self.pub.publish(msg)

    def imu(self, msg):
        if msg.orientation_covariance[0] < 0.:
            return
        q = msg.orientation
        if abs(2.*(q.w*q.y-q.z*q.x)) > math.sin(.03):
            self.hill_seen = time.monotonic()

    def odom(self, msg):
        p = msg.pose.pose.position
        twist = getattr(msg, 'twist', None)
        self.odom_speed = None if twist is None else twist.twist.linear.x
        if self.position is not None:
            self.distance += ((p.x-self.position[0])**2 + (p.y-self.position[1])**2)**.5
        self.position = (p.x, p.y)
        q = msg.pose.pose.orientation
        angle = math.atan2(2.*(q.w*q.z+q.x*q.y), 1.-2.*(q.y*q.y+q.z*q.z))
        self.pose = (p.x, p.y, angle)
        self.last_odom = time.monotonic()
        stamp = getattr(msg, 'header', None)
        if stamp is not None:
            stamp = stamp.stamp.sec + stamp.stamp.nanosec * 1e-9
            self.pose_history.add(stamp, self.pose)
            pending = self.pending_image
            if pending is not None:
                exposure = pending.header.stamp.sec + pending.header.stamp.nanosec * 1e-9
                if self.pose_history.at(exposure) is not None:
                    self.pending_image = None
                    self.image(pending)

    def watchdog(self):
        now = time.monotonic()
        if (self.last_image is None or now-self.last_image > .5 or self.done
                or self.last_odom is None or now-self.last_odom > .5
                or (getattr(self, 'last_control_stamp', None) is not None
                    and (self.last_control_wall is None or now-self.last_control_wall > .5))):
            self.corner_recovery.cancel('sensor_timeout')
            if hasattr(self,'conflict_recovery'):
                self.conflict_recovery.cancel('sensor_timeout')
            self.ground_path.stop('sensor_timeout')
            self.ground_path.reacquire_frames = 0
            self.ground_path.reacquire_anchor = None
            self.send()

    def image(self, msg):
        now = time.monotonic()
        dt = .033 if self.last_image is None else now-self.last_image
        self.last_image = now
        if self.done:
            self.send()
            return
        stamp = msg.header.stamp
        exposure = stamp.sec + stamp.nanosec * 1e-9
        observation_pose = self.pose
        if exposure > 0.:
            previous = getattr(self, 'last_control_stamp', None)
            if previous is not None and exposure <= previous:
                return
            observation_pose = self.pose_history.at(exposure)
            if observation_pose is None:
                self.pending_image = msg
                return
            dt = .033 if previous is None else exposure-previous
            self.last_control_stamp = exposure
            self.last_control_wall = now
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            obs = observe(frame, self.get_parameter('threshold').value)
        except (ValueError, cv2.error, RuntimeError) as exc:
            self.get_logger().error(str(exc))
            self.corner_recovery.cancel('image_error')
            if hasattr(self,'conflict_recovery'):
                self.conflict_recovery.cancel('image_error')
            self.ground_path.stop('image_error')
            self.ground_path.reacquire_frames = 0
            self.ground_path.reacquire_anchor = None
            self.send()
            return
        eligible = self.distance > self.get_parameter('finish_distance').value
        self.terminal_frames = self.terminal_frames + 1 if eligible and obs.terminal else 0
        stamp = msg.header.stamp
        # Cache age follows Gazebo time even when software rendering is slow.
        path_now = stamp.sec + stamp.nanosec * 1e-9
        if path_now <= 0.:
            path_now = now
        hill_age = now-getattr(self, 'hill_seen', -math.inf)
        self.ground_path.hill_until = path_now+.4-hill_age if 0. <= hill_age <= .4 else -math.inf
        ready = self.pose is not None and now-self.last_odom <= .5
        recovery = self.corner_recovery
        if not hasattr(self,'conflict_recovery'):
            self.conflict_recovery = ConflictRecovery()
        conflict_recovery = self.conflict_recovery
        if not hasattr(self,'finish'):
            self.finish = TerminalApproach(self.ground_path.max_speed)
        finish = self.finish
        if (ready and self.terminal_frames>=5 and finish.state=='idle'
                and not self.ground_path.conflict_latched and not conflict_recovery.hold_end):
            if finish.begin(obs.mask,observation_pose,path_now,self.ground_path.progress):
                self.terminal_confirmed_frames = self.terminal_frames
                recovery.cancel('terminal_approach')
                self.get_logger().info(f'Terminal T confirmed; approaching {finish.remaining:.3f} m to bar.')
        if ready:
            exit_reference = conflict_recovery.current_exit_reference(path_now)
            if self.ground_path.conflict_latched or self.ground_path.progress.blocked:
                reason = 'route_reversal_stop' if self.ground_path.progress.blocked else 'path_visual_conflict'
                speed,yaw = self.ground_path.stop(reason)
                if conflict_recovery.revalidate(self.ground_path,obs.mask,observation_pose,path_now):
                    recovery.cancel('visible_exit_revalidated')
                    self.get_logger().info('Remembered exit confirmed in camera after stationary recovery.')
                    speed,yaw = self.ground_path.command(
                        observation_pose, path_now, dt,
                        allow_straight_boost=self.terminal_frames == 0,
                        odom_speed=getattr(self, 'odom_speed', None))
                elif self.ground_path.reacquire(
                        obs.mask, observation_pose, path_now, True, exit_reference=exit_reference):
                    recovery.cancel('near_line_reacquired')
                    conflict_recovery.cancel('near_line_reacquired')
                    self.get_logger().info('Stable near line reacquired after visual conflict.')
                    speed,yaw = self.ground_path.command(
                        observation_pose, path_now, dt,
                        allow_straight_boost=self.terminal_frames == 0,
                        odom_speed=getattr(self, 'odom_speed', None))
                elif self.ground_path.progress.blocked:
                    recovery.cancel(reason)
                    conflict_recovery.cancel(reason)
                elif conflict_recovery.begin(self.ground_path,observation_pose,path_now):
                    recovery.cancel(reason)
                    # Keep a plausible near line in view while reacquisition
                    # gathers its five frames, including on ordinary corners.
                    confirming_line = (self.ground_path.reacquire_frames > 0
                                       and (exit_reference is not None
                                            or not self.ground_path.round_tracking))
                    if not confirming_line:
                        conflict_recovery.aim_at_visible_exit(
                            self.ground_path, obs.mask, observation_pose, path_now)
                    speed,yaw = conflict_recovery.command(
                        observation_pose, path_now, dt, pause=confirming_line)
                elif not self.ground_path.round_tracking and (
                        recovery.state == 'corner_recovery'
                        or recovery.begin(observation_pose,path_now,reason)):
                    speed,yaw = recovery.command(observation_pose,path_now,dt)
            elif conflict_recovery.hold_end:
                # Do not feed the exhausted cache back into normal driving.
                # Even after the one-shot turn ends, require five new near-line
                # observations and the ordinary direction/history guards.
                self.ground_path.stop('path_end')
                conflict_recovery.aim_at_visible_exit(self.ground_path,obs.mask,observation_pose,path_now)
                speed,yaw = conflict_recovery.command(observation_pose,path_now,dt)
                if self.ground_path.reacquire(obs.mask,observation_pose,path_now,
                                              single_near_line(obs.mask)):
                    conflict_recovery.cancel('near_line_reacquired')
                    recovery.cancel('near_line_reacquired')
                    self.get_logger().info('Stable exit reacquired after path end.')
                    speed,yaw = self.ground_path.command(
                        observation_pose, path_now, dt,
                        allow_straight_boost=self.terminal_frames == 0,
                        odom_speed=getattr(self, 'odom_speed', None))
            elif finish.state in ('approaching','failed'):
                speed,yaw = finish.command(observation_pose,path_now,self.ground_path.progress)
                if finish.state == 'complete':
                    self.done = True
                    self.ground_path.stop('terminal_t')
                    self.send()
                    Follower.record_finish(self,frame,observation_pose,path_now)
                    return
                if finish.state == 'failed':
                    self.ground_path.stop(finish.reason)
            elif recovery.state == 'corner_recovery':
                # Keep the failed cache out of the command loop until five
                # stable near-line frames justify rebuilding it.
                self.ground_path.stop('corner_recovery')
                speed, yaw = recovery.command(observation_pose, path_now, dt)
                if self.ground_path.reacquire(obs.mask, observation_pose, path_now,
                        obs.confidence >= .45 or self.ground_path.observed_length >= .15):
                    recovery.cancel('near_line_reacquired')
                    self.get_logger().info('Stable near line reacquired; rebuilding local path.')
                    speed, yaw = self.ground_path.command(
                        observation_pose, path_now, dt,
                        allow_straight_boost=self.terminal_frames == 0,
                        odom_speed=getattr(self, 'odom_speed', None))
            else:
                local = None
                if obs.confidence >= .23 or self.ground_path.points is not None:
                    local = self.ground_path.update(obs.points, frame.shape, observation_pose, path_now, obs.mask)
                if self.ground_path.conflict_frames and not self.ground_path.stop_reason:
                    if self.ground_path.reacquire(obs.mask,observation_pose,path_now,
                            obs.confidence>=.45 or self.ground_path.observed_length>=.15,
                            pending=True):
                        recovery.cancel('pending_conflict_realigned')
                        self.get_logger().info('Stable same-direction near line realigned before conflict stop.')
                recovery.observe(local, observation_pose, path_now)
                speed, yaw = self.ground_path.command(
                    observation_pose, path_now, dt,
                    allow_straight_boost=self.terminal_frames == 0,
                    odom_speed=getattr(self, 'odom_speed', None))
                if self.terminal_frames == 0 and recovery.begin_before_path_end(
                        self.ground_path, observation_pose, path_now):
                    self.ground_path.stop('corner_recovery')
                    speed, yaw = recovery.command(observation_pose, path_now, dt)
                elif not speed and not yaw:
                    if self.ground_path.reacquire(obs.mask, observation_pose, path_now,
                        obs.confidence >= .45 or self.ground_path.observed_length >= .15):
                        recovery.cancel('near_line_reacquired')
                        self.get_logger().info('Stable near line reacquired; rebuilding local path.')
                        speed, yaw = self.ground_path.command(
                            observation_pose, path_now, dt,
                            allow_straight_boost=self.terminal_frames == 0,
                            odom_speed=getattr(self, 'odom_speed', None))
                    elif recovery.begin(observation_pose, path_now, self.ground_path.stop_reason):
                        speed, yaw = recovery.command(observation_pose, path_now, dt)
                    elif conflict_recovery.begin(self.ground_path,observation_pose,path_now):
                        recovery.cancel('stationary_exit_observation')
                        speed,yaw = conflict_recovery.command(observation_pose,path_now,dt)
                elif not self.ground_path.conflict_frames:
                    self.ground_path.reacquire_frames = 0
                    self.ground_path.reacquire_anchor = None
        else:
            recovery.cancel('odometry_unavailable')
            conflict_recovery.cancel('odometry_unavailable')
            self.ground_path.stop('odometry_unavailable')
            speed, yaw = 0., 0.
        state = ('route_reversal_stop' if self.ground_path.progress.blocked else
                 'corner_recovery' if recovery.state == 'corner_recovery' else
                 ('short_exit_recovery' if conflict_recovery.hold_end else
                  ('conflict_recovery' if conflict_recovery.state=='conflict_recovery'
                   else 'visual_conflict_stop')) if self.ground_path.conflict_latched else
                 'path_end_recovery' if conflict_recovery.hold_end else
                 'terminal_approach_stop' if finish.state == 'failed' else
                 'terminal_approach' if finish.state == 'approaching' else
                 (('tracking' if self.ground_path.updated == path_now else 'cached_path')
                  if speed or yaw else 'path_stop'))
        if state != getattr(self, 'tracking_state', None):
            reason = self.ground_path.stop_reason if ready else 'odometry_unavailable'
            self.get_logger().info(f'State={state} confidence={obs.confidence:.2f} distance={self.distance:.2f} reason={reason} recovery={recovery.reason} path_confidence={self.ground_path.path_confidence:.2f} update={self.ground_path.update_reason}')
            self.tracking_state = state
        self.send(speed, yaw)
        control_processing_seconds = time.monotonic()-now
        self.frames += 1
        folder = self.get_parameter('capture_dir').value
        if folder and not getattr(self, 'control_trace_failed', False):
            session = Path(folder) / getattr(self, 'capture_session', 'session')
            trace = {
                'frame': self.frames, 'time': path_now, 'dt': dt,
                'distance': self.distance, 'speed': speed, 'yaw': yaw,
                'odom_speed': getattr(self, 'odom_speed', None),
                'target_yaw': self.ground_path.target_yaw,
                'curvature': self.ground_path.target_curvature,
                'visible_straight_steering': self.ground_path.visible_straight_steering
                and state == 'tracking',
                'forward_limit': self.ground_path.forward_limit_reason if state in ('tracking', 'cached_path') else '',
                'state': state, 'path_updated': self.ground_path.updated,
                'path_update_reason': self.ground_path.update_reason,
                'stop_reason': self.ground_path.stop_reason,
            }
            try:
                session.mkdir(parents=True, exist_ok=True)
                with (session/'control.jsonl').open('a') as output:
                    output.write(json.dumps(trace)+'\n')
            except OSError as exc:
                self.control_trace_failed = True
                self.get_logger().error(f'Could not write control trace: {exc}')
        if self.frames % 10 == 0:
            annotated = frame.copy()
            for x, y in obs.points:
                cv2.circle(annotated, (int(x), y), 4, (0, 0, 255), -1)
            cv2.putText(annotated, f'v={speed:.2f} w={yaw:.2f} c={obs.confidence:.2f} {state}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            out = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            out.header = msg.header
            self.debug.publish(out)
            if folder and self.frames % 30 == 0:
                session = Path(folder) / getattr(self, 'capture_session', 'session')
                session.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(session/f'camera_{self.frames:06d}.png'), frame)
                cv2.imwrite(str(session/f'debug_{self.frames:06d}.png'), annotated)
                snapshot = {'capture_format': 'raw_bgr8_v1',
                            'controller_revision': getattr(self,'controller_revision','unknown'),
                            'time': path_now, 'pose': self.pose, 'observation_pose': observation_pose, 'control_dt': dt, 'speed': speed, 'yaw': yaw,
                            'confidence': obs.confidence, 'state': state,
                            'processing_seconds': control_processing_seconds,
                            'control_gap': dt>.1,
                            'steering_lookahead': self.ground_path.lookahead,
                            'target_curvature': self.ground_path.target_curvature,
                            'straight_alignment_frames': self.ground_path.straight_alignment_frames,
                            'straight_speed_active': bool(self.ground_path.straight_speed_active),
                            'terminal_candidate': obs.terminal,
                            'terminal_frames': self.terminal_frames,
                            'travelled_distance': self.distance,
                            'terminal_approach_state': finish.state,
                            'terminal_remaining': finish.remaining,
                            'terminal_goal': None if finish.goal is None else finish.goal.tolist(),
                            'stop_reason': self.ground_path.stop_reason,
                            'recovery_reason': recovery.reason,
                            'recovery_state': recovery.state,
                            'conflict_recovery_state': conflict_recovery.state,
                            'conflict_recovery_kind': conflict_recovery.kind,
                            'recovery_remaining_at_start': conflict_recovery.remaining_at_start,
                            'conflict_recovery_reason': conflict_recovery.reason,
                            'conflict_recovery_heading': conflict_recovery.heading,
                            'conflict_recovery_rotation': conflict_recovery.rotation,
                            'conflict_recovery_support_frames': conflict_recovery.support_frames,
                            'path_memory_seconds': self.ground_path.memory_seconds,
                            'path_updated': self.ground_path.updated,
                            'path_observed_at': self.ground_path.observed_at().tolist(),
                            'path_valid_until': self.ground_path.deadlines().tolist(),
                            'path_update_reason': self.ground_path.update_reason,
                            'observed_path_length': self.ground_path.observed_length,
                            'path_confidence': self.ground_path.path_confidence,
                            'round_tracking': self.ground_path.round_tracking,
                            'alignment_frames': self.ground_path.alignment_frames,
                            'route_phase': self.ground_path.progress.phase,
                            'route_reversal': self.ground_path.progress.blocked,
                            'unverified_frames': self.ground_path.unverified_frames,
                            'unverified_since': self.ground_path.unverified_since,
                            'visual_conflict': self.ground_path.conflict_latched,
                            'conflict_frames': self.ground_path.conflict_frames,
                            'visible_path_support': self.ground_path.visible_support,
                            'splice_error': self.ground_path.splice_error,
                            'corner_heading': recovery.heading,
                            'corner_position': None if recovery.corner is None else recovery.corner.tolist(),
                            'path': None if self.ground_path.points is None else self.ground_path.points.tolist()}
                (session/f'camera_{self.frames:06d}.json').write_text(json.dumps(snapshot))

    def record_finish(self, frame, pose, now):
        self.get_logger().info(f'Terminal bar reached; stopping. remaining={self.finish.remaining:.3f} m')
        folder = self.get_parameter('capture_dir').value
        if not folder:
            return
        try:
            session = Path(folder)/getattr(self,'capture_session','session')
            session.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(session/'terminal_stop.png'),frame)
            (session/'terminal_stop.json').write_text(json.dumps({
                'state':'finished', 'stop_reason':'terminal_t',
                'time':now, 'pose':pose, 'distance':self.distance,
                'terminal_frames':self.terminal_confirmed_frames,
                'terminal_approach_started':self.finish.started,
                'terminal_remaining':self.finish.remaining,
                'terminal_goal':self.finish.goal.tolist(),
                'controller_revision':getattr(self,'controller_revision','unknown')}))
        except OSError as exc:
            self.get_logger().warning(f'Could not save terminal stop: {exc}')


def main(args=None):
    spin_follower(Follower, args)


def spin_follower(node_factory, args=None):
    rclpy.init(args=args)
    node = node_factory()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok(context=node.context):
                try:
                    node.send()
                except RCLError:
                    # SIGINT may invalidate the context between ok() and
                    # publish(). Only suppress this shutdown race.
                    if rclpy.ok(context=node.context):
                        raise
        finally:
            try:
                node.destroy_node()
            finally:
                rclpy.try_shutdown()
