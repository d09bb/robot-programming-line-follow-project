"""ROS and capture adapter for the tracker from the supplied robot_project1.zip."""
import bisect
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

from .node import spin_follower
from .reference_tracker import Tracker


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class ReferenceFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.delay = self.declare_parameter('start_delay_sec', 20.).value
        top_speed = self.declare_parameter('straight_speed', .45).value
        capture_dir = self.declare_parameter('capture_dir', '').value
        self.tracker = Tracker(v_max=top_speed, a_lat=.45, a_brake=.4, threshold=80)
        self.session = Path(capture_dir)/str(time.time_ns()) if capture_dir else None
        source = b''.join(Path(__file__).with_name(name).read_bytes()
                          for name in ('reference_node.py', 'reference_tracker.py'))
        self.controller_revision = hashlib.sha256(source).hexdigest()[:12]
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug = self.create_publisher(Image, '/line_follower/debug', qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self.on_imu, qos_profile_sensor_data)
        self.create_timer(.05, self.watchdog)
        self.t_launch = time.monotonic()
        self.odom_t, self.odom_p = [], []
        self.imu_t, self.imu_y = [], []
        self.pitch_up = self.roll = 0.
        self.imu_yaw = None
        self.last_wheel = None
        self.last_hy = self.yaw0 = 0.
        self.dr = (0., 0.)
        self.last_img = self.last_imu = self.last_odom = -math.inf
        self.last_odom_stamp = None
        self.last_t = self.start_sim = self.start_wall = None
        self.odom_speed = None
        self.frames = 0
        self.done = False
        self.get_logger().info(f'TRACK_READY controller={self.controller_revision}')

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        # Wheel odometry heading drifts badly when the wheels slip during
        # fast turns, so heading comes from the IMU; odometry supplies only
        # the travelled distance along that heading.
        px, py = m.pose.pose.position.x, m.pose.pose.position.y
        if self.imu_yaw is None:
            return
        ts = _stamp(m)
        self.last_odom_stamp = ts
        self.last_odom = time.monotonic()
        self.odom_speed = m.twist.twist.linear.x
        j = bisect.bisect_left(self.imu_t, ts)
        imu_yaw = self.imu_y[min(j, len(self.imu_y) - 1)]
        if self.last_wheel is None:
            self.yaw0 = imu_yaw
            self.dr = (0.0, 0.0)
        else:
            lx, ly, lyaw = self.last_wheel
            ds = (px - lx) * math.cos(lyaw) + (py - ly) * math.sin(lyaw)
            hy = 0.5 * (imu_yaw + self.last_hy) - self.yaw0 if abs(imu_yaw - self.last_hy) < 1 else imu_yaw - self.yaw0
            self.dr = (self.dr[0] + ds * math.cos(hy), self.dr[1] + ds * math.sin(hy))
        self.last_wheel = (px, py, yaw)
        self.last_hy = imu_yaw
        hy = math.atan2(math.sin(imu_yaw - self.yaw0), math.cos(imu_yaw - self.yaw0))
        self.odom_t.append(ts)
        self.odom_p.append((self.dr[0], self.dr[1], hy))
        if len(self.odom_t) > 120:
            del self.odom_t[:60]; del self.odom_p[:60]

    def on_imu(self, m):
        if m.orientation_covariance[0] < 0.:
            return
        self.last_imu = time.monotonic()
        q = m.orientation
        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        self.pitch_up = 0.8 * self.pitch_up + 0.2 * (-pitch)
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        self.roll = 0.8 * self.roll + 0.2 * roll
        self.imu_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.imu_t.append(_stamp(m)); self.imu_y.append(self.imu_yaw)
        if len(self.imu_t) > 400:
            del self.imu_t[:200]; del self.imu_y[:200]

    def pose_at(self, t):
        if not self.odom_t:
            return None
        i = bisect.bisect_left(self.odom_t, t)
        if i <= 0:
            return self.odom_p[0]
        if i >= len(self.odom_t):
            return self.odom_p[-1]
        t0, t1 = self.odom_t[i - 1], self.odom_t[i]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        p0, p1 = self.odom_p[i - 1], self.odom_p[i]
        dy = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
        return (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * dy)

    def send(self, speed=0., yaw=0.):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(speed), float(yaw)
        self.pub.publish(msg)

    def on_image(self, msg):
        now = time.monotonic()
        stamp = _stamp(msg)
        pose = self.pose_at(stamp)
        if (self.done or pose is None or now-self.last_odom > .5
                or now-self.last_imu > .5 or abs(self.last_odom_stamp-stamp) > .5
                or (self.last_t is not None and stamp <= self.last_t)):
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warning(f'Camera conversion failed: {exc}')
            self.send()
            return
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480))
        self.last_img = now
        self.tracker.integrate(frame, pose, self.pitch_up, self.roll)
        if now-self.t_launch < self.delay:
            return
        dt = .033 if self.last_t is None else max(.01, min(.2, stamp-self.last_t))
        self.last_t = stamp
        self.tracker.pitch_now = self.pitch_up
        speed, yaw = self.tracker.control(self.odom_p[-1], dt)
        self.send(speed, yaw)
        self.frames += 1
        if self.start_sim is None and speed > 0.:
            self.start_sim, self.start_wall = stamp, now
        self.done = self.tracker.finished
        state = ('finished' if self.done else
                 'terminal_approach' if self.tracker.goal_axis is not None else
                 'path_stop' if self.tracker.recovery_aborted else
                 'corner_recovery' if self.tracker.recovering else 'tracking')
        stop_reason = ('terminal_t' if self.done else
                       'recovery_limit' if self.tracker.recovery_aborted else '')
        if self.done:
            self.get_logger().info(
                f'TRACK_FINISHED sim_elapsed={stamp-self.start_sim:.2f}s '
                f'elapsed={now-self.start_wall:.2f}s')
        snapshot = {
            'frame': self.frames, 'time': stamp, 'dt': dt,
            'pose': pose, 'travelled_distance': self.tracker.travel,
            'distance': self.tracker.travel, 'speed': speed, 'yaw': yaw,
            'odom_speed': self.odom_speed, 'state': state, 'stop_reason': stop_reason,
            'path_update_reason': self.tracker.end_kind or '', 'forward_limit': '',
            'exit_straight_time': self.tracker.exit_straight_time,
            'controller_revision': self.controller_revision,
        }
        if self.done:
            snapshot['elapsed_sim'] = stamp-self.start_sim
            snapshot['elapsed_wall'] = now-self.start_wall
            snapshot['terminal_goal'] = self.tracker.goal
        if self.frames % 30 == 0 or self.done:
            debug = frame.copy()
            cv2.putText(debug, f'{state}  {speed:.3f} m/s', (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 180, 0), 2)
            self.debug.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        self.capture(frame, snapshot)

    def capture(self, frame, snapshot):
        if self.session is None:
            return
        try:
            self.session.mkdir(parents=True, exist_ok=True)
            with (self.session/'control.jsonl').open('a') as trace:
                trace.write(json.dumps(snapshot)+'\n')
            if self.frames % 30 == 0 or self.done:
                stem = 'terminal_stop' if self.done else f'camera_{self.frames:06}'
                cv2.imwrite(str(self.session/f'{stem}.png'), frame)
                detail = dict(snapshot, path=[list(point) for point in self.tracker.path],
                              pitch=self.pitch_up)
                (self.session/f'{stem}.json').write_text(json.dumps(detail))
        except (OSError, cv2.error) as exc:
            self.get_logger().warning(f'Capture disabled: {exc}')
            self.session = None

    def watchdog(self):
        now = time.monotonic()
        if self.done or now-min(self.last_img, self.last_odom, self.last_imu) > .5:
            self.tracker.v_cmd = self.tracker.w_cmd = 0.
            self.tracker.exit_window = self.tracker.exit_straight_time = 0.
            self.send()


def main(args=None):
    spin_follower(ReferenceFollower, args)
