# Adapted from project1_memory_v2/line_tracker.py in the supplied robot_project1.zip (Apache-2.0 package).
"""Camera line tracker core (no ROS).  Bird's-eye projection of the camera
image is fused into a short-term odometry-frame map so that the line under and
just in front of the robot (outside the camera view) is remembered.  The line
is traced in that map from the robot forward, and a pure-pursuit controller
with curvature-based speed planning follows the traced path."""
import math
import numpy as np
import cv2

FX = 320.0 / math.tan(1.085595 / 2.0)
CX, CY = 320.0, 240.0
CAM_X, CAM_Z = 0.076, 0.103

BEV_X0, BEV_X1, BEV_Y = 0.235, 1.15, 0.60
BEV_RES = 0.005
MAP_RES = 0.005
MAP_N = 2800            # 14 m square map centred on the odometry origin


class Tracker:
    def __init__(self, **kw):
        self.v_max = kw.get('v_max', 0.45)
        self.a_lat = kw.get('a_lat', 0.45)
        self.a_brake = kw.get('a_brake', 0.4)
        self.v_min = kw.get('v_min', 0.085)
        self.w_max = kw.get('w_max', 1.2)
        self.w_turn = kw.get('w_turn', 1.0)
        self.w_acc = kw.get('w_acc', 3.0)
        self.w_cmd = 0.0
        self.has_map = False
        self.hist = []
        self.pitch_now = 0.0
        self.hill_hold = 0.0
        self.v_hill = kw.get('v_hill', 0.32)
        self.travel = 0.0
        self.min_travel = kw.get('min_travel', 11.0)
        self.C = (0.0, 0.0, 0.0)       # SE2 correction applied to odometry poses
        self.reg_max_x = kw.get('reg_max_x', 0.75)
        self.max_x = kw.get('max_x', 0.75)
        self.use_reg = bool(kw.get('use_reg', 0))
        self.near_x = kw.get('near_x', 0.50)
        self.acc = kw.get('acc', 0.5)
        self.threshold = kw.get('threshold', 80)
        self.roll_sign = kw.get('roll_sign', 0.0)
        self.nx = int(round((BEV_X1 - BEV_X0) / BEV_RES))
        self.ny = int(round(2 * BEV_Y / BEV_RES))
        self.line = np.zeros((MAP_N, MAP_N), np.float32)
        self.seen = np.zeros((MAP_N, MAP_N), np.uint8)
        self._pitch_maps = {}
        self.v_cmd = 0.0
        self.finished = False
        self.path = []
        self.end_kind = None
        self.last_alpha = 0.0
        self.goal = None
        self.goal_axis = None
        self.goal_cand = None
        self.goal_hits = 0
        # A missing camera segment must never turn into an unbounded spin.
        # These values are advanced with simulation time by control().
        self.recovering = False
        self.recover_yaw = 0.0
        self.recover_sign = 1.0
        self.recover_elapsed = 0.0
        self.recover_limit = kw.get('recover_limit', 1.15)
        self.recover_timeout = kw.get('recover_timeout', 3.0)
        self.lost_events = 0
        self.recovery_aborted = False
        self.exit_window = 0.0
        self.exit_straight_time = 0.0

    # ---------- projection ----------
    def _maps(self, pitch, roll=0.0):
        key = (int(round(pitch / 0.004)), int(round(roll * self.roll_sign / 0.004)))
        m = self._pitch_maps.get(key)
        if m is not None:
            return m
        p = key[0] * 0.004
        rl = key[1] * 0.004
        gx = BEV_X0 + (np.arange(self.nx) + 0.5) * BEV_RES          # forward
        gy = BEV_Y - (np.arange(self.ny) + 0.5) * BEV_RES           # left
        GX, GY = np.meshgrid(gx, gy)        # rows: left->right, cols: near->far
        # camera-frame vector to ground point (f, l, u), then undo pitch
        f = GX - CAM_X
        l = GY
        u = -CAM_Z * np.ones_like(f)
        cp, sp = math.cos(p), math.sin(p)
        fc = f * cp + u * sp
        uc = -f * sp + u * cp
        cr, sr = math.cos(rl), math.sin(rl)
        l, uc = l * cr + uc * sr, -l * sr + uc * cr
        valid = fc > 0.05
        fc = np.where(valid, fc, 1.0)
        px = CX - FX * l / fc
        py = CY - FX * uc / fc
        valid &= (px >= 0) & (px < 639) & (py >= 0) & (py < 479)
        m = (px.astype(np.float32), py.astype(np.float32), valid.astype(np.uint8))
        if len(self._pitch_maps) > 200:
            self._pitch_maps.clear()
        self._pitch_maps[key] = m
        return m

    def corrected(self, pose):
        tx, ty, th = self.C
        c, s = math.cos(th), math.sin(th)
        x, y, yaw = pose
        return (c * x - s * y + tx, s * x + c * y + ty, yaw + th)

    def _register(self, dark, pose):
        """Align the current bird's-eye line pixels with the remembered map
        (small search) and fold the best offset into the correction C."""
        rows, cols = np.nonzero(dark > 0)
        if len(rows) < 60:
            return pose
        gx = BEV_X0 + (cols + 0.5) * BEV_RES
        gy = BEV_Y - (rows + 0.5) * BEV_RES
        keep = gx < self.reg_max_x
        gx, gy = gx[keep], gy[keep]
        if len(gx) < 60:
            return pose
        if len(gx) > 500:
            sel = np.linspace(0, len(gx) - 1, 500).astype(int)
            gx, gy = gx[sel], gy[sel]
        x, y, yaw = pose
        # region blur for a smooth score
        r = int(1.2 / MAP_RES)
        cx0 = max(0, min(MAP_N - 2 * r, int(x / MAP_RES + MAP_N / 2) - r))
        cy0 = max(0, min(MAP_N - 2 * r, int(y / MAP_RES + MAP_N / 2) - r))
        L = cv2.GaussianBlur(self.line[cy0:cy0 + 2 * r, cx0:cx0 + 2 * r], (0, 0), 1.5)
        S = self.seen[cy0:cy0 + 2 * r, cx0:cx0 + 2 * r]
        best = None
        base = None
        d = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
        dt = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
        for th in dt:
            c, s_ = math.cos(yaw + th), math.sin(yaw + th)
            wx0 = x + c * gx - s_ * gy
            wy0 = y + s_ * gx + c * gy
            for dx in d:
                for dy in d:
                    j = ((wx0 + dx) / MAP_RES + MAP_N / 2).astype(int) - cx0
                    i = ((wy0 + dy) / MAP_RES + MAP_N / 2).astype(int) - cy0
                    ok = (i >= 0) & (i < 2 * r) & (j >= 0) & (j < 2 * r)
                    i, j = i[ok], j[ok]
                    sn = S[i, j] > 0
                    if sn.sum() < 40:
                        continue
                    sc = float(L[i[sn], j[sn]].mean())
                    if best is None or sc > best[0] + 1e-6 or (abs(sc - best[0]) <= 1e-6 and abs(dx) + abs(dy) + abs(th) < best[4]):
                        best = (sc, dx, dy, th, abs(dx) + abs(dy) + abs(th))
                    if dx == 0 and dy == 0 and th == 0:
                        base = sc
        if best is None or base is None or best[0] < 0.25 or best[0] < base + 0.01:
            return pose
        _, dx, dy, th, _ = best
        # new correction: rotate about the robot by th, then shift
        tx, ty, th0 = self.C
        c, s_ = math.cos(th), math.sin(th)
        ntx = c * (tx - x) - s_ * (ty - y) + x + dx
        nty = s_ * (tx - x) + c * (ty - y) + y + dy
        self.C = (ntx, nty, th0 + th)
        return (x + dx, y + dy, yaw + th)

    def integrate(self, bgr, pose, pitch=0.0, roll=0.0):
        pose = self.corrected(pose)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        mx, my, valid = self._maps(pitch, roll)
        bev = cv2.remap(gray, mx, my, cv2.INTER_LINEAR)
        valid = valid.copy()
        valid[:, int((self.max_x - BEV_X0) / BEV_RES):] = 0
        dark = ((bev < self.threshold) & (valid > 0)).astype(np.float32)
        if self.use_reg and self.has_map:
            pose = self._register(dark, pose)
        x, y, yaw = pose
        # affine BEV pixel (col=c, row=r) -> map pixel
        c, s = math.cos(yaw), math.sin(yaw)
        # bev col -> forward gx = X0 + (c+.5)res ; row -> left gy = Y - (r+.5)res
        # world = pose + R*[gx,gy]; map col = wx/MR + N/2, map row = wy/MR + N/2
        k = BEV_RES / MAP_RES
        A = np.array([
            [c * k, s * k, (x + c * (BEV_X0 + 0.5 * BEV_RES) - s * (BEV_Y - 0.5 * BEV_RES)) / MAP_RES + MAP_N / 2],
            [s * k, -c * k, (y + s * (BEV_X0 + 0.5 * BEV_RES) + c * (BEV_Y - 0.5 * BEV_RES)) / MAP_RES + MAP_N / 2],
        ], np.float32)
        # restrict to a region of interest around the robot for speed
        r = int(1.3 / MAP_RES)
        cx0 = int(x / MAP_RES + MAP_N / 2) - r
        cy0 = int(y / MAP_RES + MAP_N / 2) - r
        cx0 = max(0, min(MAP_N - 2 * r, cx0)); cy0 = max(0, min(MAP_N - 2 * r, cy0))
        A2 = A.copy(); A2[0, 2] -= cx0; A2[1, 2] -= cy0
        w = cv2.warpAffine(dark, A2, (2 * r, 2 * r), flags=cv2.INTER_LINEAR, borderValue=0)
        vm = cv2.warpAffine(valid.astype(np.float32), A2, (2 * r, 2 * r), flags=cv2.INTER_NEAREST, borderValue=0)
        nb = valid.astype(np.float32).copy()
        nb[:, int((self.near_x - BEV_X0) / BEV_RES):] = 0
        near = cv2.warpAffine(nb, A2, (2 * r, 2 * r), flags=cv2.INTER_NEAREST, borderValue=0)
        # only trust the near part strongly; overwrite seen cells
        sl = (slice(cy0, cy0 + 2 * r), slice(cx0, cx0 + 2 * r))
        sel = vm > 0.5
        L = self.line[sl]
        # presence is written at once; absence only fades old evidence, so a
        # single badly projected frame cannot erase a remembered line
        nw = w[sel]
        nr = near[sel] > 0.5
        old = L[sel]
        # near band (accurate projection) overwrites quickly; the far band only
        # adds weak evidence and fades slowly.
        L[sel] = np.where(nr, np.where(nw > 0.5, np.maximum(nw, old), old * 0.55 + nw * 0.45),
                          np.where(nw > 0.5, np.maximum(old, 0.6 * nw + 0.4 * old), old * 0.95 + nw * 0.05))
        self.seen[sl][sel] = 1
        self.has_map = True
        return dark

    # ---------- map sampling ----------
    def _sample(self, px, py):
        i = (np.asarray(py) / MAP_RES + MAP_N / 2).astype(int)
        j = (np.asarray(px) / MAP_RES + MAP_N / 2).astype(int)
        i = np.clip(i, 0, MAP_N - 1); j = np.clip(j, 0, MAP_N - 1)
        return self.line[i, j], self.seen[i, j]

    def _recenter(self, p, d):
        nx, ny = -math.sin(d), math.cos(d)
        offs = np.linspace(-0.03, 0.03, 13)
        v, _ = self._sample(p[0] + offs * nx, p[1] + offs * ny)
        if v.sum() < 0.5:
            return p
        o = float((v * offs).sum() / v.sum())
        return (p[0] + o * nx, p[1] + o * ny)

    def _is_tbar(self, p, d):
        """A real T: a crossbar of line on both sides (>= 6 cm each) and
        no line continuing beyond it."""
        nx, ny = -math.sin(d), math.cos(d)
        offs = np.r_[np.linspace(-0.08, -0.03, 6), np.linspace(0.03, 0.08, 6)]
        best = 0.0
        for fwd in (0.0, 0.01, 0.02, 0.03):
            bx = p[0] + fwd * math.cos(d) + offs * nx
            by = p[1] + fwd * math.sin(d) + offs * ny
            v, _ = self._sample(bx, by)
            best = max(best, min(v[:6].mean(), v[6:].mean()))
        if best < 0.6:
            return False
        ax = p[0] + np.linspace(0.06, 0.15, 6) * math.cos(d)
        ay = p[1] + np.linspace(0.06, 0.15, 6) * math.sin(d)
        v, sn = self._sample(ax, ay)
        return v.max() < 0.3 and sn.min() > 0

    def trace(self, pose, max_len=1.3, step=0.02):
        x, y, yaw = pose
        angs = np.radians(np.arange(-105, 106, 5))
        # starting point: nearest line along a short forward fan
        start = None
        for fwd in np.arange(0.0, 0.5, 0.01):
            offs = np.linspace(-0.06, 0.06, 13)
            px = x + fwd * math.cos(yaw) - offs * math.sin(yaw)
            py = y + fwd * math.sin(yaw) + offs * math.cos(yaw)
            v, sn = self._sample(px, py)
            if (v > 0.5).any():
                k = np.flatnonzero(v > 0.5)
                # the run of line pixels closest to the robot's centreline
                runs = np.split(k, np.flatnonzero(np.diff(k) > 1) + 1)
                o = min((float(offs[r_].mean()) for r_ in runs), key=abs)
                start = (x + fwd * math.cos(yaw) - o * math.sin(yaw),
                         y + fwd * math.sin(yaw) + o * math.cos(yaw))
                break
        if start is None:
            return [], 'none'
        old = None
        if len(self.hist) > 40:
            H = np.array(self.hist[:-30])
            near = np.hypot(H[:, 0] - x, H[:, 1] - y) < 1.6
            if near.any():
                old = H[near]
        p = start; d = yaw
        pts = [p]
        total = 0.0
        kind = 'len'
        rs = np.array([0.02, 0.03, 0.04, 0.05])
        gap = 0.0
        while total < max_len:
            cand = d + angs
            ex = p[0] + np.outer(np.cos(cand), rs)
            ey = p[1] + np.outer(np.sin(cand), rs)
            v, sn = self._sample(ex, ey)
            score = v.mean(axis=1)
            seen = sn.min(axis=1)
            ok = score > 0.5
            if old is not None and ok.any():
                # never trace back along line the robot has already driven
                qx = p[0] + 0.03 * np.cos(cand); qy = p[1] + 0.03 * np.sin(cand)
                dd = np.hypot(qx[:, None] - old[None, :, 0], qy[:, None] - old[None, :, 1])
                close = dd < 0.05
                if close.any():
                    dots = np.cos(cand[:, None] - old[None, :, 2])
                    back = (close & (dots < -0.3)).any(axis=1)
                    ok &= ~back
            if not ok.any():
                if seen[np.abs(angs) < 0.6].min() == 0 and gap < 0.30:
                    # unobserved cells ahead (outside the camera view so
                    # far): bridge the gap straight on.
                    p = (p[0] + step * math.cos(d), p[1] + step * math.sin(d))
                    pts.append(p); total += step; gap += step
                    continue
                # short seen gap (map misregistration): look a little further
                jumped = False
                if gap < 0.001:
                    for dist in (0.06, 0.09, 0.12, 0.15):
                        offs = np.linspace(-0.08, 0.08, 33)
                        qx = p[0] + dist * math.cos(d) - offs * math.sin(d)
                        qy = p[1] + dist * math.sin(d) + offs * math.cos(d)
                        vv, _ = self._sample(qx, qy)
                        if (vv > 0.5).any():
                            kk = np.flatnonzero(vv > 0.5)
                            o = float(offs[kk].mean())
                            q = (p[0] + dist * math.cos(d) - o * math.sin(d), p[1] + dist * math.sin(d) + o * math.cos(d))
                            d = math.atan2(q[1] - p[1], q[0] - p[0])
                            p = q; pts.append(p); total += dist; jumped = True
                            break
                if jumped:
                    continue
                kind = 'unknown' if seen[np.abs(angs) < 0.6].min() == 0 else 'end'
                break
            gap = 0.0
            # T junction: straight missing, both sides present -> end
            straight = ok[np.abs(angs) <= np.radians(30)].any()
            left = ok[(angs > np.radians(60))].any()
            right = ok[(angs < -np.radians(60))].any()
            if not straight and left and right and self._is_tbar(p, d):
                kind = 'T'
                break
            idx = np.flatnonzero(ok)
            # contiguous group closest to straight; aim at its centre
            best = idx[np.argmin(np.abs(angs[idx]))]
            grp = [best]
            j = best - 1
            while j >= 0 and ok[j]:
                grp.append(j); j -= 1
            j = best + 1
            while j < len(ok) and ok[j]:
                grp.append(j); j += 1
            # if group is very wide (crossing / blob) prefer straight-most
            if len(grp) > 9:
                a = angs[best]
            else:
                a = float(np.mean(angs[grp]))
            d = d + a
            p = (p[0] + step * math.cos(d), p[1] + step * math.sin(d))
            p = self._recenter(p, d)
            pts.append(p)
            total += step
        self.end_kind = kind
        return pts, kind

    # ---------- control ----------
    def control(self, pose, dt):
        if self.finished:
            return 0.0, 0.0
        pose = self.corrected(pose)
        x, y, yaw = pose
        if not self.hist or math.hypot(x - self.hist[-1][0], y - self.hist[-1][1]) > 0.02:
            if self.hist:
                self.travel += math.hypot(x - self.hist[-1][0], y - self.hist[-1][1])
            self.hist.append((x, y, yaw))
        if self.goal_axis is not None:
            return self.approach_goal(pose, dt)
        pts, kind = self.trace(pose)
        self.path = pts
        if len(pts) < 2:
            self.exit_window = self.exit_straight_time = 0.0
            # Turn only towards the last observed path, and only through a
            # bounded angle/time.  The old behaviour repeatedly issued a
            # full-rate spin here, which made a temporary camera miss at a
            # corner unrecoverable.
            if not self.recovering:
                self.recovering = True
                self.recovery_aborted = False
                self.recover_yaw = yaw
                self.recover_sign = 1.0 if self.last_alpha >= 0.0 else -1.0
                self.recover_elapsed = 0.0
                self.lost_events += 1
            self.recover_elapsed += dt
            turned = abs(math.atan2(math.sin(yaw - self.recover_yaw),
                                    math.cos(yaw - self.recover_yaw)))
            self.v_cmd = 0.0
            if turned >= self.recover_limit or self.recover_elapsed >= self.recover_timeout:
                self.recovery_aborted = True
                self.w_cmd = 0.0
                return 0.0, 0.0
            target_w = self.recover_sign * min(0.55, self.w_turn)
            dw = self.w_acc * dt
            self.w_cmd = max(self.w_cmd - dw, min(self.w_cmd + dw, target_w))
            # A very small forward creep lets the near camera band reacquire
            # the line without driving across the corner while searching.
            return 0.025, self.w_cmd
        if self.recovering:
            self.recovering = False
            self.recovery_aborted = False
            self.recover_elapsed = 0.0
        P = np.array(pts)
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        s = np.r_[0.0, np.cumsum(seg)]
        # distance from robot to path start
        s0 = math.hypot(P[0, 0] - x, P[0, 1] - y)
        s = s + s0
        # smooth the traced points before estimating curvature: the 2 cm
        # trace steps jitter by a few mm, which would look like tight bends
        if len(P) >= 5:
            k3 = np.ones(3) / 3.0
            Ps = np.c_[np.convolve(np.pad(P[:, 0], 1, mode='edge'), k3, 'valid'),
                       np.convolve(np.pad(P[:, 1], 1, mode='edge'), k3, 'valid')]
        else:
            Ps = P
        hd = np.unwrap(np.arctan2(np.diff(Ps[:, 1]), np.diff(Ps[:, 0])))
        # curvature over ~8 cm windows
        w = 4
        vlim = self.v_max
        for i in range(1, len(hd)):
            j = min(len(hd) - 1, i + w)
            if j <= i:
                break
            dh = abs(hd[j] - hd[i - 1])
            ds = max(1e-3, s[min(j, len(s) - 1)] - s[i - 1])
            kap = dh / ds
            vi = math.sqrt(self.a_lat / kap) if kap > 1e-3 else self.v_max
            if kap > 1e-3:
                vi = min(vi, self.w_turn / kap)
            if dh > 0.9:      # sharp corner: nearly stop at it
                vi = self.v_min
            vi = max(self.v_min, min(self.v_max, vi))
            vlim = min(vlim, math.sqrt(vi * vi + 2 * self.a_brake * max(0.0, s[i - 1] - 0.05)))
        end_s = s[-1]
        if kind == 'T':
            e = (float(P[-1, 0]), float(P[-1, 1]))
            if self.goal_cand is not None and math.hypot(e[0] - self.goal_cand[0], e[1] - self.goal_cand[1]) < 0.06:
                self.goal_hits += 1
            else:
                self.goal_hits = 1
            self.goal_cand = e
            if self.goal_hits >= 3 and end_s < 1.0 and self.travel > self.min_travel:
                # Fit the incoming stem before the last few samples, where
                # tracing can start following the transverse terminal bar.
                stem = P[(s >= end_s-.32) & (s <= end_s-.08)]
                if len(stem) >= 6:
                    axis = stem[-1]-stem[0]
                    length = np.linalg.norm(axis)
                    if length >= .12:
                        axis /= length
                        normal = np.array([-axis[1], axis[0]])
                        lateral = abs((np.array(e)-stem[0])@normal)
                        deviation = np.max(np.abs((stem-stem[0])@normal))
                        if lateral <= .035 and deviation <= .015 and axis@[math.cos(yaw), math.sin(yaw)] >= .96:
                            self.goal = e
                            self.goal_axis = axis
                            return self.approach_goal(pose, dt)
        if kind == 'T':
            pass
        elif kind in ('unknown', 'none', 'end'):
            vlim = min(vlim, math.sqrt(0.15 ** 2 + 2 * self.a_brake * max(0.0, end_s - 0.05)))
            # The camera horizon may end exactly where a tight corner starts.
            # Do not accelerate into that unconfirmed frontier; the cap is
            # deliberately independent of callback frequency.
            if end_s < 0.72:
                vlim = min(vlim, 0.18 if kind != 'unknown' else 0.24)
        # pure pursuit
        v_now = max(self.v_cmd, 0.05)
        Ld = min(0.30, max(0.07, 0.05 + 0.35 * v_now))
        # A longer preview is safe only when the observed local path is
        # geometrically straight. Inspect 55 cm, well beyond the target, so
        # a visible corner restores the original short pursuit immediately.
        # No temporal steering filter: corners retain their original authority.
        preview = P[s <= 0.55]
        straight_weight = 0.0
        if len(preview) >= 15 and s[min(len(preview)-1, len(s)-1)] >= 0.40:
            origin = preview.mean(axis=0)
            _, _, axes = np.linalg.svd(preview - origin, full_matrices=False)
            residual = float(np.max(np.abs((preview-origin) @ axes[1])))
            straight_weight = float(np.clip((0.018-residual)/0.010, 0.0, 1.0))
            # A straight trace can be an unobserved gap or a different branch
            # seen during a turn. Extend only after heading capture and only
            # when the preview itself is supported by camera map evidence.
            tangent = preview[-1] - preview[0]
            heading_error = math.atan2(math.sin(math.atan2(tangent[1], tangent[0])-yaw),
                                       math.cos(math.atan2(tangent[1], tangent[0])-yaw))
            evidence, observed = self._sample(preview[:, 0], preview[:, 1])
            supported = np.mean((evidence > 0.5) & (observed > 0)) >= 0.90
            if abs(heading_error) > 0.15 or not supported:
                straight_weight = 0.0
        Ld += straight_weight * max(0.0, 0.34 - Ld)
        self.straight_weight = straight_weight
        # Interpolate by arc length rather than jumping between 2 cm samples.
        if straight_weight > 0.0:
            tx = float(np.interp(Ld, s, P[:, 0]))
            ty = float(np.interp(Ld, s, P[:, 1]))
            # Spatial fit removes map-grid/trace-step jitter without delaying
            # steering. Blend only under the same observed-straight gate.
            target = np.array([tx, ty])
            fitted = origin + axes[0] * float((target - origin) @ axes[0])
            tx, ty = target + straight_weight * (fitted - target)
        else:
            # Keep the established corner controller bit-for-bit: even the
            # original discrete target matters in very tight turns.
            k = min(np.searchsorted(s, Ld), len(P)-1)
            tx, ty = P[k]
        dx, dy = tx - x, ty - y
        xr = math.cos(yaw) * dx + math.sin(yaw) * dy
        yr = -math.sin(yaw) * dx + math.cos(yaw) * dy
        alpha = math.atan2(yr, xr)
        dist = math.hypot(xr, yr)
        self.target_dist = dist
        self.last_alpha = alpha
        # on the hill the robot bounces at speed: hold a lower limit while the
        # body is pitched and for a short while after
        if abs(self.pitch_now) > 0.03:
            self.hill_hold = 0.4
        else:
            self.hill_hold = max(0.0, self.hill_hold - dt)
        if self.hill_hold > 0.0:
            vlim = min(vlim, self.v_hill)
        # target speed with acceleration limit
        v_t = vlim
        if abs(alpha) > 0.7:
            v_t = min(v_t, 0.05)
        # Build speed a little sooner after a turn once the existing visible
        # straight fit has stayed strong. All speed ceilings still apply.
        self.exit_window = 1.2 if abs(self.w_cmd) >= .45 else max(0., self.exit_window-dt)
        if (self.exit_window > 0. and straight_weight >= .8 and abs(alpha) <= .10
                and abs(self.w_cmd) <= .20 and kind != 'T'
                and self.hill_hold == 0.):
            self.exit_straight_time += dt
        else:
            self.exit_straight_time = 0.
        acceleration = self.acc*1.3 if self.exit_straight_time >= .15 else self.acc
        if v_t > self.v_cmd:
            self.v_cmd = min(v_t, self.v_cmd + acceleration * dt)
        else:
            self.v_cmd = max(v_t, self.v_cmd - 1.5 * self.a_brake * dt)
        # the pursuit arc must be drivable within the turn-rate limit
        kpp = 2.0 * abs(math.sin(alpha)) / max(dist, 0.05)
        if kpp > 1e-3 and self.v_cmd > self.w_max * 0.85 / kpp:
            self.v_cmd = max(self.v_min * 0.5, self.w_max * 0.85 / kpp)
        v = self.v_cmd
        if abs(alpha) > 0.7:
            w_ = 1.5 * alpha
        else:
            w_ = 2.0 * max(v, 0.08) * math.sin(alpha) / max(dist, 0.05)
        w_ = max(-self.w_max, min(self.w_max, w_))
        dw = self.w_acc * dt
        self.w_cmd = max(self.w_cmd - dw, min(self.w_cmd + dw, w_))
        return v, self.w_cmd

    def stop_at_goal(self):
        self.finished = True
        self.v_cmd = self.w_cmd = 0.
        return 0., 0.

    def approach_goal(self, pose, dt):
        """Follow the confirmed stem through the final braking approach."""
        axis = self.goal_axis
        normal = np.array([-axis[1], axis[0]])
        offset = np.asarray(pose[:2])-self.goal
        remaining = -float(offset@axis)
        lateral = float(offset@normal)
        error = math.atan2(math.sin(math.atan2(axis[1], axis[0])-pose[2]),
                           math.cos(math.atan2(axis[1], axis[0])-pose[2]))
        if remaining <= .015 and abs(lateral) <= .06:
            return self.stop_at_goal()
        if abs(lateral) > .08 or abs(error) > .45 or remaining <= 0.:
            self.v_cmd = self.w_cmd = 0.
            return 0., 0.
        limit = min(self.v_max, .22, math.sqrt(.60*(remaining-.01)))
        self.v_cmd = min(limit, self.v_cmd+self.acc*dt)
        self.w_cmd = float(np.clip(1.8*error-4.*lateral, -.25, .25))
        return self.v_cmd, self.w_cmd
