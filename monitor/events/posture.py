"""posture state machine (lying/reclined/sitting/standing/floor/absent)
events: bed_edge, bed_exit, fall, absent, help, back_in_bed
"""
from collections import deque

import numpy as np

from ..pose import (L_HIP, L_KN, L_SH, L_WR, NOSE, R_HIP, R_KN, R_SH, R_WR, angle_from_vertical, kp, mid, torso,
                    torso_view)

IN_BED = ("lying", "reclined")


class FastFall:
    """fall = upright -> low and wide within 2.5s, without sitting first. works from any angle.

    on UR Fall, falls end with w/h >= 1.5, height <= 0.66 of standing, head 0.75-1.2 standing
    heights lower. ADLs never get there. lying down on purpose goes through sitting, so the time
    limit handles that. tried "standing -> horizontal fast" too, false alarms on people lying on beds.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ref_h = None
        self.last_upright_t = None
        self.upright_head_y = None
        self.upright_heads = deque()
        self.head_track = deque()
        self.low_since = None
        self.strong_since = None
        self.last_low_t = -1e9
        self.sat_since_upright = False
        self.fired = False
        self.last_seen = None
        self.last_center = None
        self.dip = None  # (t0, deepest)
        self.near_fall = False
        self.near_falls = 0

    def update(self, t, person, state=None, since=None, in_zone=None):
        """True once when a fall is confirmed"""
        self.last_seen = t
        x1, y1, x2, y2 = person.box
        w, h = x2 - x1, y2 - y1
        if h <= 0:
            return False
        thr = self.cfg.kp_conf
        head = kp(person, NOSE, thr)
        if head is None:
            head = mid(person, L_SH, R_SH, thr)
        head_y = head[1] if head is not None else y1
        self.head_track.append((t, head_y))
        while self.head_track and self.head_track[0][0] < t - 2.0:
            self.head_track.popleft()
        if h / max(w, 1) > 1.6:
            self.ref_h = h if self.ref_h is None else max(0.95 * self.ref_h, h) if h > 0.6 * self.ref_h else self.ref_h
            if h >= 0.85 * self.ref_h:
                self.last_upright_t = t
                self.upright_heads.append((t, head_y))
                self.sat_since_upright = self.fired = False
        while self.upright_heads and self.upright_heads[0][0] < t - 3.0:
            self.upright_heads.popleft()
        if self.upright_heads:
            self.upright_head_y = min(y for _, y in self.upright_heads)
        # sat >= 1.5s before going down = lying down on purpose. mid-fall bodies flash "sitting"
        # for a frame or two so that alone can't count
        if state in ("sitting", "reclined", "lying") and self.last_upright_t is not None \
                and since is not None and since > self.last_upright_t \
                and (self.low_since or t) - since >= 1.5:
            self.sat_since_upright = True
        if self.ref_h is None or self.upright_head_y is None or self.fired:
            return False
        drop = (head_y - self.upright_head_y) / self.ref_h
        self._stumble(t, drop, h)
        low_wide = w / h >= 1.2 and h <= 0.7 * self.ref_h and drop >= 0.5
        # strong case: head dropped a full standing height + body wide. falls towards the camera
        # keep the box tall (perspective), fall-03/05/07 got missed with only the 0.7 limit.
        # kneeling also drops the head that much but the torso stays upright (falls 49-70 deg, kneeling 5-34)
        tr = torso(person, thr)
        tipped = tr is not None and angle_from_vertical(tr[1] - tr[0]) >= self.cfg.fall_strong_torso
        strong = (w / h >= self.cfg.fall_strong_wh and h <= 0.95 * self.ref_h
                  and drop >= self.cfg.fall_strong_drop and tipped)
        low_wide = low_wide or strong
        if in_zone is not None and in_zone(((x1 + x2) / 2, (y1 + y2) / 2)):
            low_wide = False
        if not low_wide:
            if self.low_since is not None and t - self.last_low_t > 0.5:
                self.low_since = None
            return False
        self.last_low_t = t
        self.low_since = self.low_since or t
        self.strong_since = (self.strong_since or t) if strong else None
        # 0.8s to confirm, 0.4s on strong evidence (fall-15 ends 0.4s after impact)
        held = t - self.low_since >= 0.8 or (self.strong_since is not None and t - self.strong_since >= 0.4)
        if (held and not self.sat_since_upright
                and self.low_since - self.last_upright_t <= 2.5):
            self.fired = True
            return True
        return False


    def _stumble(self, t, drop, h):
        """stumble: fast head dip while upright, back up within 1.5s. bending down is slower"""
        self.near_fall = False
        cfg = self.cfg
        if self.last_upright_t is None or t - self.last_upright_t > 2.0:
            self.dip = None
            return
        fast = [y for tt, y in self.head_track if tt >= t - 0.3]
        speed_drop = (self.head_track[-1][1] - min(fast)) / self.ref_h if len(fast) >= 2 else 0.0
        if self.dip is None:
            if speed_drop >= cfg.nearfall_drop and drop < 0.6:
                self.dip = (t, drop)
            return
        start, deepest = self.dip
        self.dip = (start, max(deepest, drop))
        if t - start > 1.5 or drop >= 0.6:
            self.dip = None
        elif drop <= 0.08 and h >= 0.85 * self.ref_h and deepest >= cfg.nearfall_drop:
            self.dip = None
            self.near_fall = True
            self.near_falls += 1

    def vanished_mid_fall(self, t):
        """person vanished while their head was coming down fast = fell out of view (fall-18/19/21)"""
        if self.fired or self.ref_h is None or self.last_seen is None or self.sat_since_upright:
            return False
        if self.last_upright_t is None or self.last_seen - self.last_upright_t > 1.0:
            return False
        recent = [(tt, y) for tt, y in self.head_track if tt >= self.last_seen - self.cfg.fall_vanish_window_s]
        if len(recent) < 3:
            return False
        ys = [y for _, y in recent]
        drop = (ys[-1] - min(ys)) / self.ref_h
        if drop >= self.cfg.fall_vanish_drop and t - self.last_seen >= self.cfg.fall_vanish_s:
            self.fired = True
            return True
        return False


class RoomFallWatch:
    """FastFall for everyone else in the room"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.people = {}

    def update(self, t, others, in_zone=None):
        fired = False
        visible = {p.track_id for p in others}
        for p in others:
            ff = self.people.get(p.track_id)
            if ff is None:
                ff = self._inherit(t, p, visible) or FastFall(self.cfg)
                self.people[p.track_id] = ff
            ff.last_center = p.center
            fired |= ff.update(t, p, in_zone=in_zone)
        for tid in [k for k, v in self.people.items() if v.last_seen is not None and t - v.last_seen > 10]:
            del self.people[tid]
        return fired

    def _inherit(self, t, p, visible):
        """tracker often gives a falling person a new id, so take over the history of whoever just vanished nearby"""
        best, best_d = None, None
        for tid, ff in list(self.people.items()):
            if tid in visible or ff.last_seen is None or t - ff.last_seen > 1.5 or ff.ref_h is None:
                continue
            d = np.linalg.norm(p.center - ff.last_center)
            if d < 1.2 * ff.ref_h and (best_d is None or d < best_d):
                best, best_d = tid, d
        return self.people.pop(best) if best is not None else None


class PostureTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bed_zone = None  # normalised, None = whole frame
        self.zone_auto = False
        self.reset()

    def reset(self):
        self.state = "unknown"
        self.since = None
        self.known = "unknown"  # last real posture
        self.unclear_since = None
        self.cand = None
        self.cand_since = None
        self.last_seen = None
        self.help_since = None
        self.help_active = False
        self._zone_boxes = []
        self.hip_hist = deque()
        self.sit_to_stand_s = None
        self._sts_pending = False
        self.fastfall = FastFall(self.cfg)
        self.hidden = False
        self.floor_long_fired = False

    def _floor_long(self, t):
        """still on the floor long after a fall"""
        if self.state != "floor":
            self.floor_long_fired = False
            return []
        if not self.floor_long_fired and t - self.since >= self.cfg.floor_long_s:
            self.floor_long_fired = True
            return ["floor_long"]
        return []

    def floor_seconds(self, t):
        return t - self.since if self.state == "floor" and self.since is not None else 0.0

    # ---- bed zone ----
    def set_zone(self, zone, auto=False):
        self.bed_zone = tuple(float(v) for v in zone) if zone is not None else None
        self.zone_auto = auto

    def observe_for_zone(self, person, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = person.box
        self._zone_boxes.append((x1 / w, y1 / h, x2 / w, y2 / h))

    def finalize_zone(self):
        """learn bed area from where they lay during baseline"""
        if self.bed_zone is not None or len(self._zone_boxes) < 10:
            return
        x1, y1, x2, y2 = np.median(np.array(self._zone_boxes), axis=0)
        bw, bh = x2 - x1, y2 - y1
        zone = (x1 - 0.2 * bw, y1 - 0.6 * bh, x2 + 0.2 * bw, y2 + 0.15 * bh)
        self.set_zone(tuple(float(np.clip(v, 0, 1)) for v in zone), auto=True)

    def in_zone(self, pt, shape):
        if self.bed_zone is None:
            return True
        h, w = shape[:2]
        x, y = pt[0] / w, pt[1] / h
        x1, y1, x2, y2 = self.bed_zone
        return x1 <= x <= x2 and y1 <= y <= y2

    # ---- classification ----
    def classify(self, person, shape):
        thr = self.cfg.kp_conf
        # if the body is cut off / too close yolo makes up shoulders+hips at the frame edge.
        # someone standing right at the camera came out as lying. but don't reject everything at
        # the bottom edge, people on the floor after a fall end up there (lost 30 frames on UR Fall)
        if torso_view(person, shape, thr) in ("torso_hidden", "too_close"):
            return "unknown"
        tr = torso(person, thr)
        if tr is None:
            return "unknown"
        sh, hip, length = tr
        h = shape[0]
        if min(person.kps[L_HIP][1], person.kps[R_HIP][1]) > 0.97 * h or sh[1] < 0.02 * h:
            return "unknown"
        ang = angle_from_vertical(hip - sh)
        in_zone = self.in_zone((sh + hip) / 2, shape)
        if ang > 55:
            return "lying" if in_zone else "floor"
        if ang >= 35:
            return "reclined" if in_zone else "sitting"
        # wider than tall is never standing (adl-37 curled up towards the camera read as standing)
        x1, y1, x2, y2 = person.box
        if x2 - x1 >= 1.1 * (y2 - y1):
            return "lying" if in_zone else "sitting"
        knee = mid(person, L_KN, R_KN, thr)
        if knee is not None:
            thigh = knee - hip
            if angle_from_vertical(thigh) > 50:
                return "sitting"
            return "standing" if np.linalg.norm(thigh) > 0.5 * length else "sitting"
        # no legs visible -> can't tell sit from stand
        return "unknown"

    # ---- state machine ----
    def update(self, t, person, shape):
        cfg = self.cfg
        if person is None:
            if self.state != "floor" and self.fastfall.vanished_mid_fall(t):
                self.cand, self.cand_since = "floor", t
                self._set("floor", t)
                return ["fall"]
            # lost while lying in bed (blanket etc) isn't leaving, leaving goes through sitting
            self.hidden = self.known in IN_BED and self.state in IN_BED + ("unknown",)
            limit = cfg.hidden_in_bed_s if self.hidden else cfg.absent_s
            if (self.last_seen is not None and t - self.last_seen > limit
                    and self.state not in ("absent", "floor")):
                prev = self.known
                self._set("absent", t)
                self.hidden = False
                if prev in IN_BED + ("sitting",):
                    return ["absent"]
            return self._floor_long(t)

        self.last_seen = t
        self.hidden = False
        events = self._floor_long(t)
        self._track_hip(t, person)
        if self.state != "floor" and self._fast_fall(t, person, shape):
            self.cand, self.cand_since = "floor", t
            self._set("floor", t)
            return ["fall"]
        if self.fastfall.near_fall and self.state == "standing":
            events.append("near_fall")
        raw = self.classify(person, shape)
        if self.state == "floor" and raw in IN_BED:
            raw = "floor"
        if raw == "unknown":
            # unclear for a while -> say unclear instead of freezing the last guess. floor stays floor
            self.unclear_since = self.unclear_since or t
            if self.state not in ("unknown", "floor") and t - self.unclear_since >= cfg.unclear_s:
                self._set("unknown", t)
            self.cand = None
        else:
            self.unclear_since = None
            if raw != self.cand:
                self.cand, self.cand_since = raw, t
            need = cfg.floor_min_s if raw == "floor" else cfg.state_min_s
            if raw != self.state and t - self.cand_since >= need:
                prev = self.known
                self._set(raw, t)
                if prev != raw:
                    events += self._transition(prev, raw)
                self._sts_pending = raw == "standing" and prev == "sitting"
        if self._sts_pending and self.state == "standing" and t - self.since >= 2.0:
            self._sts_pending = False
            self.sit_to_stand_s = self._rise_time() or self.sit_to_stand_s
        events += self._help(t, person)
        return events

    def _fast_fall(self, t, person, shape):
        in_zone = None
        if self.bed_zone is not None:
            in_zone = lambda pt: self.in_zone(pt, shape)   # noqa: E731
        return self.fastfall.update(t, person, self.state, self.since, in_zone)

    def _track_hip(self, t, person):
        tr = torso(person, self.cfg.kp_conf)
        if tr is not None:
            self.hip_hist.append((t, tr[1][1], tr[2]))
        while self.hip_hist and self.hip_hist[0][0] < t - 10.0:
            self.hip_hist.popleft()

    def _rise_time(self):
        """sit-to-stand time, hips 10%->90%"""
        if len(self.hip_hist) < 8:
            return None
        t = np.array([h[0] for h in self.hip_hist])
        y = np.array([h[1] for h in self.hip_hist])
        lowest = int(np.argmax(y))
        if lowest >= len(y) - 2:
            return None
        y0, y1 = y[lowest], y[-1]
        rise = y0 - y1
        if rise < 0.2 * np.median([h[2] for h in self.hip_hist]):
            return None
        after_t, after_y = t[lowest:], y[lowest:]
        t10 = after_t[np.argmax(after_y <= y0 - 0.1 * rise)]
        t90 = after_t[np.argmax(after_y <= y0 - 0.9 * rise)]
        return round(float(t90 - t10), 1) if t90 > t10 else None

    def _set(self, state, t):
        prev, self.state, self.since = self.state, state, t
        if state != "unknown":
            self.known = state
        return prev

    @staticmethod
    def _transition(prev, new):
        if new == "floor":
            return ["fall"]
        if new == "sitting" and prev in IN_BED:
            return ["bed_edge"]
        if new == "standing" and prev in IN_BED + ("sitting",):
            return ["bed_exit"]
        if new in IN_BED and prev not in IN_BED + ("unknown",):
            return ["back_in_bed"]
        return []

    def _help(self, t, person):
        thr = self.cfg.kp_conf
        tr = torso(person, thr)
        raised = False
        # arms on the floor look raised, ignore
        if tr is not None and self.state != "floor":
            sh, _, length = tr
            for w in (kp(person, L_WR, thr), kp(person, R_WR, thr)):
                if w is not None and w[1] < sh[1] - 0.55 * length:
                    raised = True
        if not raised:
            self.help_since, self.help_active = None, False
            return []
        if self.help_since is None:
            self.help_since = t
        if not self.help_active and t - self.help_since >= self.cfg.help_hold_s:
            self.help_active = True
            return ["help"]
        return []
