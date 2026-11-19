"""30-second chair stand (CDC STEADI): stands counted from the hips rising above the patient's
own seated level. HMDB51 sit-down clips: 0 / 32 false stands."""
from collections import deque
import numpy as np
from ..pose import L_HIP, R_HIP, mid


class ChairStandTest:
    """30-second chair stand (CDC STEADI fall-risk screen): arms crossed, stand up fully and sit down
    as many times as possible in 30 s. Below average for 65+ is < 12 (men) / < 11 (women) at 65-69,
    falling with age to < 8 at 85-89.

    A stand = the thigh turning from the seated level to vertical (hip above knee, in torso lengths,
    relative to this person's own seated level - cameras and bodies differ) AND the hips really rising
    in the picture, held for a moment. Tested on HMDB51 stand / sit clips: a fixed 'standing' level
    missed people whose standing value is only ~0.6, and single bad pose frames made fake stands."""
    RISE = 0.25            # thigh: this much above the seated level = standing
    BACK = 0.12            # back within this of the seated level = seated again
    HIP_UP = 0.2           # hips at least this many torso lengths higher than when seated
    HOLD_S = 0.2
    SEATED_MAX = 0.45      # thigh near horizontal (side) or foreshortened (front)

    def __init__(self):
        self.state = "idle"
        self.armed_until = None
        self.start = None
        self.count = 0
        self.result = None
        self.history = []

    def arm(self, now, window_s=60.0):
        self.state, self.armed_until, self.start, self.count = "armed", now + window_s, None, 0
        self.up, self.up_since, self.seat = False, None, None
        self.seated = deque()

    @staticmethod
    def measure(person, thr):
        """(thigh rise, hip height, torso length) or None. thigh rise: hip above knee / torso length."""
        from ..pose import L_HIP, L_KN, R_HIP, R_KN, kp, torso
        tr = torso(person, thr) if person is not None else None
        if tr is None:
            return None
        v = []
        for h, k in ((L_HIP, L_KN), (R_HIP, R_KN)):
            ph, pk = kp(person, h, thr), kp(person, k, thr)
            if ph is not None and pk is not None:
                v.append((pk[1] - ph[1]) / tr[2])
        return (float(np.mean(v)), float(tr[1][1]), tr[2]) if v else None

    @classmethod
    def rise(cls, person, thr):
        m = cls.measure(person, thr)
        return None if m is None else m[0]

    def update(self, now, person, thr):
        """Returns the number of stands once, when the 30 s are over."""
        if self.state == "idle":
            return None
        m = self.measure(person, thr)
        if self.state in ("armed", "seated") and now > self.armed_until:
            self.state = "idle"
            return None
        if m is not None:
            r, hip_y, length = m
            if not self.up and r < self.SEATED_MAX:
                # seated level = the low point of the last 3 s (a running average followed the
                # first half of each rise and made it look too small)
                self.seated.append((now, r, hip_y, length))
                while self.seated[0][0] < now - 3.0:
                    self.seated.popleft()
                a = np.array([x[1:] for x in self.seated])
                self.seat = [float(np.percentile(a[:, 0], 20)), float(np.percentile(a[:, 1], 80)),
                             float(np.median(a[:, 2]))]
                if self.state == "armed":
                    self.state = "seated"                 # ready: the clock starts at the first rise
            if self.seat is not None:
                sr, sy, sl = self.seat
                standing = r >= sr + self.RISE and sy - hip_y >= self.HIP_UP * sl
                if not self.up and standing:
                    self.up_since = self.up_since or now
                    if self.state == "seated":
                        self.state, self.start = "running", now
                    if now - self.up_since >= self.HOLD_S:
                        self.up = True
                        self.count += 1                   # a full stand
                elif not standing:
                    self.up_since = None
                if self.up and r <= sr + self.BACK:
                    self.up = False
        if self.state == "running" and now - self.start >= 30.0:
            self.state, self.result = "idle", self.count
            self.history.append(self.count)
            return self.count
        return None
