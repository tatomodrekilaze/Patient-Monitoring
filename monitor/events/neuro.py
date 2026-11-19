"""Guided FAST face test (the "F" of FAST): the bedside speaker asks for a smile, then tightly
closed eyes; uneven eye closure and a mouth corner that lags in the smile are measured.
palsynet (27 palsy, 22 healthy face videos): 18 / 23 palsy flagged, 0 / 22 healthy.
A screening prompt for a clinician, not a diagnosis."""
import numpy as np
from ..pose import L_SH, L_WR, R_SH, R_WR, kp, torso
from .face import droop


class FaceTest:
    """Guided FAST face test: the speaker asks "smile, show your teeth" (6 s), then "close your eyes
    tightly" (6 s). A weak side of the face lifts one mouth corner less and closes one eye less -
    what a doctor checks. Measured on palsynet (27 Bell's palsy, 22 healthy face videos): uneven eye
    closure separates them best (AUC 0.94), then the mouth-corner droop during a smile (0.86); the face
    at rest only 0.84 and catches half. Only frames facing the camera count."""
    STEP_S = 6.0

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "idle"              # idle -> smile -> eyes -> idle
        self.until = None
        self.smile, self.eyes = [], []
        self.result = None

    def start(self, now):
        self.state, self.until, self.smile, self.eyes = "smile", now + self.STEP_S, [], []

    def update(self, now, reading, yaw):
        """reading: the face AI's latest (t, landmarks, blendshapes, trusted) or None; yaw: head turn.
        Returns 'next' when it is time to ask for the eyes, the result dict when done, else None."""
        if self.state == "idle":
            return None
        if reading is not None and reading[0] >= self.until - self.STEP_S and reading[3]                 and abs(yaw) <= self.cfg.face_max_yaw:
            lm, bs = reading[1], reading[2]
            if self.state == "smile":
                if bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0) > 0.6:
                    self.smile.append(droop(lm))       # + = the patient's right mouth corner lower
            else:
                l, r = bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)
                if max(l, r) > 0.5:
                    self.eyes.append((abs(l - r), eye_open(lm, "right") - eye_open(lm, "left")))
        if now < self.until:
            return None
        if self.state == "smile":
            self.state, self.until = "eyes", now + self.STEP_S
            return "next"
        self.state = "idle"
        s = float(np.median(self.smile)) if len(self.smile) >= 3 else None
        e = float(np.median([a for a, _ in self.eyes])) if len(self.eyes) >= 3 else None
        weak_smile = s is not None and abs(s) >= self.cfg.smile_asym_thr
        weak_eye = e is not None and e >= self.cfg.eye_asym_thr
        side = None
        if weak_eye:                          # the weak eye stays more open
            side = "right" if np.median([d for _, d in self.eyes]) > 0 else "left"
        elif weak_smile:                      # the weak corner is not lifted
            side = "right" if s > 0 else "left"
        self.result = {"smile_asym": None if s is None else round(abs(s), 3),
                       "eye_asym": None if e is None else round(e, 2),
                       "asymmetric": weak_smile or weak_eye, "side": side,
                       "measured": s is not None or e is not None}
        return self.result


# face mesh eyelids (the patient's own right / left): upper, lower, outer corner, inner corner
EYELIDS = {"right": (159, 145, 33, 133), "left": (386, 374, 263, 362)}


def eye_open(lm, side):
    up, lo, a, b = EYELIDS[side]
    return float(np.linalg.norm(lm[up] - lm[lo]) / (np.linalg.norm(lm[a] - lm[b]) + 1e-9))
