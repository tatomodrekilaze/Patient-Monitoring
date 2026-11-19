"""Pressure-injury (bedsore) prevention: how long has the patient lain in the same position?

Only time in bed counts: standing, sitting in a chair or being out of view adds nothing (it did:
"34 min in one position" was shown for a person standing in front of the camera).

A position is a "signature" of the skeleton layout relative to the torso plus which face points
are visible (turning to the side hides one eye/ear). A large, lasting change = repositioned.
"""
import numpy as np

from ..pose import L_AN, L_EAR, L_EYE, L_HIP, L_KN, L_SH, NOSE, R_AN, R_EAR, R_EYE, R_HIP, R_KN, R_SH, torso

LAYOUT = (NOSE, L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN)
FACE = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


class RepositionTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self, t=None):
        self.ref = None
        self.last_reposition = t
        self.cand_since = None
        self.in_bed_s = 0.0             # time lain in the current position
        self._t = None
        self.due_raised = False
        self.count = 0

    def signature(self, person, thr):
        tr = torso(person, thr)
        if tr is None:
            return None
        sh, hip, length = tr
        c = (sh + hip) / 2
        layout = np.full((len(LAYOUT), 2), np.nan)
        for i, k in enumerate(LAYOUT):
            if person.conf[k] >= thr:
                layout[i] = (person.kps[k] - c) / length
        face = np.array([person.conf[k] >= thr for k in FACE], float)
        return layout, face

    def distance(self, a, b):
        la, fa = a
        lb, fb = b
        both = ~np.isnan(la[:, 0]) & ~np.isnan(lb[:, 0])
        d = float(np.mean(np.linalg.norm(la[both] - lb[both], axis=1))) if both.any() else 0.0
        return d + 0.15 * float(np.abs(fa - fb).sum())

    def update(self, t, person, in_bed, out_of_bed):
        """Returns a list of events: 'repositioned', 'turn_due'."""
        if self.last_reposition is None:
            self.last_reposition = t
        if out_of_bed:                       # standing / walking takes the pressure off
            self.reset(t)
            return []
        dt = 0.0 if self._t is None else min(1.0, t - self._t)
        self._t = t
        if not in_bed or person is None:
            return []
        self.in_bed_s += dt
        sig = self.signature(person, self.cfg.kp_conf)
        if sig is None:
            return []
        events = []
        if self.ref is None:
            self.ref = sig
        elif self.distance(sig, self.ref) > self.cfg.reposition_thr:
            self.cand_since = self.cand_since or t
            if t - self.cand_since >= 5.0:  # must stay in the new position, not just a wriggle
                self.ref, self.cand_since = sig, None
                self.last_reposition, self.due_raised = t, False
                self.in_bed_s = 0.0
                self.count += 1
                events.append("repositioned")
        else:
            self.cand_since = None
        if not self.due_raised and self.in_bed_s >= self.cfg.turn_interval_s:
            self.due_raised = True
            events.append("turn_due")
        return events

    def minutes_in_position(self, t):
        return self.in_bed_s / 60.0
