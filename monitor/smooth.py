"""Keypoint smoothing with the One-Euro filter (Casiez et al., CHI 2012).

Pose keypoints jitter by a few pixels from frame to frame even when a person lies still. A plain moving
average removes the jitter but makes fast movement (a fall) lag. The One-Euro filter adapts: strong
smoothing while a point is still, almost none while it moves fast.
"""
import math

import numpy as np


class _OneEuro:
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        if self.t is None or t <= self.t:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class KeypointSmoother:
    """One filter per tracked person; keypoints below the confidence threshold are left as they are."""

    def __init__(self, min_cutoff=1.5, beta=0.02, thr=0.4):
        self.min_cutoff, self.beta, self.thr = min_cutoff, beta, thr
        self.filters = {}
        self.seen = {}

    def __call__(self, t, people):
        for p in people:
            f = self.filters.get(p.track_id)
            if f is None:
                f = self.filters[p.track_id] = _OneEuro(self.min_cutoff, self.beta)
            sm = f(t, p.kps.astype(float))
            ok = p.conf >= self.thr
            p.kps = np.where(ok[:, None], sm, p.kps)
            self.seen[p.track_id] = t
        for tid in [k for k, v in self.seen.items() if t - v > 5]:
            self.filters.pop(tid, None)
            self.seen.pop(tid, None)
        return people
