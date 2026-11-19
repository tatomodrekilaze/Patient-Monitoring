"""Sleep / wake from stillness and the eyes; nurse-visit detection."""
from collections import deque


class SleepTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = None
        self.still_since = None
        self.moving_since = None
        self.asleep_total = 0.0
        self._last_t = None

    def update(self, t, in_bed, moving, eyes_open=None):
        """eyes_open: the face model's reading when the face is seen (None = not seen): lying still
        with the eyes open is resting, not sleep."""
        if self._last_t is not None and self.state == "asleep":
            self.asleep_total += t - self._last_t
        self._last_t = t
        if not in_bed:
            self.state, self.still_since, self.moving_since = "awake", None, None
            return
        if moving:
            self.still_since = None
            self.moving_since = self.moving_since or t
            if self.state == "asleep" and t - self.moving_since >= 3:
                self.state = "awake"
        else:
            self.moving_since = None
            self.still_since = self.still_since or t
            if eyes_open:
                self.state = "awake"
            elif t - self.still_since >= self.cfg.sleep_onset_s:
                self.state = "asleep"
            elif self.state is None:
                self.state = "awake"


class VisitDetector:
    """A second person in view for a few seconds counts as a nurse visit."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.since = None
        self.logged = False

    def update(self, t, n_visitors):
        if n_visitors <= 0:
            self.since, self.logged = None, False
            return False
        self.since = self.since or t
        if not self.logged and t - self.since >= self.cfg.visit_min_s:
            self.logged = True
            return True
        return False
