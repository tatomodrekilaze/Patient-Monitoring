"""Black-box replay: the skeletons of the last 30 s before an alert (and 5 s after), so a nurse or
doctor can WATCH what happened - how the patient fell, whether the head hit the floor, what came
before an apnea. Only keypoint coordinates are kept, never an image (privacy by design); a replay
is a few tens of kilobytes.
"""
import json
import os
from collections import deque

import numpy as np

BEFORE_S, AFTER_S, RATE_HZ = 25.0, 5.0, 10.0
# routine reminders have nothing to show
NO_REPLAY = {"rounding", "turn_due", "camera_offline", "delirium_check", "patient_button", "req_water", "req_toilet",
             "req_pain", "req_nurse"}


def _person(p, is_patient):
    return {"id": int(p.track_id), "patient": is_patient,
            "box": [round(float(v), 1) for v in p.box],
            "kps": [[round(float(x), 1), round(float(y), 1)] for x, y in p.kps],
            "conf": [round(float(c), 2) for c in p.conf]}


class Recorder:
    def __init__(self, data_dir, keep=40):
        self.dir = os.path.join(data_dir, "replays")
        self.buf = deque()
        self.pending = []                 # (alert_id, kind, message, t_event)
        self.replays = {}                 # alert_id -> replay dict (most recent `keep`)
        self.keep = keep
        self._last = -1e9

    def add(self, t, shape, patient, others, state, level, zone=None):
        """Every frame; keeps ~10 snapshots a second. Returns the alert ids whose replay was finished."""
        if t - self._last >= 1.0 / RATE_HZ:
            self._last = t
            people = ([_person(patient, True)] if patient is not None else []) + [_person(p, False) for p in others]
            self.buf.append({"t": round(t, 2), "people": people, "state": state, "level": level})
            while self.buf and self.buf[0]["t"] < t - BEFORE_S - AFTER_S - 2:
                self.buf.popleft()
            self.shape = [int(shape[1]), int(shape[0])]
            self.zone = list(zone) if zone else None
        done = []
        for item in list(self.pending):
            alert_id, kind, message, t0, extra = item
            if t >= t0 + AFTER_S:
                self.pending.remove(item)
                frames = [f for f in self.buf if t0 - BEFORE_S <= f["t"] <= t0 + AFTER_S]
                r = {"alert_id": alert_id, "kind": kind, "message": message, "t_event": round(t0, 2),
                     "size": self.shape, "zone": self.zone, "frames": frames, **(extra or {})}
                self._store(alert_id, r)
                done.append(alert_id)
        return done

    def request(self, alert_id, kind, message, t, extra=None):
        if kind in NO_REPLAY or any(p[0] == alert_id for p in self.pending) or alert_id in self.replays:
            return False
        self.pending.append((alert_id, kind, message, t, extra))
        return True

    def _store(self, alert_id, r):
        self.replays[alert_id] = r
        for old in sorted(self.replays)[:-self.keep]:
            self.replays.pop(old, None)
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, f"{alert_id}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, separators=(",", ":"))
        except OSError:
            pass

    def get(self, alert_id):
        if alert_id in self.replays:
            return self.replays[alert_id]
        try:
            with open(os.path.join(self.dir, f"{int(alert_id)}.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None


def impact_speed(head_track, ref_h, height_m):
    """Fastest downward head speed in the last 1.5 s (m/s), from (t, head_y px) samples.
    Body height in pixels ~ the patient's height in metres gives the scale; perspective makes this an
    estimate. Falls from standing hit ~3-5 m/s; rolling out of bed ~2 m/s."""
    if len(head_track) < 4 or not ref_h:
        return None
    t = np.array([a for a, _ in head_track], float)
    y = np.array([b for _, b in head_track], float)
    m = t >= t[-1] - 1.5
    t, y = t[m], y[m]
    if len(t) < 4:
        return None
    best = 0.0
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + 0.2)          # speed over 0.2 s steps: robust to keypoint jitter
        if j < len(t) and t[j] > t[i]:
            best = max(best, (y[j] - y[i]) / (t[j] - t[i]))
    return round(float(best / ref_h * height_m), 1)
