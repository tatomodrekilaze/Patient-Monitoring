"""Heart rate from face colour (rPPG, POS method, Wang et al. 2017). Experimental.

Only measured while the patient is still; any movement restarts the window.
"""
from collections import deque

import cv2
import numpy as np

from .filters import bandpass, resample_uniform, spectral_peak

POS_P = np.array([[0, 1, -1], [-2, 1, 1]], float)


def pos_signal(rgb, fs):
    """rgb: (N, 3) mean skin colour in R, G, B order -> pulse signal (N,)."""
    n = len(rgb)
    win = max(2, int(1.6 * fs))
    h = np.zeros(n)
    for i in range(n - win + 1):
        c = rgb[i:i + win]
        c = c / (c.mean(axis=0) + 1e-9)
        s = c @ POS_P.T
        sd1 = s[:, 1].std()
        p = s[:, 0] + (s[:, 0].std() / sd1 if sd1 > 1e-9 else 0.0) * s[:, 1]
        h[i:i + win] += p - p.mean()
    return h


class HeartRateEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.t = deque()
        self.rgb = deque()
        self.hr = None
        self.quality = 0.0
        self.good_t = None               # last time the pulse peak was clear enough to show
        self.why = "no_face"             # why there is no fresh reading: no_face / moving / measuring

    def update(self, t, frame, box, still):
        if box is None or not still:
            self.why = "no_face" if box is None else "moving"
            self.t.clear()
            self.rgb.clear()
            return
        self.why = "measuring"
        x1, y1, x2, y2 = box
        b, g, r, _ = cv2.mean(frame[y1:y2, x1:x2])
        self.t.append(t)
        self.rgb.append((r, g, b))
        while self.t and self.t[0] < t - self.cfg.hr_window_s - 2:
            self.t.popleft()
            self.rgb.popleft()

    def estimate(self, now):
        if len(self.t) < 30 or self.t[-1] - self.t[0] < self.cfg.hr_window_s * 0.8:
            self.quality = 0.0
            return
        t = np.array(self.t)
        effective_fps = (len(t) - 1) / (t[-1] - t[0])
        if effective_fps < 7:
            self.quality = 0.0           # too slow to see a 3 Hz pulse
            return
        fs = min(20.0, effective_fps)
        lo, hi = self.cfg.hr_band
        tu, rgb = resample_uniform(t, np.array(self.rgb), fs, self.cfg.hr_window_s)
        if tu is None:
            return
        h = bandpass(pos_signal(rgb, fs), fs, lo, min(hi, fs / 2 * 0.9))
        fp, q = spectral_peak(h, fs, lo, hi, halfwidth=0.1)
        if fp is None:
            return
        self.quality = q
        if q > self.cfg.hr_quality:
            hr = fp * 60
            fresh = self.hr is None or self.good_t is None or now - self.good_t > self.cfg.hr_hold_s
            self.hr = hr if fresh else 0.7 * self.hr + 0.3 * hr
            self.good_t = now

    def shown(self, now):
        """Reading for the screen: the last clear one, kept through a short weak patch (quality on
        the Conico's compressed stream hovers around the cutoff, so the number flickered on and off).
        Returns (hr or None, status): ok / hold / no_face / moving / measuring."""
        if self.hr is None or self.good_t is None or now - self.good_t > self.cfg.hr_hold_s:
            return None, self.why
        return self.hr, "ok" if now - self.good_t <= 1.0 else "hold"
