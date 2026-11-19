"""Eulerian video magnification (Wu et al. 2012), linear version. display only.

breath: 0.12-0.8 Hz, alpha 18 on the body box (~x13 motion on synthetic tests)
pulse:  0.8-2.5 Hz, alpha 80 on a blurred face (~x36 colour)

band = fast lowpass - slow lowpass of a downscaled frame, out = frame + alpha*band
"""
import math

import cv2
import numpy as np

MODES = {
    # name: (low Hz, high Hz, alpha, spatial downsample)
    "breath": (0.12, 0.8, 18.0, 4),
    "pulse": (0.8, 2.5, 80.0, 16),
}
MAX_BOOST = 70.0        # clip the added signal (grey levels): a sudden movement must not blow the picture out


class Magnifier:
    def __init__(self):
        self.mode = None
        self.reset()

    def reset(self):
        self.lp_hi = self.lp_lo = None
        self.last_t = None
        self.gain = 0.0             # fades in after a reset / movement, so the filter has settled

    def set_mode(self, mode):
        mode = mode if mode in MODES else None
        if mode != self.mode:
            self.mode = mode
            self.reset()

    def apply(self, t, frame, box=None, still=True):
        """frame: BGR uint8. box: (x1, y1, x2, y2) region to magnify (None = whole frame).
        Must be called on every processed frame while a mode is on (the filters run in time)."""
        if self.mode is None:
            return frame
        lo, hi, alpha, down = MODES[self.mode]
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(4, w // down), max(4, h // down)), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.lp_hi is None or self.lp_hi.shape != small.shape:
            self.lp_hi, self.lp_lo, self.last_t = small.copy(), small.copy(), t
            return frame
        dt = min(0.5, max(1 / 60, t - self.last_t))
        self.last_t = t
        self.lp_hi += (1 - math.exp(-2 * math.pi * hi * dt)) * (small - self.lp_hi)
        self.lp_lo += (1 - math.exp(-2 * math.pi * lo * dt)) * (small - self.lp_lo)
        # fade in over ~3 s, and out while the body moves (a movement is not a breath or a pulse)
        target = 1.0 if still else 0.0
        self.gain += (target - self.gain) * min(1.0, dt / (3.0 if still else 0.4))
        if self.gain < 0.02:
            return frame
        band = np.clip((self.lp_hi - self.lp_lo) * (alpha * self.gain), -MAX_BOOST, MAX_BOOST)
        sh, sw = band.shape[:2]
        if box is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            # everything at the small size (cheap), then only the region around the box is touched
            band *= _soft_mask(box, w, h, sw, sh)[..., None]
            gx, gy = (box[2] - box[0]) * 0.3, (box[3] - box[1]) * 0.3
            x1, y1 = int(max(0, box[0] - gx)), int(max(0, box[1] - gy))
            x2, y2 = int(min(w, box[2] + gx)), int(min(h, box[3] + gy))
            if x2 - x1 < 2 or y2 - y1 < 2:
                return frame
        boost = cv2.resize(band, (w, h), interpolation=cv2.INTER_LINEAR)[y1:y2, x1:x2]
        out = frame.copy()
        out[y1:y2, x1:x2] = np.clip(frame[y1:y2, x1:x2].astype(np.float32) + boost, 0, 255).astype(np.uint8)
        return out


def _soft_mask(box, w, h, sw, sh, grow=0.12):
    """At the small size: 1 inside the (slightly grown) box, fading to 0 outside - no hard seam."""
    kx, ky = sw / w, sh / h
    x1, y1, x2, y2 = box
    gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
    m = np.zeros((sh, sw), np.float32)
    cv2.rectangle(m, (int((x1 - gx) * kx), int((y1 - gy) * ky)), (int((x2 + gx) * kx), int((y2 + gy) * ky)), 1.0, -1)
    k = int(max(3, min(sw, sh) * 0.08)) | 1
    return cv2.GaussianBlur(m, (k, k), 0)
