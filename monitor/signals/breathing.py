"""Respiratory rate from chest motion.

chest ROI -> dense optical flow (mean dx, dy) -> integrate to displacement
-> resample 10 Hz -> detrend + bandpass -> PCA to one axis -> peak counting.
"""
from collections import deque

import cv2
import numpy as np
from scipy.signal import detrend, find_peaks

from .filters import bandpass, resample_uniform, spectral_peak


class BreathingEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.roi_px = cfg.flow_roi_px     # the bed raises it when the chest comes from a full-resolution picture
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.t = deque()
        self.disp = deque()
        self.moving = deque()
        self.pos = np.zeros(2)
        self.ref_amp = None          # typical breath prominence for this patient
        self.last_breath_t = None
        self.rr = None
        self.quality = 0.0
        self.motion = False
        self.wave_t = np.array([])
        self.wave = np.array([])
        self.peak_times = []
        self.recent_amp = None
        self.breaths = deque(maxlen=400)   # de-duplicated breath times, for pattern analysis
        self.breath_amps = deque(maxlen=400)   # ... and each breath's depth (same order)

    def pause(self):
        """The chest went out of view: drop the signal, keep what was learned about the patient.
        The breath timer is cleared too - time spent unseen must never count as a breathing pause."""
        self.prev_gray = None
        self.t.clear()
        self.disp.clear()
        self.moving.clear()
        self.pos = np.zeros(2)
        self.last_breath_t = None
        self.rr = None
        self.quality = 0.0
        self.recent_amp = None

    # ---- per frame ----
    def update(self, t, gray, box, moving, unit=1.0):
        """unit: pixels of `gray` per working-frame pixel (k for a full-resolution picture), so the
        displacement keeps one unit when night mode switches between full and working resolution -
        the learned breath size (apnea reference) must not jump."""
        if box is None:
            self.prev_gray = None
            return
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            dx, dy = self._flow(self.prev_gray, gray, box)
            dx, dy = dx / unit, dy / unit
            self.pos += (dx, dy)
            self.t.append(t)
            self.disp.append(self.pos.copy())
            self.moving.append(bool(moving))
            cutoff = t - self.cfg.breath_window_s - 5
            while self.t and self.t[0] < cutoff:
                self.t.popleft()
                self.disp.popleft()
                self.moving.popleft()
        self.prev_gray = gray

    def _flow(self, prev, cur, box):
        x1, y1, x2, y2 = box
        p, c = prev[y1:y2, x1:x2], cur[y1:y2, x1:x2]
        scale = self.roi_px / max(p.shape)
        size = (max(8, int(p.shape[1] * scale)), max(8, int(p.shape[0] * scale)))
        p = cv2.resize(p, size, interpolation=cv2.INTER_AREA)
        c = cv2.resize(c, size, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 2, 9, 3, 5, 1.1, 0)
        return float(flow[..., 0].mean()) / scale, float(flow[..., 1].mean()) / scale

    # ---- periodic estimate ----
    def estimate(self, now):
        cfg = self.cfg
        if len(self.t) < 10 or self.t[-1] - self.t[0] < cfg.rr_min_data_s:
            return
        fs = cfg.resample_hz
        lo, hi = cfg.breath_band
        t = np.array(self.t)
        tu, xy = resample_uniform(t, np.array(self.disp), fs, cfg.breath_window_s)
        if tu is None or len(tu) < fs * cfg.rr_min_data_s:
            return

        recent = t >= now - cfg.rr_window_s
        self.motion = bool(np.any(np.array(self.moving)[recent]))

        xy = detrend(xy, axis=0)
        try:
            f = bandpass(xy, fs, lo, hi)
        except ValueError:
            return
        # one axis: direction in which the chest actually moves
        _, vecs = np.linalg.eigh(np.cov(f.T))
        s = f @ vecs[:, -1]
        self.wave_t, self.wave = tu, s

        std = float(np.std(s))
        thr = 0.8 * std
        if self.ref_amp is not None:
            thr = max(thr, 0.25 * self.ref_amp)
        thr = max(thr, 1e-4)
        dist = max(1, int(fs * 60.0 / cfg.max_rr))
        peaks, props = find_peaks(s, distance=dist, prominence=thr)
        # ignore filter edges
        keep = (tu[peaks] >= tu[0] + 1.0) & (tu[peaks] <= tu[-1] - 0.8)
        peaks, prom = peaks[keep], props["prominences"][keep]
        pt = tu[peaks]
        self.peak_times = pt.tolist()
        if len(pt):
            if self.last_breath_t is None or pt[-1] > self.last_breath_t:
                self.last_breath_t = float(pt[-1])
        # the same breath is re-found on every window with slight jitter: keep it once
        min_gap = 0.8 * 60.0 / cfg.max_rr
        for p, a in zip(pt, prom):
            if not self.breaths or p > self.breaths[-1] + min_gap:
                self.breaths.append(float(p))
                self.breath_amps.append(float(a))      # depth of each breath: patterns, hypopneas

        last5 = s[tu >= tu[-1] - 6.0]
        self.recent_amp = float(last5.max() - last5.min()) if len(last5) else None

        if self.motion:
            self.quality = 0.0          # hold the last value while the patient moves
            return

        in_win = pt >= now - cfg.rr_window_s
        fp, q = spectral_peak(s[tu >= now - cfg.rr_window_s - 5], fs, lo, hi)
        rr = None
        if in_win.sum() >= 3:
            rr = 60.0 / float(np.median(np.diff(pt[in_win])))
            if fp is not None and abs(rr - fp * 60) > 3:
                q *= 0.5
            if q > 0.25:
                amp = float(np.median(prom[in_win]))
                self.ref_amp = amp if self.ref_amp is None else 0.8 * self.ref_amp + 0.2 * amp
        elif fp is not None and q > 0.5:
            rr = fp * 60.0               # very slow breathing: too few peaks, trust the spectrum
        if rr is not None:
            self.rr = float(np.clip(rr, 0, cfg.max_rr))
            self.quality = float(q)
        else:
            self.quality = 0.0

    def apnea_seconds(self, now):
        """Seconds without a breath, only when we are confident it is real (else 0)."""
        if self.last_breath_t is None or self.ref_amp is None or self.motion:
            return 0.0
        gap = now - self.last_breath_t
        if self.recent_amp is not None and self.recent_amp > 0.5 * self.ref_amp:
            return 0.0   # chest still moving like breathing; the peak detector just missed it
        return max(0.0, gap)

    def irregularity(self, now, window=60.0):
        """Coefficient of variation of breath-to-breath intervals (0 = metronome, >0.4 = irregular)."""
        b = [x for x in self.breaths if x >= now - window]
        if len(b) < 8:
            return None
        iv = np.diff(b)
        return float(np.std(iv) / np.mean(iv))

    def waveform(self, seconds=20.0):
        if not len(self.wave):
            return []
        m = self.wave_t >= self.wave_t[-1] - seconds
        s = self.wave[m]
        scale = self.ref_amp / 2 if self.ref_amp else (np.abs(s).max() or 1.0)
        return np.round(np.clip(s / scale, -2, 2), 3).tolist()
