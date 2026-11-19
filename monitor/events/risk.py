"""Personal baseline, NEWS2-style risk score, trend arrow, colour level with hysteresis."""
from collections import deque

import numpy as np

LEVELS = ["green", "yellow", "red"]


class Baseline:
    """Learns this patient's normal breathing (and heart) rate while they rest."""

    def __init__(self, duration_s, min_samples=8):
        self.duration_s = duration_s
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self.start = None
        self.rr, self.hr = [], []
        self.ready = False
        self.rr_med = self.rr_sd = self.hr_med = self.hr_sd = None

    def add(self, t, rr, rr_ok, hr=None, hr_ok=False):
        if self.ready:
            return
        if self.start is None:
            self.start = t
        if rr is not None and rr_ok:
            self.rr.append(rr)
        if hr is not None and hr_ok:
            self.hr.append(hr)
        if t - self.start >= self.duration_s and len(self.rr) >= self.min_samples:
            self.rr_med, self.rr_sd = _robust(self.rr, floor=1.5)
            if len(self.hr) >= self.min_samples:
                self.hr_med, self.hr_sd = _robust(self.hr, floor=4.0)
            self.ready = True

    def progress(self, t):
        if self.ready:
            return 1.0
        if self.start is None:
            return 0.0
        return float(min(0.99, (t - self.start) / self.duration_s))

    def as_dict(self, t):
        return {
            "ready": self.ready,
            "progress": round(self.progress(t), 2),
            "rr": _r(self.rr_med), "rr_sd": _r(self.rr_sd),
            "hr": _r(self.hr_med),
        }


def _robust(values, floor):
    v = np.array(values, float)
    med = float(np.median(v))
    sd = float(1.4826 * np.median(np.abs(v - med)))
    return med, max(sd, floor)


def _r(v, n=1):
    return None if v is None else round(float(v), n)


def news_rr(rr):
    if rr is None:
        return 0
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def news_hr(hr):
    if hr is None:
        return 0
    if hr <= 40:
        return 3
    if hr <= 50:
        return 1
    if hr <= 90:
        return 0
    if hr <= 110:
        return 1
    if hr <= 130:
        return 2
    return 3


def score_parts(rr, hr, baseline, trend, use_hr):
    """Points per reason. Kept separate so the dashboard can explain *why* a bed is red."""
    parts = {"rr": news_rr(rr)}
    if use_hr and hr is not None:
        parts["hr"] = news_hr(hr)
    if baseline.ready and rr is not None:
        z = (rr - baseline.rr_med) / baseline.rr_sd
        parts["deviation"] = 2 if abs(z) >= 3 else 1 if abs(z) >= 2 else 0
        parts["trend"] = 1 if trend in ("up2", "down2") and abs(z) >= 1 else 0
    return parts


def level_for(parts):
    score = sum(parts.values())
    if score >= 5:
        return "red"
    if score >= 3 or max(parts.values()) >= 3:
        return "yellow"
    return "green"


class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.history = deque()          # (t, rr)
        self.level = "green"
        self._pending = None
        self._pending_since = None
        self.score = 0
        self.trend = None
        self.slope = 0.0
        self.parts = {}
        self.forecast = None

    def update(self, t, rr, rr_ok, hr, hr_ok, baseline):
        cfg = self.cfg
        if rr is not None and rr_ok:
            self.history.append((t, rr))
        while self.history and self.history[0][0] < t - max(cfg.trend_window_s, 600):
            self.history.popleft()
        self.trend, self.slope = self._trend(t)

        use_hr = cfg.hr_in_risk and hr_ok
        self.parts = score_parts(rr, hr, baseline, self.trend, use_hr)
        self.score = int(sum(self.parts.values()))
        self._apply(t, level_for(self.parts))
        self.forecast = self._forecast(rr, hr if use_hr else None, baseline)
        return self.level

    def _forecast(self, rr, hr, baseline):
        """If the trend continues, when does this patient reach red? Straight-line extrapolation."""
        if rr is None or self.trend in (None, "flat") or self.level == "red":
            return None
        step = 0.5 if self.slope > 0 else -0.5
        r = rr
        while 4 <= r <= 60:
            r += step
            if level_for(score_parts(r, hr, baseline, self.trend, hr is not None)) == "red":
                eta_s = (r - rr) / self.slope * 60
                if 0 < eta_s <= self.cfg.forecast_horizon_s:
                    return {"level": "red", "eta_s": round(eta_s), "rr_at": round(r, 1)}
                return None
        return None

    def explain(self):
        return [{"key": k, "points": v} for k, v in sorted(self.parts.items(), key=lambda kv: -kv[1]) if v > 0]

    def _trend(self, t):
        pts = [(ti, r) for ti, r in self.history if ti >= t - self.cfg.trend_window_s]
        if len(pts) < 5 or pts[-1][0] - pts[0][0] < self.cfg.trend_window_s * 0.4:
            return None, 0.0
        x = np.array([p[0] for p in pts]) / 60.0
        y = np.array([p[1] for p in pts])
        slope = float(np.polyfit(x - x[0], y, 1)[0])    # breaths/min per minute
        c = self.cfg
        if slope >= c.trend_steep_thr:
            return "up2", slope
        if slope >= c.trend_thr:
            return "up", slope
        if slope <= -c.trend_steep_thr:
            return "down2", slope
        if slope <= -c.trend_thr:
            return "down", slope
        return "flat", slope

    def _apply(self, t, target):
        """Escalate fast, calm down slowly, so the colour does not flicker."""
        if target == self.level:
            self._pending = None
            return
        if target != self._pending:
            self._pending, self._pending_since = target, t
        up = LEVELS.index(target) > LEVELS.index(self.level)
        wait = self.cfg.escalate_s if up else self.cfg.deescalate_s
        if t - self._pending_since >= wait:
            self.level, self._pending = target, None

    def history_list(self, seconds=600, step=5.0):
        """Downsampled RR history for sparklines: [[t, rr], ...]."""
        out, last = [], -1e9
        for t, rr in self.history:
            if t >= self.history[-1][0] - seconds and t - last >= step:
                out.append([round(t, 1), round(rr, 1)])
                last = t
        return out
