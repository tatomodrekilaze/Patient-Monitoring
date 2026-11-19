"""Simulated extra beds so the ward dashboard shows ranking. Clearly marked as simulated
on screen and never send phone alerts."""
import random
import threading
import time

from . import i18n
from .events.risk import Baseline, RiskEngine

PROFILES = [
    ("stable", 14, 72),
    ("slow_decline", 16, 84),
    ("stable", 12, 64),
    ("restless", 18, 90),
    ("stable", 15, 70),
]


class SimBed:
    def __init__(self, bed_id, name, cfg, profile, seed):
        self.bed_id, self.name, self.cfg = bed_id, name, cfg
        self.kind, self.base_rr, self.base_hr = profile
        self.rng = random.Random(seed)
        self.rr, self.hr = float(self.base_rr), float(self.base_hr)
        self.start = time.time()
        self.baseline = Baseline(0, min_samples=1)
        self.risk = RiskEngine(cfg)
        self.posture = "lying"
        self.last_visit = self.start - self.rng.uniform(5, 90) * 60
        self.snap = {}
        self.running = False

    def start_thread(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            self.step(time.time())
            time.sleep(1.0)

    def step(self, t):
        mins = (t - self.start) / 60
        target = self.base_rr
        if self.kind == "slow_decline":
            target += min(9, mins * 1.2)        # creeps up over ~8 minutes
        self.rr += 0.15 * (target - self.rr) + self.rng.gauss(0, 0.4)
        self.hr += 0.1 * (self.base_hr + (target - self.base_rr) * 2 - self.hr) + self.rng.gauss(0, 1)
        if self.kind == "restless":
            self.posture = "sitting" if self.rng.random() < 0.05 else "lying" if self.rng.random() < 0.2 else self.posture
        self.baseline.add(t, self.rr, True, self.hr, True)
        level = self.risk.update(t, self.rr, True, self.hr, True, self.baseline)
        self.snap = {
            "id": self.bed_id, "name": self.name, "simulated": True,
            "level": level, "level_label": i18n.LEVEL[level],
            "score": self.risk.score, "parts": self.risk.parts,
            "rr": round(self.rr, 1), "rr_valid": True, "rr_quality": 1.0,
            "trend": self.risk.trend, "trend_label": i18n.TREND[self.risk.trend],
            "slope": round(self.risk.slope, 1),
            "hr": round(self.hr), "hr_quality": 1.0,
            "present": True, "moving": False,
            "posture": self.posture, "posture_label": i18n.POSTURE[self.posture],
            "apnea_s": 0, "baseline": self.baseline.as_dict(t),
            "sleep": None, "sleep_label": i18n.SLEEP[None],
            "last_visit": self.last_visit, "bed_exits": 0,
            "rr_history": self.risk.history_list(), "alerts": [],
            "explain": self.risk.explain(), "forecast": self.risk.forecast,
        }

    def mark_visit(self, auto=False):
        self.last_visit = time.time()
