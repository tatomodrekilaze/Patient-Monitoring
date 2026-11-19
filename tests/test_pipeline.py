"""Run:  python -m unittest discover tests -v

No camera or model needed: breathing is tested on synthetic video (a textured chest that moves
by ~1 px), posture on hand-made skeletons.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.alerts import AlertManager, EventLog
from monitor.config import make_config
from monitor.events.posture import PostureTracker
from monitor.events.risk import Baseline, RiskEngine, news_rr
from monitor.pose import Person
from monitor.signals.breathing import BreathingEstimator
from monitor.signals.rppg import HeartRateEstimator

BOX = (100, 80, 220, 180)


def synthetic_breathing(est, rr_fn, seconds, fps=15, amp=1.2, noise=2.0, t0=0.0, seed=0):
    """Feed frames whose chest texture moves vertically like breathing. rr_fn(t) -> breaths/min."""
    rng = np.random.default_rng(seed)
    tex = (rng.random((400, 400)) * 255).astype(np.float32)
    import cv2
    tex = cv2.GaussianBlur(tex, (0, 0), 2.5)
    phase, t = 0.0, t0
    dt = 1.0 / fps
    while t < t0 + seconds:
        phase += 2 * np.pi * rr_fn(t) / 60 * dt
        shift = amp * np.sin(phase)
        m = np.float32([[1, 0, 0], [0, 1, shift]])
        frame = cv2.warpAffine(tex, m, (320, 240), borderMode=cv2.BORDER_REFLECT)
        frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
        est.update(t, frame, BOX, moving=False)
        if int(t * fps) % (fps // 2) == 0:
            est.estimate(t)
        t += dt + rng.uniform(-0.01, 0.01)       # real webcams jitter
    est.estimate(t)
    return t


class BreathingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config("demo")

    def test_rates_within_2_per_min(self):
        for true_rr in (8, 12, 15, 20, 28, 36):
            est = BreathingEstimator(self.cfg)
            synthetic_breathing(est, lambda t, r=true_rr: r, 40)
            with self.subTest(rr=true_rr):
                self.assertIsNotNone(est.rr)
                self.assertLessEqual(abs(est.rr - true_rr), 2.0, f"got {est.rr:.1f}")

    def test_apnea_detected_after_breath_hold(self):
        est = BreathingEstimator(self.cfg)
        t = synthetic_breathing(est, lambda t: 14, 35)
        self.assertEqual(est.apnea_seconds(t), 0.0)
        hold_start = t
        t = synthetic_breathing(est, lambda t: 14, 12, amp=0.0, t0=t)  # holds breath for 12 s
        ap = est.apnea_seconds(t)
        self.assertGreaterEqual(ap, self.cfg.apnea_s, f"apnea {ap:.1f}s after {t - hold_start:.1f}s hold")

    def test_no_apnea_while_breathing(self):
        est = BreathingEstimator(self.cfg)
        t0 = synthetic_breathing(est, lambda t: 16, 30)
        worst = 0.0
        t = t0
        for k in range(6):
            t = synthetic_breathing(est, lambda t: 16, 5, t0=t, seed=k + 1)
            worst = max(worst, est.apnea_seconds(t))
        self.assertLess(worst, self.cfg.apnea_s)

    def test_fast_breathing_rate_climbs(self):
        est = BreathingEstimator(self.cfg)
        t = synthetic_breathing(est, lambda t: 14, 30)
        calm = est.rr
        synthetic_breathing(est, lambda t: 32, 25, t0=t)
        self.assertGreater(est.rr, calm + 10)


class HeartRateTests(unittest.TestCase):
    def test_pos_on_synthetic_pulse(self):
        cfg = make_config("demo")
        est = HeartRateEstimator(cfg)
        rng = np.random.default_rng(1)
        fps, hr = 15, 72
        for i in range(int(14 * fps)):
            t = i / fps
            pulse = np.sin(2 * np.pi * hr / 60 * t)
            # skin colour: pulse is strongest in green (BGR order for OpenCV frames)
            bgr = np.array([120 + 0.3 * pulse, 150 + 1.0 * pulse, 190 + 0.5 * pulse]) + rng.normal(0, 0.3, 3)
            frame = np.tile(bgr, (40, 40, 1)).astype(np.float32)
            est.update(t, frame, (0, 0, 40, 40), still=True)
        est.estimate(t)
        self.assertIsNotNone(est.hr)
        self.assertLess(abs(est.hr - hr), 5)


def person(sh, hip, knee=None, wrist=None, conf=0.9):
    kps = np.zeros((17, 2))
    c = np.zeros(17)
    for idx in (5, 6):
        kps[idx], c[idx] = sh, conf
    for idx in (11, 12):
        kps[idx], c[idx] = hip, conf
    if knee is not None:
        for idx in (13, 14):
            kps[idx], c[idx] = knee, conf
    if wrist is not None:
        kps[9], c[9] = wrist, conf
    pts = kps[c > 0]
    box = np.array([*pts.min(0) - 20, *pts.max(0) + 20])
    return Person(1, box, kps, c, 0.9)


LYING = person((200, 200), (330, 205), (430, 210))
SITTING = person((210, 100), (215, 210), (330, 215))
STANDING = person((210, 60), (212, 180), (214, 300))
FLOOR = person((200, 420), (330, 425), (430, 430))
SHAPE = (480, 640, 3)


class PostureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config("demo")
        self.p = PostureTracker(self.cfg)
        self.p.set_zone((0.15, 0.2, 0.85, 0.6))
        self.t = 0.0

    def hold(self, who, seconds=2.0):
        events = []
        end = self.t + seconds
        while self.t < end:
            events += self.p.update(self.t, who, SHAPE)
            self.t += 0.1
        return events

    def test_classify(self):
        self.assertEqual(self.p.classify(LYING, SHAPE), "lying")
        self.assertEqual(self.p.classify(SITTING, SHAPE), "sitting")
        self.assertEqual(self.p.classify(STANDING, SHAPE), "standing")
        self.assertEqual(self.p.classify(FLOOR, SHAPE), "floor")
        # curled up in bed, legs toward the camera: torso looks upright but the body is wider than tall
        curled = person((300, 200), (305, 250), (308, 300), wrist=(480, 260))
        self.assertEqual(self.p.classify(curled, SHAPE), "lying")

    def test_demo_sequence(self):
        self.assertEqual(self.hold(LYING), [])
        self.assertEqual(self.hold(SITTING), ["bed_edge"])
        self.assertEqual(self.hold(STANDING), ["bed_exit"])
        self.assertEqual(self.hold(FLOOR), ["fall"])

    def test_brief_wobble_is_ignored(self):
        self.hold(LYING)
        self.assertEqual(self.hold(SITTING, 0.4), [])
        self.assertEqual(self.p.state, "lying")

    def test_help_gesture(self):
        self.hold(LYING)
        raised = person((200, 200), (330, 205), (430, 210), wrist=(190, 60))
        self.assertIn("help", self.hold(raised, 3.5))

    def test_auto_zone_detects_floor(self):
        p = PostureTracker(self.cfg)
        for _ in range(20):
            p.observe_for_zone(LYING, SHAPE)
        p.finalize_zone()
        self.assertIsNotNone(p.bed_zone)
        self.assertEqual(p.classify(LYING, SHAPE), "lying")
        self.assertEqual(p.classify(FLOOR, SHAPE), "floor")


class RiskTests(unittest.TestCase):
    def test_news_scores(self):
        self.assertEqual([news_rr(r) for r in (6, 10, 16, 22, 30)], [3, 1, 0, 2, 3])

    def test_fast_breathing_goes_red(self):
        cfg = make_config("demo")
        b, r = Baseline(cfg.baseline_s), RiskEngine(cfg)
        t = 0.0
        while t < 60:
            b.add(t, 14 + np.sin(t), True)
            self.assertEqual(r.update(t, 14 + np.sin(t), True, None, False, b), "green")
            t += 0.5
        self.assertTrue(b.ready)
        rr = 14.0
        while t < 100:
            rr = min(32, rr + 0.5)
            level = r.update(t, rr, True, None, False, b)
            t += 0.5
        self.assertEqual(level, "red")
        self.assertIn(r.trend, ("up", "up2"))


class AlertTests(unittest.TestCase):
    def test_dedupe_escalate_ack(self):
        cfg = make_config("demo")
        with tempfile.TemporaryDirectory() as d:
            am = AlertManager(cfg, EventLog(d))
            a = am.raise_alert("b1", "B1", "rr_high", "warning", "x")
            self.assertIs(am.raise_alert("b1", "B1", "rr_high", "warning", "y"), a)
            am.raise_alert("b1", "B1", "rr_high", "critical", "z")
            self.assertEqual(a.severity, "critical")
            self.assertEqual(len(am.active("b1")), 1)
            am.ack(a.id)
            self.assertEqual(am.active("b1"), [])
            self.assertIsNone(am.raise_alert("b1", "B1", "rr_high", "warning", "again"))  # cooldown


if __name__ == "__main__":
    unittest.main()
