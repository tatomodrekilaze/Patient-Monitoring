"""Tests for the camera path: keypoint smoothing, full-resolution side frame, breathing units."""
import os
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.capture import Capture
from monitor.config import make_config
from monitor.pose import Person
from monitor.signals.breathing import BreathingEstimator
from monitor.smooth import KeypointSmoother


def person(kps):
    return Person(1, np.array([0, 0, 100, 100]), np.array(kps, float), np.full(len(kps), 0.9), 0.9)


class SmoothingTests(unittest.TestCase):
    def test_removes_jitter_when_still(self):
        s, rng = KeypointSmoother(), np.random.default_rng(0)
        raw, out = [], []
        for i in range(150):
            k = 200 + rng.normal(0, 3, (17, 2))
            raw.append(k[0, 0])
            out.append(s(i / 25, [person(k)])[0].kps[0, 0])
        self.assertLess(np.std(out[30:]), 0.5 * np.std(raw[30:]))

    def test_follows_fast_movement(self):
        s = KeypointSmoother()
        for i in range(50):                               # still, then a fall-speed drop of 300 px in 0.5 s
            s(i / 25, [person(np.full((17, 2), 100.0))])
        y = None
        for j in range(1, 13):
            y = s((50 + j) / 25, [person(np.full((17, 2), 100.0 + 25 * j))])[0].kps[0, 1]
        self.assertGreater(y, 100 + 0.8 * 300)            # within 20 % of the true position at the end


class FullFrameTests(unittest.TestCase):
    def test_big_video_keeps_full_frame(self):
        path = os.path.join(tempfile.mkdtemp(), "big.mp4")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
        for i in range(30):
            w.write(np.full((720, 1280, 3), i * 5, np.uint8))
        w.release()
        cap = Capture(path, width=640, loop=False)
        try:
            r = cap.read(0, timeout=3)
            self.assertIsNotNone(r)
            seq, t, frame = r
            self.assertEqual(frame.shape[1], 640)
            full = cap.full_frame(seq)
            if full is not None:                          # None only if a newer frame arrived meanwhile
                self.assertEqual(full.shape[1], 1280)
        finally:
            cap.release()


def chest_frames(scale, n=600, fps=15, rr=15):
    """A textured patch moving up and down like a breathing chest, drawn at a given resolution."""
    rng = np.random.default_rng(1)
    tex = (rng.random((60, 60)) * 255).astype(np.uint8)
    tex = cv2.resize(tex, (60 * scale, 60 * scale), interpolation=cv2.INTER_NEAREST)
    for i in range(n):
        img = np.zeros((200 * scale, 200 * scale), np.uint8)
        dy = 1.5 * scale * np.sin(2 * np.pi * rr / 60 * i / fps)
        m = np.float32([[1, 0, 70 * scale], [0, 1, 70 * scale + dy]])
        img = cv2.warpAffine(cv2.copyMakeBorder(tex, 0, 140 * scale, 0, 140 * scale, cv2.BORDER_CONSTANT), m,
                             (200 * scale, 200 * scale))
        yield i / fps, img


class BreathingUnitTests(unittest.TestCase):
    def run_est(self, scale, roi):
        cfg = make_config("demo")
        b = BreathingEstimator(cfg)
        b.roi_px = roi
        box = (60 * scale, 60 * scale, 140 * scale, 140 * scale)
        amp = []
        for t, img in chest_frames(scale):
            b.update(t, img, box, False, unit=scale)
            if len(b.disp):
                amp.append(b.disp[-1][1])
        b.estimate(40.0)
        return b.rr, np.ptp(amp[-100:])

    def test_full_resolution_same_rate_and_units(self):
        rr1, a1 = self.run_est(1, 96)
        rr3, a3 = self.run_est(3, 128)
        self.assertAlmostEqual(rr1, 15, delta=1.5)
        self.assertAlmostEqual(rr3, 15, delta=1.5)
        self.assertAlmostEqual(a3 / a1, 1.0, delta=0.35)   # same units: the apnea reference does not jump


if __name__ == "__main__":
    unittest.main()
