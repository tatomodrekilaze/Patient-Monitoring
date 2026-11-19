"""Bedside examination tools: tremor analyzer, Timed Up and Go, goniometer."""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor.config import make_config
from monitor.events.mobility import ChairStandTest
from monitor.events.neuro import FaceTest
from monitor.pose import Person

PTS = {5: (280, 150), 6: (360, 150), 7: (260, 230), 8: (380, 230), 9: (250, 300), 10: (390, 300),
       11: (290, 250), 12: (350, 250), 13: (290, 350), 14: (350, 350), 15: (290, 450), 16: (350, 450)}


def person(pts=PTS):
    kps, c = np.zeros((17, 2)), np.zeros(17)
    for k, xy in pts.items():
        kps[k], c[k] = xy, 0.9
    return Person(1, np.array([240, 100, 400, 470], float), kps, c, 0.9)


TEXTURE = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, (480, 640, 3)).astype(np.uint8), (5, 5), 1.5)


def shake(freq, amp_px, seconds=20, fps=30, noise=0.0):
    cfg = make_config("demo")
    tr, p, events = TremorAnalyzer(cfg), person(), []
    rng = np.random.default_rng(1)
    for i in range(int(seconds * fps)):
        t = i / fps
        dx = amp_px * np.sin(2 * np.pi * freq * t)
        frame = cv2.warpAffine(TEXTURE, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480), borderMode=cv2.BORDER_REFLECT)
        if noise:
            frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
        tr.update(t, p, frame)
        if i % 15 == 0:
            events += tr.analyze(t)
    return tr.result, events


def arms_out(left_drop=0.0, right_drop=0.0):
    pts = dict(PTS)
    pts[7], pts[8] = (230, 150 + left_drop / 2), (410, 150 + right_drop / 2)
    pts[9], pts[10] = (180, 150 + left_drop), (460, 150 + right_drop)
    return person(pts)


def seated(rise):
    """rise: hip above knee in torso lengths (0 = sitting, ~0.9 = standing). Knees stay where the feet
    are; hips and shoulders go up as the person stands."""
    up = 100 * rise
    pts = dict(PTS)
    pts[13], pts[14] = (290, 350), (350, 350)
    pts[11], pts[12] = (290, 350 - up), (350, 350 - up)
    pts[5], pts[6] = (280, 250 - up), (360, 250 - up)
    return person(pts)


class ChairStandTests(unittest.TestCase):
    def test_counts_stands(self):
        c = ChairStandTest()
        c.arm(0.0)
        out, t = None, 0.0
        while out is None and t < 60:
            phase = max(0.0, t - 2.0) % 2.5 / 2.5           # seated 2 s, then a stand every 2.5 s
            rise = 0.1 if t < 2.0 else 0.1 + 0.8 * np.sin(np.pi * phase) ** 2
            out = c.update(t, seated(rise), 0.3)
            t += 0.05
        self.assertEqual(out, 12)                           # 30 s / 2.5 s
        self.assertEqual(c.history, [12])

    def test_half_rises_not_counted(self):
        c = ChairStandTest()
        c.arm(0.0)
        out, t = None, 0.0
        while out is None and t < 60:
            rise = 0.1 if t < 2.0 else 0.1 + 0.2 * np.sin(np.pi * t / 2.5) ** 2    # leaning forward, never up
            out = c.update(t, seated(rise), 0.3)
            t += 0.05
        self.assertEqual(out, None if out is None else 0)

    def test_single_frame_glitch_not_counted(self):
        c = ChairStandTest()
        c.arm(0.0)
        for i in range(200):
            t = i * 0.05
            c.update(t, seated(0.9 if i == 100 else 0.1), 0.3)   # one bad pose frame
        self.assertEqual(c.count, 0)


class FaceTestTests(unittest.TestCase):
    def run_test(self, smile, eyes, sag=0.0):
        from test_vision_ai import face3d
        ft, out, t = FaceTest(make_config("demo")), [], 0.0
        ft.start(0.0)
        while t < 13:
            bs = smile if ft.state == "smile" else eyes
            r = ft.update(t, (t, face3d(sag=sag), bs, True), 0.0)
            if r is not None:
                out.append(r)
            t += 0.5
        return out

    def test_symmetric_face(self):
        out = self.run_test({"mouthSmileLeft": .7, "mouthSmileRight": .7}, {"eyeBlinkLeft": .9, "eyeBlinkRight": .9})
        self.assertEqual(out[0], "next")
        self.assertFalse(out[1]["asymmetric"])
        self.assertTrue(out[1]["measured"])

    def test_one_eye_does_not_close(self):
        out = self.run_test({"mouthSmileLeft": .7, "mouthSmileRight": .7}, {"eyeBlinkLeft": .9, "eyeBlinkRight": .3})
        self.assertTrue(out[1]["asymmetric"])

    def test_mouth_corner_not_lifted(self):
        out = self.run_test({"mouthSmileLeft": .7, "mouthSmileRight": .4}, {"eyeBlinkLeft": .9, "eyeBlinkRight": .9}, sag=8)
        self.assertTrue(out[1]["asymmetric"])

    def test_face_not_seen(self):
        ft = FaceTest(make_config("demo"))
        ft.start(0.0)
        r = [ft.update(t, None, 0.0) for t in (7.0, 14.0)]
        self.assertFalse(r[1]["measured"])
        self.assertFalse(r[1]["asymmetric"])


if __name__ == "__main__":
    unittest.main()
