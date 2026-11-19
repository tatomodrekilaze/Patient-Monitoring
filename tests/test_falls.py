"""Fall rule for falls toward the camera (strong evidence) and the kneeling guard."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.config import make_config
from monitor.events.posture import FastFall
from monitor.pose import Person

NOSE, LSH, RSH, LHIP, RHIP = 0, 5, 6, 11, 12


def body(box, head, sh, hip):
    kps, c = np.zeros((17, 2)), np.zeros(17)
    for k, xy in ((NOSE, head), (LSH, sh), (RSH, sh), (LHIP, hip), (RHIP, hip)):
        kps[k], c[k] = xy, 0.9
    return Person(1, np.array(box, float), kps, c, 0.9)


STANDING = body([300, 100, 360, 400], (330, 120), (330, 160), (330, 260))   # 60 x 300, head at 120


def run(end):
    f, t, fired = FastFall(make_config("demo")), 0.0, False
    for _ in range(50):                          # standing 2 s
        fired |= f.update(t, STANDING)
        t += 0.04
    for _ in range(30):                          # then 1.2 s in the end pose
        fired |= f.update(t, end)
        t += 0.04
    return fired


class StrongFallTests(unittest.TestCase):
    def test_fall_toward_camera(self):
        # looks big (perspective): 250 px tall = 0.83 of standing, too tall for the normal rule; head
        # 330 px lower = 1.1 standing heights; torso tipped ~60 deg
        fallen = body([150, 230, 500, 480], (200, 450), (250, 420), (340, 470))
        self.assertTrue(run(fallen))

    def test_kneeling_same_drop_is_not_a_fall(self):
        # same box and head drop, but the upper body stays upright (kneeling, reaching to the floor)
        kneeling = body([150, 230, 500, 480], (330, 450), (330, 300), (335, 400))
        self.assertFalse(run(kneeling))

    def test_normal_rule_still_works(self):
        flat = body([150, 380, 480, 470], (180, 420), (230, 420), (330, 430))   # low and wide on the floor
        self.assertTrue(run(flat))


if __name__ == "__main__":
    unittest.main()


class VanishedMidFallTests(unittest.TestCase):
    def run_drop(self, px_per_frame, frames):
        """Standing 2 s, head (and body) coming down px_per_frame per 0.04 s, then out of view 2 s."""
        f, t = FastFall(make_config("demo")), 0.0
        for _ in range(50):
            f.update(t, STANDING)
            t += 0.04
        for k in range(1, frames + 1):
            d = px_per_frame * k
            f.update(t, body([300, 100 + d, 360, 400], (330, 120 + d), (330, 160 + d), (330, 260)))
            t += 0.04
        fired = False
        while t < 6:
            fired |= f.vanished_mid_fall(t)
            t += 0.04
        return fired

    def test_falls_out_of_view(self):
        self.assertTrue(self.run_drop(30, 12))       # head down 360 px (1.2 standing heights) in 0.5 s

    def test_slowly_sitting_out_of_view_is_not_a_fall(self):
        self.assertFalse(self.run_drop(6, 12))       # 72 px in 0.5 s: someone sitting down off screen

    def test_walking_out_of_view_is_not_a_fall(self):
        self.assertFalse(self.run_drop(0, 12))


class NearFallTests(unittest.TestCase):
    def run_motion(self, dip_fn, seconds):
        """Standing 2 s, then the head (and box top) moves by dip_fn(time since start) px."""
        f, t, near, fell = FastFall(make_config("demo")), 0.0, 0, False
        for _ in range(50):
            f.update(t, STANDING)
            t += 0.04
        t0 = t
        while t - t0 < seconds:
            d = dip_fn(t - t0)
            fell |= f.update(t, body([300, 100 + d, 360, 400], (330, 120 + d), (330, 160 + d), (330, 260)))
            near += f.near_fall
            t += 0.04
        return near, fell

    def test_stumble_caught(self):
        # 75 px (25 % of standing height) dip in 0.2 s, back up by 0.8 s
        dip = lambda s: 75 * min(1, s / 0.2) if s < 0.4 else max(0.0, 75 * (1 - (s - 0.4) / 0.4))  # noqa: E731
        near, fell = self.run_motion(dip, 2.0)
        self.assertEqual((near, fell), (1, False))

    def test_slow_bend_is_not_a_stumble(self):
        # bending 120 px over 1 s, staying down 1 s, up over 1 s (picking something up)
        bend = lambda s: 120 * min(1, s) if s < 2 else max(0.0, 120 * (3 - s))  # noqa: E731
        self.assertEqual(self.run_motion(bend, 4.0)[0], 0)

    def test_walking_bob_is_not_a_stumble(self):
        bob = lambda s: 6 * np.sin(2 * np.pi * 2 * s)  # noqa: E731
        self.assertEqual(self.run_motion(bob, 4.0)[0], 0)
