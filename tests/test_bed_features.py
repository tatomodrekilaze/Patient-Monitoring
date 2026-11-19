"""Tests for the bed / hands / walking features: long lie, hidden in bed, line-pull risk, agitation,
drinking, side map, gait speed, delirium watch, lite mode."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.config import make_config
from monitor.events.posture import PostureTracker
from monitor.events.reposition import RepositionTracker
from monitor.pose import Person

from test_pipeline import FLOOR, LYING, SHAPE, STANDING

# COCO ids
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(17)


def body(points, box=None):
    kps, c = np.zeros((17, 2)), np.zeros(17)
    for k, xy in points.items():
        kps[k], c[k] = xy, 0.9
    pts = kps[c > 0]
    box = np.array(box if box is not None else [*pts.min(0) - 20, *pts.max(0) + 20], float)
    return Person(1, box, kps, c, 0.9)


def in_bed(lwr=(260, 250), rwr=(300, 305), lan=(560, 240), ran=(560, 280)):
    """Lying on the back, head left, shown from above: shoulders spread vertically."""
    return body({NOSE: (150, 260), LEYE: (145, 252), REYE: (145, 268), LEAR: (160, 245), REAR: (160, 275),
                 LSH: (200, 230), RSH: (200, 290), LEL: (250, 220), REL: (230, 300),
                 LWR: lwr, RWR: rwr, LHIP: (330, 240), RHIP: (330, 280), LKN: (440, 240), RKN: (440, 280),
                 LAN: lan, RAN: ran})


def run(det, frames, dt=0.1, t0=0.0, **kw):
    events, t = [], t0
    for p in frames:
        events += det.update(t, p, **kw)
        t += dt
    return events, t


class LongLieTests(unittest.TestCase):
    def test_floor_long_after_fall(self):
        cfg = make_config("demo", floor_long_s=10)
        p = PostureTracker(cfg)
        p.set_zone((0.15, 0.2, 0.85, 0.6))
        ev, t = [], 0.0
        for who, secs in ((LYING, 3), (FLOOR, 15)):
            for _ in range(int(secs * 10)):
                ev += p.update(t, who, SHAPE)
                t += 0.1
        self.assertIn("fall", ev)
        self.assertEqual(ev.count("floor_long"), 1)
        self.assertGreater(p.floor_seconds(t), 10)

    def test_kneel_and_get_up_has_no_long_lie(self):
        cfg = make_config("demo", floor_long_s=10)
        p = PostureTracker(cfg)
        p.set_zone((0.15, 0.2, 0.85, 0.6))
        ev, t = [], 0.0
        for who, secs in ((STANDING, 3), (FLOOR, 4), (STANDING, 15)):
            for _ in range(int(secs * 10)):
                ev += p.update(t, who, SHAPE)
                t += 0.1
        self.assertNotIn("floor_long", ev)


class HiddenInBedTests(unittest.TestCase):
    def test_lost_in_bed_is_not_absent(self):
        cfg = make_config("demo", hidden_in_bed_s=30)
        p = PostureTracker(cfg)
        ev, t = [], 0.0
        for _ in range(30):
            ev += p.update(t, LYING, SHAPE)
            t += 0.1
        for _ in range(200):                         # 20 s not detected (under the blanket)
            ev += p.update(t, None, SHAPE)
            t += 0.1
        self.assertNotIn("absent", ev)
        self.assertTrue(p.hidden)
        for _ in range(150):                         # 35 s in total: now it counts
            ev += p.update(t, None, SHAPE)
            t += 0.1
        self.assertIn("absent", ev)

    def test_lost_while_standing_is_absent_fast(self):
        p = PostureTracker(make_config("demo"))
        ev, t = [], 0.0
        for _ in range(30):
            ev += p.update(t, STANDING, SHAPE)
            t += 0.1
        for _ in range(80):
            ev += p.update(t, None, SHAPE)
            t += 0.1
        self.assertEqual(p.state, "absent")


class LiteTests(unittest.TestCase):
    def test_lite_profile(self):
        c = make_config("demo", lite=True)
        self.assertEqual((c.imgsz, c.hr_enabled), (320, False))     # pose size kept: 256 missed falls
        self.assertEqual(make_config("demo", lite=True, imgsz=480).imgsz, 480)


if __name__ == "__main__":
    unittest.main()
