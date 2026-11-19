"""The breathing validity gate: no measurement (and no alarm) unless the chest is really in view."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.config import make_config
from monitor.events.posture import PostureTracker
from monitor.pose import Person, torso_view
from monitor.signals.breathing import BreathingEstimator

SHAPE = (360, 640, 3)
LSH, RSH, LHIP, RHIP, NOSE = 5, 6, 11, 12, 0


def body(sh_y, hip_y, x=320, spread=60, conf=0.9, hip_conf=None):
    kps, c = np.zeros((17, 2)), np.zeros(17)
    kps[NOSE], c[NOSE] = (x, sh_y - 40), conf
    kps[LSH], kps[RSH] = (x - spread, sh_y), (x + spread, sh_y)
    kps[LHIP], kps[RHIP] = (x - spread * 0.7, hip_y), (x + spread * 0.7, hip_y)
    c[[LSH, RSH]] = conf
    c[[LHIP, RHIP]] = conf if hip_conf is None else hip_conf
    pts = kps[c > 0]
    return Person(1, np.array([*pts.min(0) - 20, *pts.max(0) + 20]), kps, c, 0.9)


class TorsoViewTests(unittest.TestCase):
    def test_whole_torso(self):
        self.assertIsNone(torso_view(body(120, 240), SHAPE))

    def test_hips_at_frame_edge(self):
        # sitting close to the camera: the pose model puts the hips on the bottom edge
        self.assertEqual(torso_view(body(250, 355), SHAPE), "cut_off")

    def test_hips_under_blanket_still_measured(self):
        self.assertIsNone(torso_view(body(120, 240, hip_conf=0.3), SHAPE))

    def test_hips_not_found(self):
        self.assertEqual(torso_view(body(120, 240, hip_conf=0.1), SHAPE), "torso_hidden")

    def test_tiny_person(self):
        self.assertEqual(torso_view(body(150, 170, spread=5), SHAPE), "too_far")


class PostureEdgeTests(unittest.TestCase):
    def test_invented_hips_are_not_reclined(self):
        p = PostureTracker(make_config("demo"))
        # shoulders high, "hips" on the bottom edge and to the side: torso tilted 40-50 deg
        person = body(200, 356)
        person.kps[LHIP][0] += 150
        person.kps[RHIP][0] += 150
        self.assertEqual(p.classify(person, SHAPE), "unknown")


class PauseTests(unittest.TestCase):
    def test_pause_clears_the_breath_timer(self):
        b = BreathingEstimator(make_config("demo"))
        b.last_breath_t, b.ref_amp, b.recent_amp = 100.0, 1.0, 0.1
        self.assertGreater(b.apnea_seconds(130.0), 20)       # would look like a 30 s apnea
        b.pause()                                             # ... but the chest was out of view
        self.assertEqual(b.apnea_seconds(130.0), 0.0)
        self.assertEqual(b.ref_amp, 1.0)                      # what was learned is kept


if __name__ == "__main__":
    unittest.main()
