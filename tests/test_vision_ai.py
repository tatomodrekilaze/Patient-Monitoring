"""Tests for the vision-AI features: scene objects (bed zone, cup at mouth), bed-exit prediction,
room zones (bathroom, door, hand hygiene), face AI (pain, eyes, asymmetry), dynamic fall risk."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.config import make_config
from monitor.events.face import FaceAnalyzer, droop, head_pose, pain_score, whole_face
from monitor.scene import SceneDetector

from test_bed_features import LAN, LKN, LSH, LWR, NOSE, RAN, RKN, RSH, RWR, LHIP, RHIP, body, in_bed

SHAPE = (480, 640, 3)
FRAME = np.zeros(SHAPE, np.uint8)


class SceneTests(unittest.TestCase):
    def test_bed_zone_from_detector(self):
        det = lambda f: [("bed", np.array([100, 200, 500, 400]), 0.8), ("chair", np.array([10, 10, 60, 90]), 0.6)]  # noqa: E731
        s = SceneDetector(make_config("demo"), detector=det)
        for i in range(6):
            s.update(float(i), FRAME, None, None)
        z = s.bed_zone(SHAPE)
        self.assertIsNotNone(z)
        self.assertAlmostEqual(z[0], 80 / 640, places=2)
        self.assertAlmostEqual(z[3], 410 / 480, places=2)
        self.assertIn("chair", [o[0] for o in s.overlay()])

    def test_cup_at_mouth_is_intake(self):
        p = in_bed()                                   # nose at (150, 260)
        det = lambda f: [("cup", np.array([135, 250, 170, 290]), 0.7)]   # noqa: E731
        s = SceneDetector(make_config("demo"), detector=det)
        ev = []
        for i in range(30):
            ev += s.update(i * 0.1, FRAME, p, 130.0)
        self.assertEqual(ev, ["intake"])

    def test_cup_on_table_is_not_intake(self):
        det = lambda f: [("cup", np.array([500, 400, 540, 440]), 0.7)]   # noqa: E731
        s = SceneDetector(make_config("demo"), detector=det)
        ev = []
        for i in range(30):
            ev += s.update(i * 0.1, FRAME, in_bed(), 130.0)
        self.assertEqual(ev, [])


def mesh(droop_px=0.0, roll_deg=0.0):
    """Minimal 478-point face: eye corners and mouth corners; optional droop of one corner, head roll."""
    lm = np.zeros((478, 2))
    lm[33], lm[263] = (60, 80), (140, 80)
    lm[61], lm[291] = (75, 140), (125, 140 + droop_px)
    c, a = np.array([100, 100]), np.radians(roll_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (lm - c) @ rot.T + c


FACE3D = {33: (-45, -20, 5), 263: (45, -20, 5), 61: (-25, 45, -5), 291: (25, 45, -5), 1: (0, 15, -35),
          234: (-70, 5, 40), 454: (70, 5, 40), 152: (0, 90, 0), 10: (0, -80, 10)}


def face3d(yaw=0.0, pitch=0.0, sag=0.0, dist=None):
    """3-D face points rotated and projected (orthographic, or perspective at dist): head pose that a
    2-D droop measure can mistake for a droop. sag: one mouth corner lowered (a real droop)."""
    lm = np.zeros((478, 2))
    y, p = np.radians([yaw, pitch])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    for k, v in FACE3D.items():
        v = np.array(v, float) + (0, sag if k == 291 else 0, 0)
        q = rx @ ry @ v
        lm[k] = q[:2] * (1.0 if dist is None else dist / (dist + q[2])) + 200
    return lm


class FaceTests(unittest.TestCase):
    def test_pain_score(self):
        self.assertLess(pain_score({}), 0.5)
        grimace = {"browDownLeft": .8, "browDownRight": .8, "eyeSquintLeft": .7, "eyeSquintRight": .7,
                   "noseSneerLeft": .6, "noseSneerRight": .6}
        self.assertGreater(pain_score(grimace), 6)
        smile = {"mouthSmileLeft": .9, "mouthSmileRight": .9, "cheekSquintLeft": .5, "cheekSquintRight": .5}
        self.assertLess(pain_score(smile), 2)

    def test_droop_ignores_head_roll(self):
        self.assertAlmostEqual(droop(mesh(0, 0)), droop(mesh(0, 20)), places=6)
        self.assertGreater(abs(droop(mesh(12, 15)) - droop(mesh(0, 15))), 0.12)

    def run_face(self, frames):
        """frames: list of (landmarks, blendshapes) fed straight into the analysis, 2 per second."""
        f = FaceAnalyzer(make_config("demo", asym_hold_s=5, face_baseline_n=10))
        ev = []
        for i, (lm, bs) in enumerate(frames):
            ev += f._analyse(i * 0.5, lm, bs)
        return f, ev

    def test_new_droop_alerts(self):
        f, ev = self.run_face([(mesh(0), {})] * 20 + [(mesh(12), {})] * 30)
        self.assertIn("face_asymmetry", ev)

    def test_existing_asymmetry_is_baseline(self):
        f, ev = self.run_face([(mesh(12), {})] * 60)    # the patient's face was always like this
        self.assertNotIn("face_asymmetry", ev)

    def test_head_pose_does_not_alert(self):
        base = [(face3d(), {})] * 20
        for pose in (dict(yaw=25), dict(pitch=25), dict(yaw=10, pitch=20, dist=250)):
            f, ev = self.run_face(base + [(face3d(**pose), {})] * 40)
            self.assertNotIn("face_asymmetry", ev, pose)

    def test_real_droop_alerts_with_small_turn(self):
        f, ev = self.run_face([(face3d(), {})] * 20 + [(face3d(yaw=6, sag=8), {})] * 30)
        self.assertIn("face_asymmetry", ev)

    def test_untrusted_frames_do_not_alert(self):
        f = FaceAnalyzer(make_config("demo", asym_hold_s=5, face_baseline_n=10))
        ev = []
        for i in range(20):
            ev += f._analyse(i * 0.5, face3d(), {})
        for i in range(20, 60):                      # face cut by the frame edge / too close
            ev += f._analyse(i * 0.5, face3d(sag=8), {}, trusted=False)
        self.assertNotIn("face_asymmetry", ev)
        self.assertIsNone(f.asym)

    def test_head_pose_and_whole_face(self):
        self.assertAlmostEqual(head_pose(face3d())[0], 0.0, places=6)
        self.assertGreater(abs(head_pose(face3d(yaw=20))[0]), 0.2)
        self.assertTrue(whole_face(face3d(), (400, 400)))
        self.assertFalse(whole_face(face3d(), (250, 400)))   # chin below the picture

    def test_pain_needs_to_last(self):
        grimace = {"browDownLeft": .8, "browDownRight": .8, "eyeSquintLeft": .7, "eyeSquintRight": .7,
                   "noseSneerLeft": .6, "noseSneerRight": .6}
        f, ev = self.run_face([(mesh(), {})] * 10 + [(mesh(), grimace)] * 2 + [(mesh(), {})] * 10)
        self.assertNotIn("pain_face", ev)              # a one-second grimace
        f, ev = self.run_face([(mesh(), {})] * 10 + [(mesh(), grimace)] * 12)
        self.assertIn("pain_face", ev)

    def test_eyes(self):
        f, _ = self.run_face([(mesh(), {"eyeBlinkLeft": .9, "eyeBlinkRight": .9})] * 10)
        self.assertFalse(f.eyes_open)
        f, _ = self.run_face([(mesh(), {"eyeBlinkLeft": .1, "eyeBlinkRight": .1})] * 10)
        self.assertTrue(f.eyes_open)
