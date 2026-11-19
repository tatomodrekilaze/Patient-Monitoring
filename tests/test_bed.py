"""The trimmed bed pipeline: feature-status list, unclear posture, in-bed-only turning timer."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor import i18n
from monitor.config import make_config
from monitor.events.posture import PostureTracker
from monitor.events.reposition import RepositionTracker

from test_exam import TEXTURE
from test_pipeline import LYING, SHAPE, SITTING, STANDING, person

# standing right in front of the lens: shoulders above the top edge, hips invented at the bottom edge
TOO_CLOSE = person((320, 2), (330, 478), None)
# upright torso with the legs out of view: sitting and standing look the same
NO_LEGS = person((210, 60), (212, 180), None)


class UnclearPostureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config("demo")
        self.p = PostureTracker(self.cfg)
        self.p.set_zone((0.15, 0.2, 0.85, 0.6))
        self.t = 0.0

    def hold(self, who, seconds=3.0):
        events = []
        end = self.t + seconds
        while self.t < end:
            events += self.p.update(self.t, who, SHAPE)
            self.t += 0.1
        return events

    def test_cut_off_body_is_unclear_not_lying(self):
        self.hold(STANDING)
        self.assertEqual(self.p.state, "standing")
        self.hold(TOO_CLOSE)
        self.assertEqual(self.p.state, "unknown")

    def test_no_legs_is_not_guessed(self):
        self.hold(NO_LEGS)
        self.assertEqual(self.p.state, "unknown")

    def test_bed_exit_across_an_unclear_stretch(self):
        self.hold(LYING)
        self.assertEqual(self.p.state, "lying")
        self.hold(TOO_CLOSE)                     # sat up close to the camera, hips hidden
        events = self.hold(STANDING)
        self.assertIn("bed_exit", events)

    def test_back_in_bed_after_unclear(self):
        self.hold(LYING)
        self.hold(TOO_CLOSE)
        self.assertEqual(self.hold(LYING), [])   # never left the bed: no event

    def test_sitting_still_recognised(self):
        self.hold(SITTING)
        self.assertEqual(self.p.state, "sitting")


class TurningTimerTests(unittest.TestCase):
    def test_only_time_in_bed_counts(self):
        r = RepositionTracker(make_config("demo"))
        t = 0.0
        for _ in range(600):                     # 60 s sitting in a chair: no minutes
            r.update(t, SITTING, in_bed=False, out_of_bed=False)
            t += 0.1
        self.assertEqual(r.minutes_in_position(t), 0.0)
        for _ in range(600):                     # 60 s in bed
            r.update(t, LYING, in_bed=True, out_of_bed=False)
            t += 0.1
        self.assertAlmostEqual(r.minutes_in_position(t), 1.0, delta=0.05)

    def test_standing_resets(self):
        r = RepositionTracker(make_config("demo"))
        t = 0.0
        for _ in range(300):
            r.update(t, LYING, in_bed=True, out_of_bed=False)
            t += 0.1
        r.update(t, STANDING, in_bed=False, out_of_bed=True)
        self.assertEqual(r.minutes_in_position(t), 0.0)


class FeatureStatusTests(unittest.TestCase):
    """Real bed pipeline (pose model, no camera): every feature is listed with its state."""

    @classmethod
    def setUpClass(cls):
        import monitor.bed as mb
        from monitor.alerts import AlertManager, EventLog
        from monitor.bed import BedMonitor, Settings

        class NoCapture:                              # frames are fed straight into process()
            restarts = 0

            def __init__(self, *a, **k):
                pass

            def release(self):
                pass
        cls._orig_capture, mb.Capture = mb.Capture, NoCapture
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = tempfile.mkdtemp()
        cfg = make_config("demo", data_dir=d, patient_prompts=False, face_ai=False, scene_ai=False,
                          pose_model=os.path.join(root, "yolo11n-pose.pt"))
        log = EventLog(d)
        cls.bed = BedMonitor("b1", "test", cfg, AlertManager(cfg, log), log, Settings(d), source="none.mp4")

    @classmethod
    def tearDownClass(cls):
        import monitor.bed as mb
        mb.Capture = cls._orig_capture

    def step(self, seconds, t0=0.0, fps=10):
        for i in range(int(seconds * fps)):
            self.bed.process(t0 + i / fps, TEXTURE.copy())
        return t0 + seconds

    def test_every_feature_listed_with_a_reason(self):
        self.step(2.0)
        f = {x["key"]: x for x in self.bed.snap["features"]}
        self.assertEqual(set(f), set(i18n.FEATURE))
        self.assertEqual(f["breathing"]["state"], "wait")
        self.assertEqual(f["apnea"]["detail"], i18n.FEATURE_WAIT["no_patient"])
        self.assertTrue(all(x["detail"] for x in f.values()))

    def test_exams_show_running(self):
        t = self.step(1.0, 10.0)
        self.bed.start_chair_stand()
        self.step(1.0, t)
        f = {x["key"]: x for x in self.bed.snap["features"]}
        self.assertEqual(f["chair_stand"]["state"], "on")

    def test_camera_offline_is_a_technical_alarm(self):
        b = self.bed
        self.step(1.0, 30.0)
        active = lambda: [a.kind for a in b.alerts.active("b1")]       # noqa: E731
        b._last_frame_wall = 1000.0
        b.camera_silent(1005.0)                                         # 5 s: a Wi-Fi hiccup, nothing yet
        self.assertNotIn("camera_offline", active())
        b.camera_silent(1010.0)
        self.assertIn("camera_offline", active())
        self.assertIsNone(b.snap["rr"])                                 # no stale vitals on screen
        self.assertTrue(all(f["state"] == "off" for f in b.snap["features"]))
        b.camera_alive(1011.0)
        self.assertNotIn("camera_offline", active())

    def test_magnify_view_switch(self):
        b = self.bed
        b.set_magnify("pulse")
        self.step(0.5, 40.0)
        self.assertEqual(b.snap["magnify"], "pulse")
        b.set_magnify(None)
        self.step(0.3, 41.0)
        self.assertIsNone(b.snap["magnify"])

    def test_snapshot_has_no_cut_features(self):
        self.step(0.5, 20.0)
        for k in ("news2", "cpr", "sepsis", "rigors", "tremor", "ahi", "fall_risk", "agitation"):
            self.assertNotIn(k, self.bed.snap)


if __name__ == "__main__":
    unittest.main()
