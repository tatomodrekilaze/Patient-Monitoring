"""Skeleton replay, fall impact speed, doctor's report, respiratory distress score."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.alerts import EventLog
from monitor.analytics import clinical_report
from monitor.config import make_config
from monitor.pose import Person
from monitor.replay import Recorder


LSH, RSH, LHIP, RHIP = 5, 6, 11, 12


def skeleton(x=300.0, y=200.0, sh_lift=0.0, tid=1):
    kps, c = np.zeros((17, 2)), np.full(17, 0.9)
    kps[:] = (x, y)
    kps[LSH], kps[RSH] = (x - 30 - sh_lift, y - 50), (x - 30 - sh_lift, y + 50)     # shoulders, head to the left
    kps[LHIP], kps[RHIP] = (x + 120, y - 40), (x + 120, y + 40)
    return Person(tid, np.array([x - 80, y - 60, x + 200, y + 60]), kps, c, 0.9)


class ReplayTests(unittest.TestCase):
    def test_captures_before_and_after(self):
        rec = Recorder(tempfile.mkdtemp())
        done = []
        for i in range(400):                                   # 40 s at 10 fps
            t = i / 10
            if i == 300:
                self.assertTrue(rec.request(7, "fall", "fall!", t, {"impact_mps": 3.2}))
            done += rec.add(t, (360, 640, 3), skeleton(300 + i), [], "standing", "green")
        self.assertEqual(done, [7])
        r = rec.get(7)
        self.assertEqual(r["impact_mps"], 3.2)
        ts = [f["t"] for f in r["frames"]]
        self.assertAlmostEqual(min(ts), 30 - 25, delta=0.2)
        self.assertAlmostEqual(max(ts), 30 + 5, delta=0.2)
        self.assertTrue(r["frames"][0]["people"][0]["patient"])
        rec.replays.clear()
        self.assertIsNotNone(rec.get(7))                       # reloaded from disk

    def test_routine_reminders_have_no_replay(self):
        rec = Recorder(tempfile.mkdtemp())
        self.assertFalse(rec.request(1, "rounding", "", 0.0))
        self.assertTrue(rec.request(2, "apnea", "", 0.0))
        self.assertFalse(rec.request(2, "apnea", "", 1.0))   # the same alert raised again


class ReportTests(unittest.TestCase):
    def test_report_counts(self):
        log = EventLog(tempfile.mkdtemp())
        for i in range(120):                                   # 20 min of vitals, lying
            log.add("b1", "vitals", rr=14 + (i % 3), level="green", posture="lying", sleep="asleep")
        log.add("b1", "baseline", rr=15.0, rr_sd=1.5, hr=None)
        for _ in range(3):
            log.add("b1", "apnea_episode")
        log.add("b1", "alert", alert="apnea", severity="critical", message="apnea")
        r = clinical_report(log, "b1", {"name": "Bed 1", "bed_exits": 2}, hours=1)
        b = r["breathing"]
        self.assertEqual(b["apneas"], 3)
        self.assertEqual(b["baseline"], 15.0)
        self.assertEqual(r["alerts"]["by_kind"], {"apnea": 1})
        self.assertEqual(r["mobility"]["bed_exits"], 2)
