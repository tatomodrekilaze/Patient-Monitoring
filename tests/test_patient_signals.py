"""Tests for hand signals, chest-pain sign, alert feedback, sleep report, voice-note parsing,
nurse->patient messages through the cloud relay."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.alerts import AlertManager, EventLog
from monitor.analytics import alert_stats, sleep_report
from monitor.assistant import _json
from monitor.config import make_config
from monitor.events.hands import HandSignals, read_hand

from test_pipeline import person


def hand(fingers_up, thumb="in"):
    """Synthetic MediaPipe landmarks, palm facing the camera, fingers pointing up (y grows down)."""
    lm = np.zeros((21, 2))
    lm[0] = (100, 200)                                  # wrist
    for k, x in enumerate((85, 98, 111, 124)):          # index..pinky
        mcp, pip, dip, tip = 5 + 4 * k, 6 + 4 * k, 7 + 4 * k, 8 + 4 * k
        lm[mcp] = (x, 150)
        if fingers_up[k]:
            lm[pip], lm[dip], lm[tip] = (x, 120), (x, 105), (x, 90)
        else:
            lm[pip], lm[dip], lm[tip] = (x, 132), (x, 145), (x, 152)
    lm[1], lm[2] = (80, 185), (70, 170)
    if thumb == "out":
        lm[3], lm[4] = (58, 160), (45, 150)
    elif thumb == "up":
        lm[3], lm[4] = (75, 100), (75, 70)
    else:
        lm[3], lm[4] = (82, 165), (92, 160)
    return lm


class ReadHandTests(unittest.TestCase):
    def test_finger_counts(self):
        self.assertEqual(read_hand(hand([1, 0, 0, 0])), 1)
        self.assertEqual(read_hand(hand([1, 1, 0, 0])), 2)
        self.assertEqual(read_hand(hand([1, 1, 1, 0])), 3)
        self.assertEqual(read_hand(hand([1, 1, 1, 1])), 4)
        self.assertEqual(read_hand(hand([1, 1, 1, 1], thumb="out")), 5)
        self.assertEqual(read_hand(hand([0, 0, 0, 0])), 0)

    def test_thumbs_up(self):
        self.assertEqual(read_hand(hand([0, 0, 0, 0], thumb="up")), "ok")

    def test_thumbs_down(self):
        lm = hand([0, 0, 0, 0], thumb="up")
        lm[:, 1] = 300 - lm[:, 1]                       # the same fist turned upside down: 👎
        self.assertEqual(read_hand(lm), "bad")


class HandSignalTests(unittest.TestCase):
    def make(self, signal_sequence):
        cfg = make_config("demo")
        hs = HandSignals(cfg)
        seq = iter(signal_sequence)
        hs._read = lambda frame, w, e, length: next(seq)        # skip MediaPipe, feed readings
        return cfg, hs

    def test_two_fingers_held_gives_toilet_once(self):
        cfg, hs = self.make([2] * 100)
        raised = person((210, 200), (215, 310), (330, 315), wrist=(200, 60))
        frame = np.zeros((480, 640, 3), np.uint8)
        events, t = [], 0.0
        while t < 5:
            events += hs.update(t, frame, raised, "reclined")
            t += 0.1
        self.assertEqual(events, ["req_toilet"])

    def test_brief_flash_ignored_and_thumbs_up(self):
        cfg, hs = self.make([3] * 5 + ["ok"] * 100)
        raised = person((210, 200), (215, 310), (330, 315), wrist=(200, 60))
        frame = np.zeros((480, 640, 3), np.uint8)
        events, t = [], 0.0
        while t < 6:
            events += hs.update(t, frame, raised, "lying")
            t += 0.1
        self.assertEqual(events, ["patient_ok"])

class FeedbackSleepTests(unittest.TestCase):
    def test_false_alarm_rate(self):
        cfg = make_config("demo")
        with tempfile.TemporaryDirectory() as d:
            am = AlertManager(cfg, EventLog(d))
            a = am.raise_alert("b", "B", "bed_edge", "warning", "x")
            b = am.raise_alert("b", "B", "fall", "critical", "y")
            c = am.raise_alert("b", "B", "apnea", "critical", "z")
            am.feedback(a.id, "false")
            am.feedback(b.id, "true")
            am.feedback(c.id, "true")
            self.assertFalse(am.feedback(c.id, "maybe"))
            s = alert_stats(am.all())
            self.assertEqual(s["rated"], 3)
            self.assertEqual(s["false_alarm_pct"], 33)
            self.assertEqual(s["precision_by_kind"]["bed_edge"]["precision_pct"], 0)
            self.assertTrue(a.acked)

    def test_sleep_report(self):
        with tempfile.TemporaryDirectory() as d:
            log = EventLog(d)
            now = time.time()
            pattern = ["awake"] * 6 + ["asleep"] * 30 + ["awake"] * 3 + ["asleep"] * 20
            for i, s in enumerate(pattern):
                log.entries.append({"ts": now - 600 + i * 10, "bed_id": "bed1", "kind": "vitals",
                                    "rr": 13.0 if s == "asleep" else 16.0, "posture": "lying", "sleep": s})
            log.entries.append({"ts": now - 300, "bed_id": "bed1", "kind": "apnea_episode"})
            r = sleep_report(log, "bed1")
            self.assertEqual(r["asleep_min"], 8)
            self.assertEqual(r["wakeups"], 1)
            self.assertEqual(r["longest_sleep_min"], 5)
            self.assertEqual(r["rr_asleep_median"], 13.0)
            self.assertEqual(r["apnea_episodes"], 1)


class AIParsingTests(unittest.TestCase):
    def test_json_with_fences(self):
        self.assertEqual(_json('```json\n{"summary": "ok"}\n```', {}), {"summary": "ok"})
        self.assertEqual(_json("not json", {"transcript": "x"}), {"transcript": "x"})

    def test_ai_without_project_gives_georgian_error(self):
        from monitor.assistant import WardAssistant
        ai = WardAssistant(make_config("demo", ai_provider="gemini", vertex_project=""))
        self.assertIn("--vertex-project", ai.ask("?", "{}"))
        self.assertIn("error", ai.voice_note(b"abc", "audio/webm"))


class RelayMessageTests(unittest.TestCase):
    def test_nurse_message_from_cloud_reaches_laptop(self):
        from cloud import RemoteWard
        from monitor.relay import execute
        from monitor.server import Ward

        class FakeBed:
            snap = {"id": "bed1", "name": "Bed 1", "level": "green", "simulated": False}
            capture = None

            def __init__(self):
                self.got, self.notes = [], []

            def say_to_patient(self, **kw):
                pass

            def nurse_message(self, **kw):
                self.got.append(kw)
                return kw

            def add_note(self, note):
                self.notes.append(note)
                return note

        with tempfile.TemporaryDirectory() as d:
            cfg = make_config("demo", data_dir=d)
            log = EventLog(d)
            bed = FakeBed()
            edge = Ward(cfg, {"bed1": bed}, AlertManager(cfg, log), log, 8000)
            cloud = RemoteWard(cfg, 8080)
            cloud.ingest({"state": edge.state()})
            cloud.say("bed1", preset="n_coming")
            cloud.say_audio("bed1", b"\x1aE\xdf\xa3fakewebm", "audio/webm")
            cloud.add_note("bed1", {"summary": "pain 6/10", "urgency": "soon"})
            for cmd in cloud.ingest({"state": edge.state()}):
                execute(edge, cmd)
            self.assertEqual(bed.got[0], {"preset": "n_coming"})
            self.assertEqual(bed.got[1]["audio"], b"\x1aE\xdf\xa3fakewebm")
            self.assertEqual(bed.notes[0]["summary"], "pain 6/10")


if __name__ == "__main__":
    unittest.main()
