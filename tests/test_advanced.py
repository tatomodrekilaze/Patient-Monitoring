"""Tests for the advanced features: seizure, repositioning, forecast, escalation, FHIR,
analytics, family view, login, and the edge -> cloud relay."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.alerts import AlertManager, EventLog
from monitor.analytics import validation_stats
from monitor.config import make_config
from monitor.events.posture import PostureTracker
from monitor.events.reposition import RepositionTracker
from monitor.events.risk import Baseline, RiskEngine
from monitor.fhir import bundle
from monitor.pose import Person
from monitor.server import family_status

from test_pipeline import LYING, SHAPE, person


def skeleton(sh=(200, 200), hip=(330, 205), wrist=(260, 150), ankle=(520, 215), face=(1, 1, 1, 1, 1)):
    kps = np.zeros((17, 2))
    c = np.zeros(17)
    kps[0], c[0] = (150, 190), face[0]
    kps[1], c[1] = (145, 185), face[1]
    kps[2], c[2] = (145, 195), face[2]
    kps[3], c[3] = (160, 182), face[3]
    kps[4], c[4] = (160, 198), face[4]
    for i in (5, 6):
        kps[i], c[i] = sh, 0.9
    for i in (11, 12):
        kps[i], c[i] = hip, 0.9
    for i in (13, 14):
        kps[i], c[i] = (430, 210), 0.9
    for i in (15, 16):
        kps[i], c[i] = ankle, 0.9
    for i in (9, 10):
        kps[i], c[i] = wrist, 0.9
    return Person(1, np.array([100, 150, 560, 260]), kps, c, 0.9)


class RepositionTests(unittest.TestCase):
    def test_turn_due_then_reposition(self):
        cfg = make_config("demo", turn_interval_s=30)
        r = RepositionTracker(cfg)
        evs = []
        t = 0.0
        while t < 35:
            evs += r.update(t, skeleton(), True, False)
            t += 0.5
        self.assertEqual(evs, ["turn_due"])
        turned = skeleton(face=(0.9, 0, 0.9, 0, 0.9), sh=(210, 190), hip=(330, 225), ankle=(500, 170))
        while t < 45:
            evs += r.update(t, turned, True, False)
            t += 0.5
        self.assertEqual(evs[-1], "repositioned")
        self.assertLess(r.minutes_in_position(t), 0.2)

    def test_small_wriggle_is_not_reposition(self):
        cfg = make_config("demo")
        r = RepositionTracker(cfg)
        evs = []
        for i in range(40):
            evs += r.update(i * 0.5, skeleton(wrist=(260 + (i % 3) * 4, 150)), True, False)
        self.assertNotIn("repositioned", evs)


class ForecastTests(unittest.TestCase):
    def test_rising_rate_gets_eta_and_explanation(self):
        cfg = make_config("demo")
        b, r = Baseline(cfg.baseline_s), RiskEngine(cfg)
        t = 0.0
        while t < 50:
            b.add(t, 14, True)
            r.update(t, 14, True, None, False, b)
            t += 0.5
        rr = 14.0
        while t < 75:
            rr += 0.1                          # +12 breaths/min per minute
            r.update(t, rr, True, None, False, b)
            t += 0.5
        self.assertIsNotNone(r.forecast)
        self.assertEqual(r.forecast["level"], "red")
        self.assertGreater(r.forecast["eta_s"], 0)
        self.assertTrue(any(p["key"] == "deviation" for p in r.explain()))

    def test_stable_has_no_forecast(self):
        cfg = make_config("demo")
        b, r = Baseline(cfg.baseline_s), RiskEngine(cfg)
        for i in range(200):
            b.add(i * 0.5, 14, True)
            r.update(i * 0.5, 14, True, None, False, b)
        self.assertIsNone(r.forecast)
        self.assertEqual(r.explain(), [])


class SitToStandTests(unittest.TestCase):
    def test_rise_time_measured(self):
        cfg = make_config("demo")
        p = PostureTracker(cfg)
        t = 0.0
        sit = person((210, 100), (215, 210), (330, 215))
        while t < 3:
            p.update(t, sit, SHAPE)
            t += 0.1
        for k in range(20):                       # 2 s rise: hips go from y=210 to y=150
            y = 210 - 60 * (k + 1) / 20
            p.update(t, person((212, y - 110), (213, y), (214, y + 120)), SHAPE)
            t += 0.1
        stand = person((212, 40), (213, 150), (214, 270))
        while t < 7:
            p.update(t, stand, SHAPE)
            t += 0.1
        self.assertEqual(p.state, "standing")
        self.assertIsNotNone(p.sit_to_stand_s)
        self.assertAlmostEqual(p.sit_to_stand_s, 1.6, delta=0.5)


class AlertExtrasTests(unittest.TestCase):
    def test_escalation_once(self):
        cfg = make_config("demo", escalate_after_s=30)
        with tempfile.TemporaryDirectory() as d:
            am = AlertManager(cfg, EventLog(d))
            a = am.raise_alert("b1", "B1", "fall", "critical", "x")
            self.assertEqual(am.tick(a.ts + 10), [])
            self.assertEqual(am.tick(a.ts + 31), [a])
            self.assertEqual(am.tick(a.ts + 60), [])
            self.assertTrue(a.escalated)

    def test_fhir_bundle(self):
        entries = [{"ts": time.time(), "bed_id": "bed1", "kind": "vitals", "rr": 16.0, "hr": None},
                   {"ts": time.time(), "bed_id": "bed2", "kind": "vitals", "rr": 20.0, "hr": 80.0}]
        b = bundle(entries, "bed1")
        self.assertEqual(b["resourceType"], "Bundle")
        self.assertEqual(b["total"], 1)
        obs = b["entry"][0]["resource"]
        self.assertEqual(obs["code"]["coding"][0]["code"], "9279-1")
        self.assertEqual(obs["valueQuantity"]["value"], 16.0)
        self.assertEqual(bundle(entries)["total"], 3)

    def test_validation_stats(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "validation.csv"), "w", encoding="utf-8") as f:
                f.write("time,bed,duration_s,manual_rr,system_rr,error\n")
                for m, s in [(14, 15), (16, 15.5), (20, 21), (12, 12)]:
                    f.write(f"x,bed1,60,{m},{s},{s - m}\n")
            v = validation_stats(d)
            self.assertEqual(v["n"], 4)
            self.assertEqual(v["within_2_pct"], 100)
            self.assertAlmostEqual(v["mae"], 0.62, places=2)

    def test_family_hides_numbers(self):
        f = family_status({"name": "B1", "level": "red", "rr": 30, "alerts": [1], "posture": "lying"})
        shown = {k: v for k, v in f.items() if k != "updated"}   # the timestamp can contain "30" by chance
        self.assertNotIn("30", str(shown))
        self.assertIn("notified", f["status"])


class ServerTests(unittest.TestCase):
    """Login, cloud mirror and relay round-trip using simulated beds only (no camera needed)."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from cloud import RemoteWard
        from monitor.server import Ward, create_app
        from monitor.simulate import PROFILES, SimBed

        self.tmp = tempfile.TemporaryDirectory()
        cfg = make_config("demo", data_dir=self.tmp.name, access_code="1234", relay_key="k")
        log = EventLog(self.tmp.name)
        self.alerts = AlertManager(cfg, log)
        sim = SimBed("sim2", "Bed 2", cfg, PROFILES[0], 0)
        sim.step(time.time())
        self.edge = Ward(cfg, {"sim2": sim}, self.alerts, log, 8000)
        self.cloud = RemoteWard(cfg, 8080)
        self.client = TestClient(create_app(self.cloud))

    def tearDown(self):
        self.tmp.cleanup()

    def test_login_required(self):
        r = self.client.get("/api/state")
        self.assertEqual(r.status_code, 401)
        r = self.client.post("/login", content="code=wrong&next=/", headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(r.status_code, 401)
        r = self.client.post("/login", content="code=1234&next=/", follow_redirects=False,
                             headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.client.get("/api/state").status_code, 200)

    def test_relay_round_trip(self):
        from monitor.relay import execute
        a = self.alerts.raise_alert("sim2", "Bed 2", "fall", "critical", "x")
        payload = {"state": self.edge.state(), "frames": {},
                   "extras": {"handover": self.edge.handover(8), "analytics": self.edge.analytics(),
                              "fhir": self.edge.fhir()}}
        self.cloud.ingest(payload)
        st = self.cloud.state()
        self.assertTrue(st["edge_online"])
        self.assertEqual(st["beds"][0]["name"], "Bed 2")
        self.cloud.ack(a.id)                         # a nurse acknowledges on the cloud page...
        for cmd in self.cloud.ingest(payload):       # ...the next push carries it back to the laptop
            execute(self.edge, cmd)
        self.assertTrue(a.acked)

    def test_family_link_is_public_but_scoped(self):
        self.cloud.ingest({"state": self.edge.state()})
        tok = self.cloud.family_token("sim2")
        r = self.client.get(f"/api/family/{tok}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/family/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
