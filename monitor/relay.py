"""Edge -> cloud relay. The AI runs on the bedside laptop; only numbers, events and a
skeleton-only picture go to the cloud. The cloud answers with what nurses did there
(acknowledge, visit, recalibrate...) and the laptop applies it.
"""
import base64
import json
import threading
import time
import urllib.request


class RelayClient:
    def __init__(self, ward, url, key, interval=0.5):
        self.ward, self.url, self.key, self.interval = ward, url.rstrip("/") + "/ingest", key, interval
        self.online = False
        self.last_error = None
        self._last_extras = 0.0
        self._last_err_print = 0.0

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="relay").start()

    def _run(self):
        while True:
            t0 = time.time()
            try:
                self._push()
                self.online = True
            except Exception as e:
                self.online, self.last_error = False, str(e)
                if time.time() - self._last_err_print > 60:
                    self._last_err_print = time.time()
                    print(f"[relay] cloud not reachable: {e}")
            time.sleep(max(0.05, self.interval - (time.time() - t0)))

    def _push(self):
        w = self.ward
        payload = {"state": w.state(), "frames": {}}
        for bed_id, bed in list(w.beds.items()):
            jpg = getattr(bed, "cloud_jpeg", None)
            if jpg:
                payload["frames"][bed_id] = base64.b64encode(jpg).decode("ascii")
        if time.time() - self._last_extras > 10:
            self._last_extras = time.time()
            payload["extras"] = {
                "handover": w.handover(8), "analytics": w.analytics(), "fhir": w.fhir(None, 300),
                "sleep": {b: w.sleep_report(b) for b, bed in list(w.beds.items()) if hasattr(bed, "capture")},
            }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "X-Relay-Key": self.key})
        with urllib.request.urlopen(req, timeout=5) as r:
            commands = json.loads(r.read().decode("utf-8")).get("commands", [])
        for cmd in commands:
            try:
                execute(w, cmd)
            except Exception as e:
                print(f"[relay] command {cmd.get('op')} failed: {e}")


def execute(ward, cmd):
    """Apply an action a nurse took on the cloud dashboard."""
    op, bed_id, a = cmd.get("op"), cmd.get("bed_id"), cmd.get("args") or {}
    if op == "ack":
        ward.ack(int(a["alert_id"]))
    elif op == "feedback":
        ward.feedback(int(a["alert_id"]), a["value"])
    elif op == "say":
        ward.say(bed_id, a.get("preset"), a.get("text"))
    elif op == "say_audio":
        ward.say_audio(bed_id, base64.b64decode(a["audio"]), a.get("mime"))
    elif op == "note":
        ward.add_note(bed_id, a["note"])
    elif op == "add_camera":
        ward.add_camera(a["name"], a["source"])
    elif op == "remove_bed":
        ward.remove_bed(bed_id)
    elif bed_id in ward.beds:
        bed = ward.beds[bed_id]
        if op == "visit":
            bed.mark_visit()
        elif op == "privacy" and a.get("mode") != "raw":   # never switch to real video from the internet
            bed.set_privacy(a["mode"])
        elif op == "recalibrate":
            bed.recalibrate()
        elif op == "zone":
            bed.set_zone(a.get("zone"))
        elif op == "validate":
            bed.validate(a["duration_s"], a["manual_rr"])
