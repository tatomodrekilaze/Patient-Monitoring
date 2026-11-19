"""Cloud side (Google Cloud Run): serves the dashboard, nurse phones, family view and the AI
assistant over HTTPS, mirroring what the bedside laptop pushes. No camera, no video, no torch.

    ACCESS_CODE=... RELAY_KEY=... python cloud.py
"""
import argparse
import base64
import os
import secrets
import time
from collections import deque

import uvicorn
from fastapi import HTTPException, Request

from monitor.config import make_config
from monitor.server import Ward, ai_context, create_app, sort_beds


class RemoteBed:
    """A bed that lives on the laptop. Actions are queued and sent back on the next push."""

    def __init__(self, ward, bed_id):
        self.ward, self.bed_id = ward, bed_id
        self.snap, self.jpeg = {}, None

    def _cmd(self, op, **args):
        self.ward.commands.append({"op": op, "bed_id": self.bed_id, "args": args})

    def mark_visit(self, auto=False):
        self._cmd("visit")
        self.snap["last_visit"] = time.time()

    def set_privacy(self, mode):
        self._cmd("privacy", mode=mode)

    def recalibrate(self):
        self._cmd("recalibrate")

    def set_zone(self, zone):
        self._cmd("zone", zone=zone)

    def validate(self, duration_s, manual_rr):
        self._cmd("validate", duration_s=duration_s, manual_rr=manual_rr)
        return {"manual_rr": round(manual_rr, 1), "system_rr": None, "error": None, "pass": None,
                "queued": True}


class RemoteWard(Ward):
    def __init__(self, cfg, port):
        super().__init__(cfg, {}, None, None, port)
        self.commands = deque(maxlen=200)
        self.alerts_list = []
        self.extras = {}
        self.cameras = {}
        self.last_push = 0.0
        self.acked_here = set()

    def ingest(self, payload):
        state = payload.get("state") or {}
        seen = set()
        for snap in state.get("beds", []):
            bed = self.beds.get(snap["id"]) or RemoteBed(self, snap["id"])
            bed.snap = snap
            self.beds[snap["id"]] = bed
            seen.add(snap["id"])
        for bed_id in list(self.beds):
            if bed_id not in seen:
                del self.beds[bed_id]
        for bed_id, b64 in (payload.get("frames") or {}).items():
            if bed_id in self.beds:
                self.beds[bed_id].jpeg = base64.b64decode(b64)
        self.alerts_list = state.get("alerts", [])
        self.cameras = state.get("cameras", {})
        self.extras.update(payload.get("extras") or {})
        self.last_push = time.time()
        commands = list(self.commands)
        self.commands.clear()
        return commands

    def state(self):
        alerts = []
        for a in self.alerts_list:
            if a["id"] in self.acked_here and a["active"]:   # show the ack before the laptop confirms it
                a = dict(a, acked=True, active=False)
            alerts.append(a)
        return {"t": time.time(), "beds": sort_beds([b.snap for b in self.beds.values() if b.snap]),
                "alerts": alerts, "cameras": self.cameras, "edge_online": time.time() - self.last_push < 5,
                "cloud": True}

    def ack(self, alert_id):
        self.acked_here.add(alert_id)
        self.commands.append({"op": "ack", "bed_id": None, "args": {"alert_id": alert_id}})
        return True

    def handover(self, hours):
        return self.extras.get("handover", "No data yet.")

    def analytics(self):
        return self.extras.get("analytics", {})

    def fhir(self, bed_id=None, count=100):
        b = self.extras.get("fhir") or {"resourceType": "Bundle", "type": "searchset", "entry": []}
        entries = [e for e in b.get("entry", [])
                   if not bed_id or e["resource"]["subject"]["reference"] == f"Patient/{bed_id}"][-count:]
        return {**b, "total": len(entries), "entry": entries}

    def ai_data(self):
        return ai_context(self.state(), self.handover(8))

    def camera_bed(self, bed_id):
        bed = self.bed(bed_id)
        if bed.snap.get("simulated"):
            raise HTTPException(400, "simulated bed has no bedside screen")
        return bed

    def feedback(self, alert_id, value):
        self.acked_here.add(alert_id)
        self.commands.append({"op": "feedback", "bed_id": None, "args": {"alert_id": alert_id, "value": value}})
        return True

    def sleep_report(self, bed_id):
        self.bed(bed_id)
        return (self.extras.get("sleep") or {}).get(bed_id, {"bed_id": bed_id, "samples": 0})

    def say(self, bed_id, preset=None, text=None):
        self.camera_bed(bed_id)
        self.commands.append({"op": "say", "bed_id": bed_id, "args": {"preset": preset, "text": text}})
        return {"queued": True}

    def say_audio(self, bed_id, audio, mime):
        self.camera_bed(bed_id)
        self.commands.append({"op": "say_audio", "bed_id": bed_id,
                              "args": {"audio": base64.b64encode(audio).decode("ascii"), "mime": mime}})
        return {"queued": True}

    def add_note(self, bed_id, note):
        note = dict(note, ts=time.time())
        self.commands.append({"op": "note", "bed_id": bed_id, "args": {"note": note}})
        return note

    def bedside(self, bed_id, after):
        raise HTTPException(404, "the bedside screen runs on the bedside laptop")

    def patient_button(self, bed_id):
        raise HTTPException(404, "the bedside screen runs on the bedside laptop")

    def message_audio(self, bed_id, msg_id):
        raise HTTPException(404, "no audio in the cloud")

    def add_camera(self, name, source):
        self.commands.append({"op": "add_camera", "bed_id": None, "args": {"name": name, "source": source}})
        return "queued"

    def remove_bed(self, bed_id):
        self.commands.append({"op": "remove_bed", "bed_id": bed_id, "args": {}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    ap.add_argument("--access-code", default=os.environ.get("ACCESS_CODE", ""))
    ap.add_argument("--relay-key", default=os.environ.get("RELAY_KEY", ""))
    ap.add_argument("--vertex-project", default=os.environ.get("VERTEX_PROJECT", ""))
    ap.add_argument("--vertex-region", default=os.environ.get("VERTEX_REGION", "global"))
    a = ap.parse_args()
    if not a.relay_key or not a.access_code:
        raise SystemExit("Set ACCESS_CODE (dashboard login) and RELAY_KEY (laptop push secret).")

    cfg = make_config("demo", access_code=a.access_code, relay_key=a.relay_key,
                      vertex_project=a.vertex_project, vertex_region=a.vertex_region,
                      ai_provider=os.environ.get("AI_PROVIDER", "auto"),
                      gemini_model=os.environ.get("GEMINI_MODEL") or None)
    ward = RemoteWard(cfg, a.port)
    app = create_app(ward)

    @app.post("/ingest")
    async def ingest(request: Request):
        if not secrets.compare_digest(request.headers.get("x-relay-key", ""), a.relay_key):
            raise HTTPException(403, "bad relay key")
        return {"commands": ward.ingest(await request.json())}

    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning", proxy_headers=True,
                forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
