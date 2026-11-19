"""web server: dashboard, nurse page, family view, analytics, fhir, mjpeg video.

Ward = local, owns the cameras. RemoteWard (cloud.py) just mirrors what the laptop pushes.
"""
import asyncio
import hashlib
import io
import json
import os
import re
import secrets
import socket
import threading
import time
from html import escape
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse, Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import i18n
from .analytics import alert_stats, clinical_report, sleep_report, validation_stats
from .assistant import WardAssistant
from .fhir import bundle as fhir_bundle
from .handover import ward_summary

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
RANK = {"red": 3, "yellow": 2, "green": 1, "grey": 0}


def sort_beds(snaps):
    return sorted(snaps, key=lambda s: (-RANK.get(s.get("level"), 0), -(s.get("score") or 0),
                                        s.get("simulated", False)))


def ai_context(state, handover_text):
    """ward summary for the llm, no images"""
    now = state["t"]
    keys = ("name", "simulated", "level", "score", "explain", "rr", "trend", "slope", "hr",
            "posture", "apnea_s", "apnea_episodes", "sleep", "position_min", "bed_exits", "near_falls",
            "night", "hand_label", "staff_min", "facetest", "chair_count")
    beds = []
    for s in state["beds"]:
        b = {k: s.get(k) for k in keys if s.get(k) not in (None, [], {})}
        if s.get("notes"):
            b["nurse_notes"] = [{"time": time.strftime("%H:%M", time.localtime(n["ts"])),
                                 "summary": n.get("summary"), "urgency": n.get("urgency"),
                                 "vitals": n.get("vitals")} for n in s["notes"]]
        if (s.get("baseline") or {}).get("ready"):
            b["own_normal_rr"] = s["baseline"]["rr"]
        if s.get("last_visit"):
            b["minutes_since_nurse_visit"] = round((now - s["last_visit"]) / 60)
        beds.append(b)
    alerts = [{"time": time.strftime("%H:%M", time.localtime(a["ts"])), "bed": a["bed_name"],
               "message": a["message"], "severity": a["severity"],
               "status": "active" if a["active"] else "acknowledged" if a["acked"] else "resolved",
               "escalated": a.get("escalated", False)} for a in state["alerts"]]
    return json.dumps({"now": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
                       "beds_by_risk": beds, "recent_alerts": alerts,
                       "rule_based_summary": handover_text}, ensure_ascii=False, indent=1)


def family_status(snap):
    """family view: no numbers, no video"""
    if snap.get("level") == "red" or snap.get("alerts"):
        status = "Staff notified"
    elif snap.get("level") == "yellow":
        status = "Being watched"
    else:
        status = "Stable"
    activity = "sleeping" if snap.get("sleep") == "asleep" else {
        "lying": "resting", "reclined": "resting", "sitting": "sitting",
        "standing": "standing", "absent": "not in room"}.get(snap.get("posture"), "-")
    return {"name": snap.get("name"), "status": status, "activity": activity,
            "level": "red" if snap.get("level") == "red" else "ok",
            "last_visit": snap.get("last_visit"), "updated": time.time()}


class Ward:
    def __init__(self, cfg, beds, alerts, log, port, camera_factory=None):
        self.cfg, self.beds, self.alerts, self.log, self.port = cfg, beds, alerts, log, port
        self.camera_factory = camera_factory
        self.assistant = WardAssistant(cfg)
        self.family_tokens = {}
        self.camera_status = {}
        self._state_json = (0.0, "")      # (built at, text): shared by every open dashboard
        self.ws_clients_closed = 0

    def bed(self, bed_id):
        b = self.beds.get(bed_id)
        if b is None:
            raise HTTPException(404, "unknown bed")
        return b

    def state(self):
        snaps = [b.snap for b in list(self.beds.values()) if b.snap]
        return {"t": time.time(), "beds": sort_beds(snaps), "alerts": self.alerts.recent(30),
                "cameras": self.camera_status, "edge_online": True}

    def state_json(self, max_age=0.3):
        """cached json, rebuilt max every 0.3s"""
        built, text = self._state_json
        now = time.time()
        if now - built > max_age or not text:
            text = json.dumps(self.state(), ensure_ascii=False, default=str)
            self._state_json = (now, text)
        return text

    def ack(self, alert_id):
        return self.alerts.ack(alert_id)

    def handover(self, hours):
        return ward_summary(list(self.beds.values()), self.log, hours)

    def analytics(self):
        return {"alerts": alert_stats(self.alerts.all()), "validation": validation_stats(self.cfg.data_dir),
                "beds": len(self.beds), "cameras": sum(1 for b in self.beds.values() if hasattr(b, "capture"))}

    def fhir(self, bed_id=None, count=100):
        return fhir_bundle(self.log.query(kinds={"vitals"}), bed_id, count)

    def ai_data(self):
        return ai_context(self.state(), self.handover(8))

    def family_token(self, bed_id):
        self.bed(bed_id)
        for tok, b in self.family_tokens.items():
            if b == bed_id:
                return tok
        tok = secrets.token_urlsafe(9)
        self.family_tokens[tok] = bed_id
        return tok

    def family(self, token):
        bed_id = self.family_tokens.get(token)
        if bed_id is None or bed_id not in self.beds:
            raise HTTPException(404, "link expired")
        return family_status(self.beds[bed_id].snap)

    def add_camera(self, name, source):
        """add a camera in the background, opening a stream can take a while"""
        if self.camera_factory is None:
            raise HTTPException(400, "cameras cannot be added here")
        bed_id = f"cam{len(self.beds) + 1}_{secrets.token_hex(2)}"
        self.camera_status[bed_id] = {"name": name, "state": "connecting"}

        def start():
            try:
                bed = self.camera_factory(bed_id, name, source)
                bed.start()
                self.beds[bed_id] = bed
                self.camera_status[bed_id] = {"name": name, "state": "ok"}
                self.log.add(bed_id, "camera_added", source_kind=bed.snap.get("source_kind"))
            except Exception as e:
                self.camera_status[bed_id] = {"name": name, "state": "error", "error": str(e)[:200]}

        threading.Thread(target=start, daemon=True).start()
        return bed_id

    def remove_bed(self, bed_id):
        bed = self.beds.pop(bed_id, None)
        self.camera_status.pop(bed_id, None)
        if bed is not None and hasattr(bed, "stop"):
            bed.stop()

    def camera_bed(self, bed_id):
        bed = self.bed(bed_id)
        if not hasattr(bed, "say_to_patient"):
            raise HTTPException(400, "simulated bed has no bedside screen")
        return bed

    def feedback(self, alert_id, value):
        return self.alerts.feedback(alert_id, value)

    def sleep_report(self, bed_id):
        self.bed(bed_id)
        return sleep_report(self.log, bed_id)

    def say(self, bed_id, preset=None, text=None):
        bed = self.camera_bed(bed_id)
        if preset:
            return bed.nurse_message(preset=preset)
        audio = synthesize(text)                       # spoken if online, otherwise shown on screen only
        return bed.nurse_message(text=text, audio=audio, mime="audio/mpeg" if audio else None)

    def say_audio(self, bed_id, audio, mime):
        return self.camera_bed(bed_id).nurse_message(text="voice message from nurse", audio=audio, mime=mime)

    def bedside(self, bed_id, after):
        return self.camera_bed(bed_id).bedside(after)

    def patient_button(self, bed_id):
        self.camera_bed(bed_id).patient_button()

    def message_audio(self, bed_id, msg_id):
        a = self.camera_bed(bed_id).audio.get(msg_id)
        if a is None:
            raise HTTPException(404, "no audio")
        return a

    def add_note(self, bed_id, note):
        return self.camera_bed(bed_id).add_note(note)


def synthesize(text):
    """tts via edge-tts (needs internet)"""
    try:
        import edge_tts
    except ImportError:
        return None

    async def run():
        out = bytearray()
        async for chunk in edge_tts.Communicate(text, "en-US-AriaNeural").stream():
            if chunk["type"] == "audio":
                out.extend(chunk["data"])
        return bytes(out)

    try:
        return asyncio.run(run()) or None
    except Exception as e:
        print(f"[tts] offline, message will be shown as text only: {e}")
        return None


def lan_urls(port):
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))     # no packet is sent; picks the LAN interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ips.discard("127.0.0.1")
    return [f"http://{ip}:{port}" for ip in sorted(ips)]


class Privacy(BaseModel):
    mode: str


class Magnify(BaseModel):
    mode: str | None = None


class AlarmPause(BaseModel):
    seconds: int = 120


class Zone(BaseModel):
    x1: float = 0
    y1: float = 0
    x2: float = 1
    y2: float = 1
    clear: bool = False


class Validation(BaseModel):
    duration_s: float
    manual_rr: float


class Question(BaseModel):
    question: str


class Camera(BaseModel):
    name: str
    source: str


class CameraSettings(BaseModel):
    stream: str | None = None       # main | sub
    rotate: int | None = None       # 0 / 90 / 180 / 270
    mirror: bool | None = None


class Ptz(BaseModel):
    pan: float = 0.0                # -1 left .. 1 right
    tilt: float = 0.0               # -1 down .. 1 up
    zoom: float = 0.0
    ms: int = 400


class Probe(BaseModel):
    ip: str


class Feedback(BaseModel):
    value: str


class Say(BaseModel):
    preset: str = None
    text: str = None


MAX_AUDIO = 6 * 1024 * 1024


async def read_audio(request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > MAX_AUDIO:
        raise HTTPException(413, "recording too long")
    return data, request.headers.get("content-type", "audio/webm").split(";")[0]


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>login</title>
<link rel="stylesheet" href="/static/style.css?v=2"></head>
<body style="display:grid;place-items:center;min-height:100vh">
<form method="post" action="/login" class="box" style="width:min(300px,100%)">
<div class="hd">Patient-Monitoring</div>
<div class="bd row"><input id="code" name="code" type="password" placeholder="access code" autocomplete="current-password" autofocus class="sp">
<input type="hidden" name="next" value="{next}"><button>login</button></div>{error}</form></body></html>"""


def create_app(ward):
    app = FastAPI(title="Silent Early Warning")
    app.mount("/static", StaticFiles(directory=WEB), name="static")
    code = ward.cfg.access_code
    token = hashlib.sha256(f"sew:{code}".encode()).hexdigest() if code else None
    open_paths = ("/login", "/static/", "/family/", "/api/family/", "/ingest")

    def authed(cookies):
        return token is None or secrets.compare_digest(cookies.get("sew_auth", ""), token)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if token and not path.startswith(open_paths) and not authed(request.cookies):
            if path.startswith(("/api/", "/video/", "/fhir/", "/qr.png")):
                return JSONResponse({"detail": "login required"}, status_code=401)
            return RedirectResponse(f"/login?next={path}", status_code=303)
        return await call_next(request)

    @app.get("/login")
    def login_page(next: str = "/"):
        nxt = next if next.startswith("/") and not next.startswith("//") else "/"
        return HTMLResponse(LOGIN_PAGE.format(next=escape(nxt), error=""))

    @app.post("/login")
    async def login(request: Request):
        form = {k: v[0] for k, v in parse_qs((await request.body()).decode("utf-8", "replace")).items()}
        nxt = str(form.get("next") or "/")
        nxt = nxt if nxt.startswith("/") and not nxt.startswith("//") else "/"
        if token and secrets.compare_digest(str(form.get("code", "")), code):
            r = RedirectResponse(nxt, status_code=303)
            r.set_cookie("sew_auth", token, httponly=True, samesite="lax", max_age=7 * 86400,
                         secure=request.url.scheme == "https")
            return r
        return HTMLResponse(LOGIN_PAGE.format(next=escape(nxt), error='<div class="bd bad">wrong code</div>'),
                            status_code=401)

    @app.get("/")
    def dashboard():
        return FileResponse(os.path.join(WEB, "dashboard.html"))

    @app.get("/nurse")
    def nurse():
        return FileResponse(os.path.join(WEB, "nurse.html"))

    @app.get("/analytics")
    def analytics_page():
        return FileResponse(os.path.join(WEB, "analytics.html"))

    @app.post("/api/alarm_pause")
    def alarm_pause(body: AlarmPause, request: Request):
        """mute alarm sound for a bit, goes in the audit log"""
        ward.log.add("ward", "alarm_audio_pause", seconds=max(0, min(300, body.seconds)),
                     client=request.client.host if request.client else None)
        return {"ok": True}

    @app.get("/api/audit/verify")
    def audit_verify():
        from .alerts import verify_chain
        return verify_chain(ward.log.path)

    @app.get("/evidence")
    def evidence_page():
        return FileResponse(os.path.join(WEB, "evidence.html"))

    @app.get("/family/{tok}")
    def family_page(tok: str):
        ward.family(tok)
        return FileResponse(os.path.join(WEB, "family.html"))

    @app.get("/api/family/{tok}")
    def family_api(tok: str):
        return ward.family(tok)

    @app.get("/api/state")
    def state():
        return ward.state()

    @app.get("/api/info")
    def info(request: Request):
        urls = lan_urls(ward.port) if not ward.cfg.relay_key else []
        base = str(request.base_url).rstrip("/")
        if not urls or base.startswith("https://"):
            urls = [base] + [u for u in urls if u != base]
        return {"urls": urls, "nurse_urls": [u + "/nurse" for u in urls],
                "ntfy_topic": ward.cfg.ntfy_topic or None, "ntfy_server": ward.cfg.ntfy_server,
                "profile": ward.cfg.profile, "base": base}

    @app.get("/qr.png")
    def qr(url: str):
        try:
            import segno
        except ImportError:
            raise HTTPException(404, "pip install segno for QR codes")
        buf = io.BytesIO()
        segno.make(url, error="m").save(buf, kind="png", scale=6, border=2)
        return Response(buf.getvalue(), media_type="image/png")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        if not authed(sock.cookies):
            await sock.close(code=4401)
            return
        await sock.accept()
        # send() doesn't fail when a client just disappears (phone locked, laptop asleep), so we
        # kept sending to dead sockets forever. after a day the ai thread was down to 3fps.
        # watch the receive side for the disconnect instead
        gone = asyncio.Event()

        async def watch():
            try:
                while (await sock.receive()).get("type") != "websocket.disconnect":
                    pass
            except Exception:
                pass
            gone.set()

        watcher = asyncio.create_task(watch())
        try:
            while not gone.is_set():
                await sock.send_text(ward.state_json())
                try:
                    await asyncio.wait_for(gone.wait(), 0.4)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            pass
        finally:
            watcher.cancel()
            ward.ws_clients_closed += 1

    @app.get("/video/{bed_id}")
    async def video(bed_id: str):
        bed = ward.bed(bed_id)
        if not hasattr(bed, "jpeg"):
            raise HTTPException(404, "simulated bed has no camera")

        async def frames():
            last = None
            while bed_id in ward.beds:
                jpg = bed.jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(1.0 / ward.cfg.stream_fps)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    # ---------- camera panel ----------
    def cam_bed(bed_id):
        bed = ward.bed(bed_id)
        if not hasattr(bed, "camera_info"):
            raise HTTPException(404, "simulated bed has no camera")
        return bed

    @app.get("/api/beds/{bed_id}/camera")
    def camera_get(bed_id: str, probe: bool = False):
        return cam_bed(bed_id).camera_info(probe=probe)

    @app.post("/api/beds/{bed_id}/camera")
    def camera_set(bed_id: str, body: CameraSettings):
        return cam_bed(bed_id).set_camera(body.stream, body.rotate, body.mirror)

    @app.post("/api/beds/{bed_id}/camera/ptz")
    def camera_ptz(bed_id: str, body: Ptz):
        if not cam_bed(bed_id).ptz(body.pan, body.tilt, body.zoom, body.ms):
            raise HTTPException(400, "this camera did not accept pan / tilt")
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/camera/reconnect")
    def camera_reconnect(bed_id: str):
        cam_bed(bed_id).reconnect_camera()
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/face_test")
    def face_test(bed_id: str):
        """FAST face test: the speaker asks for a smile, then tightly closed eyes."""
        cam_bed(bed_id).start_face_test()
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/chair_stand")
    def chair_stand(bed_id: str):
        """30-second chair stand test."""
        cam_bed(bed_id).start_chair_stand()
        return {"ok": True}

    @app.get("/api/replays/{alert_id}")
    def replay(alert_id: int):
        """Skeleton replay around an alert (coordinates only - no image is ever stored)."""
        for b in list(ward.beds.values()):
            rec = getattr(b, "recorder", None)
            r = rec.get(alert_id) if rec is not None else None
            if r is not None:
                return r
        raise HTTPException(404, "no replay for this alert")

    @app.get("/api/beds/{bed_id}/snapshot.jpg")
    def camera_snapshot(bed_id: str):
        """The current dashboard view (same privacy mode as on screen) as a downloadable picture."""
        jpg = cam_bed(bed_id).jpeg
        if jpg is None:
            raise HTTPException(404, "no picture yet")
        name = f"snapshot_{bed_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        return Response(jpg, media_type="image/jpeg", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/api/cameras/probe")
    def camera_probe(body: Probe):
        """Ask an IP camera over ONVIF what it is and which streams it offers (no login needed on many)."""
        from .onvif import cached_info, device_info, host_of, profiles
        ip = body.ip.strip()
        if not re.fullmatch(r"[\w.\-]+", ip):
            raise HTTPException(400, "bad address")
        # a camera that is already streaming here is not asked again: it would drop its stream
        live = any(getattr(getattr(b, "capture", None), "source", "").startswith(("onvif://", "rtsp://"))
                   and host_of(b.capture.source) == ip for b in ward.beds.values())
        if live and cached_info(ip):
            info = cached_info(ip)
            return {"ip": ip, "device": info.get("device", {}), "profiles": info.get("profiles", []),
                    "source": f"onvif://{ip}", "already_added": True}
        profs = profiles(ip, query=True)
        if not profs:
            raise HTTPException(404, "no ONVIF camera answered at this address")
        return {"ip": ip, "device": device_info(ip), "profiles": profs, "source": f"onvif://{ip}"}

    @app.post("/api/alerts/{alert_id}/ack")
    def ack(alert_id: int):
        return {"ok": ward.ack(alert_id)}

    @app.post("/api/beds/{bed_id}/visit")
    def visit(bed_id: str):
        ward.bed(bed_id).mark_visit()
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/privacy")
    def privacy(bed_id: str, body: Privacy):
        ward.bed(bed_id).set_privacy(body.mode)
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/magnify")
    def magnify(bed_id: str, body: Magnify):
        """Eulerian magnification view: 'breath' (x18 motion), 'pulse' (x80 colour) or null (off)."""
        b = ward.bed(bed_id)
        if not hasattr(b, "set_magnify"):
            raise HTTPException(400, "not a camera bed")
        b.set_magnify(body.mode)
        return {"ok": True, "mode": b.magnifier.mode}

    @app.post("/api/beds/{bed_id}/recalibrate")
    def recalibrate(bed_id: str):
        ward.bed(bed_id).recalibrate()
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/zone")
    def zone(bed_id: str, body: Zone):
        z = None if body.clear else (min(body.x1, body.x2), min(body.y1, body.y2),
                                     max(body.x1, body.x2), max(body.y1, body.y2))
        ward.bed(bed_id).set_zone(z)
        return {"ok": True, "zone": z}

    @app.post("/api/beds/{bed_id}/validate")
    def validate(bed_id: str, body: Validation):
        return ward.bed(bed_id).validate(body.duration_s, body.manual_rr)

    @app.get("/api/beds/{bed_id}/family_link")
    def family_link(bed_id: str, request: Request):
        base = str(request.base_url).rstrip("/")
        return {"url": f"{base}/family/{ward.family_token(bed_id)}"}

    @app.post("/api/cameras")
    def add_camera(body: Camera):
        if not body.source.strip():
            raise HTTPException(400, "source required")
        return {"id": ward.add_camera(body.name.strip() or "Bed", body.source.strip())}

    @app.delete("/api/beds/{bed_id}")
    def remove_bed(bed_id: str):
        ward.remove_bed(bed_id)
        return {"ok": True}

    @app.get("/api/handover", response_class=PlainTextResponse)
    def handover(hours: float = 8, ai: bool = False):
        if ai:
            return ward.assistant.handover(ward.ai_data())
        return ward.handover(hours)

    @app.post("/api/assistant")
    def assistant(body: Question):
        q = body.question.strip()[:1000]
        if not q:
            raise HTTPException(400, "empty question")
        return {"answer": ward.assistant.ask(q, ward.ai_data())}

    @app.get("/api/analytics")
    def analytics():
        return ward.analytics()

    @app.get("/fhir/Observation")
    def fhir(bed: str = None, count: int = 100):
        return JSONResponse(ward.fhir(bed, min(count, 1000)), media_type="application/fhir+json")

    @app.get("/api/i18n")
    def strings():
        return {"risk_part": i18n.RISK_PART, "alert_titles": i18n.alert_names()}

    # ---- alert feedback, sleep report ----
    @app.post("/api/alerts/{alert_id}/feedback")
    def feedback(alert_id: int, body: Feedback):
        return {"ok": ward.feedback(alert_id, body.value)}

    @app.get("/report/{bed_id}")
    def report_page(bed_id: str):
        return FileResponse(os.path.join(WEB, "report.html"))

    @app.get("/api/beds/{bed_id}/report")
    def report(bed_id: str, hours: float = 24):
        bed = ward.bed(bed_id)
        return clinical_report(ward.log, bed_id, bed.snap or {}, max(1.0, min(48.0, hours)))

    @app.get("/api/beds/{bed_id}/sleep_report")
    def sleep(bed_id: str):
        return ward.sleep_report(bed_id)

    # ---- talk to the patient ----
    @app.post("/api/beds/{bed_id}/say")
    def say(bed_id: str, body: Say):
        if not body.preset and not (body.text or "").strip():
            raise HTTPException(400, "preset or text required")
        try:
            msg = ward.say(bed_id, body.preset, (body.text or "").strip()[:300] or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "message": msg}

    @app.post("/api/beds/{bed_id}/say_audio")
    async def say_audio(bed_id: str, request: Request):
        audio, mime = await read_audio(request)
        return {"ok": True, "message": ward.say_audio(bed_id, audio, mime)}

    @app.get("/api/beds/{bed_id}/audio/{msg_id}")
    def message_audio(bed_id: str, msg_id: int):
        data, mime = ward.message_audio(bed_id, msg_id)
        return Response(data, media_type=mime)

    # ---- patient-facing bedside screen (runs on the bedside laptop) ----
    @app.get("/bedside/{bed_id}")
    def bedside_page(bed_id: str):
        ward.camera_bed(bed_id)
        return FileResponse(os.path.join(WEB, "bedside.html"))

    @app.get("/api/bedside/{bed_id}")
    def bedside(bed_id: str, after: int = 0):
        return ward.bedside(bed_id, after)

    @app.post("/api/bedside/{bed_id}/button")
    def bedside_button(bed_id: str):
        ward.patient_button(bed_id)
        return {"ok": True}

    # ---- voice: nurse notes and spoken questions  ----
    @app.post("/api/beds/{bed_id}/note_audio")
    async def note_audio(bed_id: str, request: Request):
        ward.camera_bed(bed_id)
        audio, mime = await read_audio(request)
        note = await asyncio.to_thread(ward.assistant.voice_note, audio, mime)
        if "error" in note:
            return JSONResponse(note, status_code=502)
        return {"ok": True, "note": ward.add_note(bed_id, note)}

    @app.post("/api/assistant/voice")
    async def assistant_voice(request: Request):
        audio, mime = await read_audio(request)
        data = ward.ai_data()
        return await asyncio.to_thread(ward.assistant.ask_voice, audio, mime, data)

    @app.get("/api/ai_status")
    def ai_status():
        a = ward.assistant
        return {"provider": a.provider_name, "vertex_project": ward.cfg.vertex_project or None,
                "model": ward.cfg.gemini_model if a.provider_name == "gemini" else ward.cfg.ai_model,
                "voice": a.provider_name == "gemini",
                # the dashboard hides the AI panel when no credentials are set (it showed an error)
                "ready": bool(ward.cfg.vertex_project or (a.provider_name == "claude" and (
                    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))))}

    return app
