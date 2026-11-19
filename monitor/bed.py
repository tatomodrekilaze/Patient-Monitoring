"""One bed = one camera, one thread.

capture -> pose/tracking -> breathing, hr -> posture, falls, hand signs -> baseline/risk
-> alerts -> rendered frame + snapshot for the dashboard
"""
import csv
import json
import os
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np

from . import i18n
from .capture import Capture
from .events.activity import SleepTracker, VisitDetector
from .events.face import FaceAnalyzer, head_pose
from .events.hands import REQUESTS, HandSignals
from .events.mobility import ChairStandTest
from .events.neuro import FaceTest
from .events.posture import IN_BED, PostureTracker, RoomFallWatch
from .events.reposition import RepositionTracker
from .events.risk import LEVELS, Baseline, RiskEngine
from .pose import PatientSelector, PoseTracker, chest_box, face_box, torso, torso_view
from .magnify import Magnifier
from .privacy import MODES, render
from .scene import SceneDetector
from .replay import Recorder
from .smooth import KeypointSmoother
from .signals.breathing import BreathingEstimator
from .signals.rppg import HeartRateEstimator

POSTURE_ALERTS = {
    "fall": "critical",
    "floor_long": "critical",
    "bed_edge": "warning",
    "bed_exit": "warning",
    "absent": "warning",
    "help": "warning",
}

HAND_ALERTS = {
    "req_water": "warning", "req_toilet": "warning", "req_pain": "warning", "req_nurse": "warning",
    "help": "warning", "feel_worse": "warning",
}

NURSE_PRESETS = i18n.NURSE_PRESETS


def enhance_low_light(frame):
    """brighten dark frames (CLAHE + gamma), pose falls apart otherwise"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(out, lut)


class Settings:
    """per-bed settings saved to disk (bed zone)"""

    def __init__(self, data_dir):
        self.path = os.path.join(data_dir, "settings.json")
        try:
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    def get(self, bed_id, key, default=None):
        return self.data.get(bed_id, {}).get(key, default)

    def set(self, bed_id, key, value):
        self.data.setdefault(bed_id, {})[key] = value
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


class BedMonitor:
    def __init__(self, bed_id, name, cfg, alerts, log, settings, source=None, record_raw=""):
        self.bed_id, self.name, self.cfg = bed_id, name, cfg
        self.alerts, self.log, self.settings = alerts, log, settings
        self.source = str(cfg.source if source is None else source)
        self.capture = Capture(self.source, cfg.width, cfg.height, cfg.fps, cfg.loop_video, record_raw)
        threads = max(1, cfg.torch_threads // max(1, cfg.camera_count))
        self.pose = PoseTracker(cfg.pose_model, cfg.imgsz, cfg.pose_conf, threads)
        self.breath = BreathingEstimator(cfg)
        self.hrest = HeartRateEstimator(cfg)
        self.posture = PostureTracker(cfg)
        self.baseline = Baseline(cfg.baseline_s)
        self.risk = RiskEngine(cfg)
        self.sleep = SleepTracker(cfg)
        self.visits = VisitDetector(cfg)
        self.reposition = RepositionTracker(cfg)
        self.hands = HandSignals(cfg)
        self.room_falls = RoomFallWatch(cfg)  # other people falling
        self.scene = SceneDetector(cfg, cfg.scene_model) if cfg.scene_ai else None   # only for the bed box
        self.face = FaceAnalyzer(cfg, cfg.face_model) if cfg.face_ai else None
        self.smoother = KeypointSmoother(thr=cfg.kp_conf) if cfg.kp_smoothing else None
        self.recorder = Recorder(cfg.data_dir)
        self.facetest = FaceTest(cfg)
        self.chair = ChairStandTest()
        self.magnifier = Magnifier()  # display only
        self._now = 0.0
        self.messages = deque(maxlen=50)    # to the bedside screen
        self.audio = {}                     # message id -> recorded nurse audio
        self._msg_id = 0
        self.notes = deque(maxlen=50)       # nurse voice notes
        self.staff_seconds = 0.0
        self.rounding_raised = False
        self.privacy = cfg.privacy_default
        self.cloud_jpeg = None  # skeleton only, real frames never leave the box
        self.night = False
        self.connected = False
        self.apnea_episodes = 0
        self.monitor_start = time.time()
        zone = settings.get(bed_id, "zone")
        if zone:
            self.posture.set_zone(zone)
        self._cam_cache = None
        cam = settings.get(bed_id, "camera") or {}
        if hasattr(self.capture, "rotate"):
            self.capture.rotate = int(cam.get("rotate", 0))
            self.capture.mirror = bool(cam.get("mirror", False))
            if cam.get("stream") in ("main", "sub") and self.capture.source.startswith("onvif://"):
                want = _onvif_base(self.capture.source) + "/" + cam["stream"]
                if want != self.capture.source:
                    self.capture.switch(want)
        self.jpeg = None
        self.snap = {"id": bed_id, "name": name, "level": "grey", "simulated": False}
        self.last_visit = None
        self.bed_exits = 0
        self.running = False
        self._recal_request = None
        self._last_frame_wall = time.time()
        self._offline = False
        self._reset_tracking()

    def _reset_tracking(self):
        self.selector = PatientSelector()
        self.chest_s = None
        self._head_box = None
        self._zone_source = "learned" if self.posture.zone_auto else ("manual" if self.posture.bed_zone else None)
        self.track_hist = deque()
        self.speed = 0.0
        self.moving_until = 0.0
        self.rr_valid_t = 0.0
        self.rr_samples = deque(maxlen=2400)
        self.apnea_active = False
        self.apnea_s = 0.0
        self.shape = (self.cfg.height, self.cfg.width)
        self.fps = 0.0
        self._proc_times = deque()
        self.delay = 0.0
        self._flow_wins = {}
        self.breath_view = "no_patient"
        self._view_ok_since = None
        self.measurable = False
        self._good_breath_t = None
        self._rr_cond = None  # (kind, since)
        self._last_t = None
        self._last_est = 0.0
        self._last_jpeg = 0.0
        self._last_cloud_jpeg = 0.0
        self._last_vitals = 0.0
        self._irregular_since = None

    # ---------------- thread ----------------
    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True, name=f"bed-{self.bed_id}").start()

    def stop(self):
        self.running = False
        self.capture.release()

    def _run(self):
        if self.cfg.hand_signals:
            try:
                self.hands._detector()  # warm up mediapipe
            except Exception as e:
                print(f"[hands] finger signals off: {e}")
                self.cfg.hand_signals = False
        seq, restarts = 0, self.capture.restarts
        while self.running:
            r = self.capture.read(seq)
            if r is None:
                self.camera_silent(time.time())
                continue
            self.camera_alive(time.time())
            seq, t, frame = r
            full = self.capture.full_frame(seq) if hasattr(self.capture, "full_frame") else None
            if self.capture.restarts != restarts:  # video looped
                restarts = self.capture.restarts
                self._recal_request = True
            if self._recal_request is not None:
                self._recalibrate(keep_zone=self._recal_request)
                self._recal_request = None
            try:
                self.process(t, frame, full)
            except Exception:
                traceback.print_exc()
                time.sleep(0.2)

    # ---------------- technical alarm ----------------
    def camera_silent(self, now):
        """no frames for a while -> technical alarm + blank the vitals (don't show stale numbers as live)"""
        gap = now - self._last_frame_wall
        if gap < self.cfg.camera_offline_s:
            return
        if not self._offline:
            self._offline = True
            self.log.add(self.bed_id, "camera_offline")
        self._alert("camera_offline", "warning", seconds=int(gap))
        # self.snap can be nearly empty if the camera never connected, so fill what the dashboard reads
        snap = dict(self.snap, present=False, rr=None, hr=None, moving=False, apnea_s=0.0, level="grey",
                    level_label=i18n.LEVEL["grey"], camera_offline=True, offline_s=int(gap),
                    fps=0.0, score=0, privacy=self.privacy)
        W, F = i18n.FEATURE_WAIT, i18n.FEATURE
        snap["features"] = [{"key": k, "name": F[k]["name"], "what": F[k]["what"], "state": "off",
                             "detail": W["camera"]} for k in F]
        self.snap = snap

    def camera_alive(self, now):
        self._last_frame_wall = now
        if self._offline:
            self._offline = False
            self.alerts.resolve(self.bed_id, "camera_offline")
            self.log.add(self.bed_id, "camera_back")

    # ---------------- pipeline ----------------
    def process(self, t, frame, full=None):
        cfg = self.cfg
        self._now = t
        self.connected = True
        shape = self.shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.night = float(gray.mean()) < cfg.night_thr
        seen = enhance_low_light(frame) if self.night else frame
        if self.night:
            gray = cv2.cvtColor(seen, cv2.COLOR_BGR2GRAY)
        people = self.pose.infer(seen)
        if self.smoother is not None:
            people = self.smoother(t, people)
        patient, others = self.selector.select(people, t)
        # full res frame for chest/hands/face crops. not at night, we work on the brightened copy then
        hi = (full, full.shape[1] / frame.shape[1]) if full is not None and cfg.hires_crops and not self.night else None
        self.hands.hi = hi
        if self.face is not None:
            self.face.hi = hi

        chest = face = None
        if patient is not None:
            self.speed = self._body_speed(t, patient)
            chest = self._smooth_chest(chest_box(patient, cfg.kp_conf, shape))
            face = face_box(patient, cfg.kp_conf, shape)
        else:
            self.chest_s, self.speed = None, 0.0
            self.track_hist.clear()
        if self.speed > cfg.motion_thr:
            self.moving_until = t + 1.5
        moving = t < self.moving_until

        # only measure breathing with the whole torso in view. guessing the chest = noise = fake apneas
        self.breath_view = torso_view(patient, shape, cfg.breath_view_conf) if patient is not None else "no_patient"
        if self.breath_view is None:
            self._view_ok_since = self._view_ok_since or t
        else:
            if self._view_ok_since is not None:
                self.breath.pause()
            self._view_ok_since = None
            chest = None
        self.measurable = self._view_ok_since is not None and t - self._view_ok_since >= cfg.breath_view_hold_s
        self._track_flow(self.breath, "chest", t, chest, hi, gray, moving)
        if cfg.hr_enabled:
            self.hrest.update(t, frame, face, still=not moving and self.speed < 0.1)

        for ev in self.posture.update(t, patient, shape):
            self._on_posture_event(ev)
        in_bed = patient is not None and self.posture.state in IN_BED
        self._vision_ai(t, seen, patient, shape)
        eyes = self.face.eyes_open if self.face is not None and self.face.eye_hist \
            and t - self.face.eye_hist[-1][0] < 3 else None
        self.sleep.update(t, in_bed, self.speed > 0.08, eyes)
        standing = patient is not None and self.posture.state == "standing"
        for ev in self.reposition.update(t, patient, in_bed, standing):
            self._on_reposition_event(t, ev)
        if self.visits.update(t, len(others)):
            self.mark_visit(auto=True)
        if others and self._last_t is not None:
            self.staff_seconds += min(1.0, t - self._last_t)
        zone_fn = (lambda pt: self.posture.in_zone(pt, shape)) if self.posture.bed_zone else None
        if self.room_falls.update(t, others, zone_fn):
            self.log.add(self.bed_id, "fall_other")
            self._alert("fall_other", "critical")
        for ev in self.hands.update(t, seen, patient, self.posture.state):
            self._on_hand_event(t, ev)
        self._bedside_exams(t, patient)

        if t - self._last_est >= 0.5:
            self._last_est = t
            self.breath.estimate(t)
            if cfg.hr_enabled:
                self.hrest.estimate(t)
            self._assess(t, patient, in_bed, moving)

        # count frames in the last 2s. 1/dt went crazy when wifi cams deliver in bursts
        now = time.time()
        self._proc_times.append(now)
        while self._proc_times and self._proc_times[0] < now - 2.0:
            self._proc_times.popleft()
        self.fps = len(self._proc_times) / 2.0
        self.delay = 0.8 * self.delay + 0.2 * max(0.0, now - t)
        self._last_t = t

        # magnifier filters need every frame, not just the ones we send
        shown = frame
        if self.magnifier.mode is not None:
            box = self._head_box if self.magnifier.mode == "pulse" else (patient.box if patient is not None else None)
            if box is not None:
                shown = self.magnifier.apply(t, frame, box, still=not moving)
        if t - self._last_jpeg >= 1.0 / cfg.stream_fps:
            self._last_jpeg = t
            if shown is not frame or self.magnifier.mode is not None:
                img = render(shown, "raw", None, (), None, None, None, self.snap.get("level", "grey"), cfg.kp_conf)
            else:
                img = render(frame, self.privacy, patient, others, chest, face,
                             self.posture.bed_zone, self.snap.get("level", "grey"), cfg.kp_conf,
                             head=self._head_box, label=self._patient_label(patient))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                self.jpeg = buf.tobytes()
        if cfg.relay_url and t - self._last_cloud_jpeg >= 0.33:
            self._last_cloud_jpeg = t
            img = render(frame, "skeleton", patient, others, chest, None,
                         self.posture.bed_zone, self.snap.get("level", "grey"), cfg.kp_conf)
            ok, buf = cv2.imencode(".jpg", cv2.resize(img, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                self.cloud_jpeg = buf.tobytes()
        self.snap = self._snapshot(t, patient is not None, moving)
        for alert_id in self.recorder.add(t, shape, patient, others, self.posture.state,
                                          self.snap.get("level", "grey"), self.posture.bed_zone):
            self.alerts.mark_replay(alert_id)

    def _track_flow(self, est, key, t, box, hi, gray, moving):
        """chest motion, from the full res frame if we have it (at 640px a breath is < 1px)"""
        cfg = self.cfg
        if hi is None or box is None:
            est.roi_px = cfg.flow_roi_px
            est.update(t, gray, box, moving)
            return
        k = hi[1]
        hb = tuple(int(v * k) for v in box)
        # grey only a window around the chest (full 1080p was ~3ms). window stays put so crops keep their size
        H, W = hi[0].shape[:2]
        win = self._flow_wins.get(key)
        if win is None or hb[0] < win[0] or hb[1] < win[1] or hb[2] > win[2] or hb[3] > win[3]:
            pw, ph = (hb[2] - hb[0]) // 2, (hb[3] - hb[1]) // 2
            win = self._flow_wins[key] = (max(0, hb[0] - pw), max(0, hb[1] - ph), min(W, hb[2] + pw), min(H, hb[3] + ph))
        crop = cv2.cvtColor(hi[0][win[1]:win[3], win[0]:win[2]], cv2.COLOR_BGR2GRAY)
        est.roi_px = cfg.flow_roi_hi_px
        est.update(t, crop, (hb[0] - win[0], hb[1] - win[1], hb[2] - win[0], hb[3] - win[1]), moving, unit=k)

    def _assess(self, t, patient, in_bed, moving):
        cfg, b = self.cfg, self.baseline
        present = patient is not None
        measurable = present and self.measurable
        rr = self.breath.rr if measurable else None
        rr_ok = measurable and rr is not None and self.breath.quality >= 0.25 and not self.breath.motion
        good = rr_ok and self.breath.quality >= cfg.breath_good_q and 6 <= rr <= 40
        if rr_ok:
            self.rr_valid_t = t
        if good:
            self._good_breath_t = t
        if rr is not None:
            self.rr_samples.append((t, rr, self.breath.quality))
        rr_for_risk = rr if measurable and t - self.rr_valid_t < 15 else None
        hr = self.hrest.hr if cfg.hr_enabled else None
        hr_ok = hr is not None and self.hrest.quality > 0.3

        if present and in_bed and not moving and not b.ready:
            b.add(t, rr, good, hr, hr_ok)
            self.posture.observe_for_zone(patient, self.shape)
            if b.ready:
                cv = b.rr_sd / b.rr_med if b.rr_med else 1.0
                if not (8 <= b.rr_med <= 30) or cv > 0.35:
                    # junk baseline (once learned 8.9/min off a chest that wasn't even visible), redo
                    self.log.add(self.bed_id, "baseline_rejected", rr=b.rr_med, rr_sd=b.rr_sd)
                    b.reset()
                else:
                    self.posture.finalize_zone()
                    self.log.add(self.bed_id, "baseline", rr=b.rr_med, rr_sd=b.rr_sd, hr=b.hr_med)

        level = self.risk.update(t, rr_for_risk, rr_ok, hr, hr_ok, b)

        # apnea only if: lying still in bed, chest visible, baseline done, and we actually saw
        # breaths right before. otherwise we can't tell a pause from "wasn't looking"
        ap = self.breath.apnea_seconds(t) if measurable and in_bed and not moving and b.ready else 0.0
        seen_breathing = self._good_breath_t is not None and t - self._good_breath_t <= cfg.apnea_s + 15
        if not self.apnea_active and not seen_breathing:
            ap = 0.0
        self.apnea_s = ap
        if ap >= cfg.apnea_s:
            if not self.apnea_active:
                self.apnea_episodes += 1
                self.log.add(self.bed_id, "apnea_episode")
            self.apnea_active = True
            self._alert("apnea", "critical", seconds=int(ap))
        elif self.apnea_active and ap < cfg.apnea_s:
            self.apnea_active = False
            self.alerts.resolve(self.bed_id, "apnea")

        # rate alerts need to persist, single noisy readings fired high+low 1s apart
        kind = None
        if level != "green" and rr_for_risk is not None and good \
                and self.risk.parts.get("rr", 0) + self.risk.parts.get("deviation", 0) > 0:
            ref = b.rr_med if b.ready else 16.0
            kind = "rr_high" if rr_for_risk > ref else "rr_low"
        if kind is None:
            if level == "green" or not measurable:
                self._rr_cond = None
                self.alerts.resolve(self.bed_id, "rr_high")
                self.alerts.resolve(self.bed_id, "rr_low")
        else:
            if self._rr_cond is None or self._rr_cond[0] != kind:
                self._rr_cond = (kind, t)
            if t - self._rr_cond[1] >= cfg.rr_alert_hold_s:
                ref = b.rr_med if b.ready else 16.0
                self._alert(kind, "critical" if level == "red" else "warning",
                            rr=int(round(rr_for_risk)), base=int(round(ref)))

        if cfg.hr_in_risk and hr_ok and level != "green" and self.risk.parts.get("hr", 0) >= 2:
            self._alert("hr_high" if hr > 90 else "hr_low", "warning", hr=int(hr))

        # rounding overdue
        since_visit = t - (self.last_visit or self.monitor_start)
        if present and since_visit >= cfg.rounding_interval_s and not self.rounding_raised:
            self.rounding_raised = True
            self._alert("rounding", "warning", minutes=int(since_visit / 60))

        if t - self._last_vitals >= 10:
            self._last_vitals = t
            self.log.add(self.bed_id, "vitals", rr=_r(rr_for_risk), hr=_r(hr if hr_ok else None),
                         level=level, posture=self.posture.state, sleep=self.sleep.state)

    def _on_posture_event(self, ev):
        self.log.add(self.bed_id, "posture", event=ev, state=self.posture.state)
        if ev == "bed_exit":
            self.bed_exits += 1
        if ev == "near_fall":
            return  # just counted
        if ev == "floor_long":
            self._alert(ev, "critical", seconds=int(self.posture.floor_seconds(self._last_t or 0)))
            return
        if ev == "back_in_bed":
            for kind in ("bed_edge", "bed_exit", "absent"):
                self.alerts.resolve(self.bed_id, kind)
            return
        if ev == "help" and time.time() - self.hands.last_seen_hand < 3:
            return
        if ev == "fall":
            self._alert("fall", "critical")
            self.say_to_patient("fall")
            return
        if ev in POSTURE_ALERTS:
            self._alert(ev, POSTURE_ALERTS[ev])
            self.say_to_patient(ev)

    def _on_hand_event(self, t, ev):
        self.log.add(self.bed_id, "hand_signal", signal=ev)
        if ev == "patient_ok":
            for kind in REQUESTS:
                self.alerts.resolve(self.bed_id, kind)
            self.say_to_patient("patient_ok")
            return
        self._alert(ev, HAND_ALERTS.get(ev, "warning"))
        self.say_to_patient(ev)

    # ---------------- bedside screen / speaker ----------------
    def say_to_patient(self, kind=None, text=None, clip=None, audio=None, mime=None, sender="system"):
        """message for the bedside screen (shown + spoken)"""
        if kind is not None:
            if not self.cfg.patient_prompts or kind not in i18n.PATIENT_PROMPT:
                return None
            text, clip = i18n.PATIENT_PROMPT[kind], f"p_{kind}"
        self._msg_id += 1
        msg = {"id": self._msg_id, "ts": time.time(), "text": text or "", "clip": clip, "sender": sender,
               "audio": bool(audio)}
        if audio:
            self.audio[self._msg_id] = (audio, mime or "audio/webm")
            for old in list(self.audio)[:-10]:
                self.audio.pop(old, None)
        self.messages.append(msg)
        if sender != "system":
            self.log.add(self.bed_id, "nurse_message", text=text, audio=bool(audio))
        return msg

    def nurse_message(self, preset=None, text=None, audio=None, mime=None):
        if preset:
            if preset not in NURSE_PRESETS:
                raise ValueError("unknown preset")
            return self.say_to_patient(text=NURSE_PRESETS[preset], clip=preset, sender="nurse")
        return self.say_to_patient(text=text, audio=audio, mime=mime, sender="nurse")

    def patient_button(self):
        self.log.add(self.bed_id, "patient_button")
        self._alert("patient_button", "warning")
        self.say_to_patient("patient_button")

    def bedside(self, after=0):
        active = self.alerts.active(self.bed_id)
        if any(a.severity == "critical" for a in active) or active:
            status = "A nurse is coming"
        else:
            status = "All good. Rest."
        return {"name": self.name, "status": status, "alerting": bool(active),
                "messages": [m for m in self.messages if m["id"] > after],
                "hand": self.hands.current}

    # ---------------- nurse voice notes ----------------
    def add_note(self, note):
        note = dict(note, ts=note.get("ts") or time.time())
        self.notes.append(note)
        self.log.add(self.bed_id, "note", summary=note.get("summary"), urgency=note.get("urgency"),
                     observations=note.get("observations"), vitals=note.get("vitals"),
                     actions=note.get("actions"), transcript=note.get("transcript"))
        return note

    def _on_reposition_event(self, t, ev):
        self.log.add(self.bed_id, ev)
        if ev == "repositioned":
            self.alerts.resolve(self.bed_id, "turn_due")
        elif ev == "turn_due":
            self._alert("turn_due", "warning", minutes=int(self.reposition.minutes_in_position(t)))

    def _alert(self, kind, severity, extra=None, meta=None, **kw):
        msg = i18n.alert_message(kind, **kw) + (f" - {extra}" if extra else "")
        a = self.alerts.raise_alert(self.bed_id, self.name, kind, severity, msg)
        if a is not None and getattr(self, "recorder", None) is not None:
            self.recorder.request(a.id, kind, msg, self._now, meta)
        return a

    # ---------------- helpers ----------------
    def _body_speed(self, t, p):
        tr = torso(p, self.cfg.kp_conf)
        if tr is None:
            return self.speed
        sh, hip, length = tr
        c = (sh + hip) / 2
        self.track_hist.append((t, c))
        while self.track_hist and self.track_hist[0][0] < t - 1.0:
            self.track_hist.popleft()
        old = [h for h in self.track_hist if h[0] <= t - 0.4]
        if not old:
            return self.speed
        t0, c0 = old[-1]
        return float(np.linalg.norm(c - c0) / (t - t0) / length)

    def _smooth_chest(self, box):
        if box is None:
            return None
        b = np.array(box, float)
        if self.chest_s is None or np.abs(b - self.chest_s).max() > 0.3 * (b[2] - b[0]):
            self.chest_s = b
        else:
            self.chest_s = 0.9 * self.chest_s + 0.1 * b
        x1, y1, x2, y2 = (int(round(v)) for v in self.chest_s)
        return x1, y1, x2, y2

    # ---------------- actions from the dashboard ----------------
    def recalibrate(self, keep_zone=False):
        self._recal_request = keep_zone

    def _recalibrate(self, keep_zone):
        self.breath.reset()
        self.hrest.reset()
        self.baseline.reset()
        self.risk.reset()
        self.reposition.reset()
        self.hands.reset()
        if self.face:
            self.face.reset()
        self.apnea_episodes = 0
        self.monitor_start = time.time()
        zone, auto = self.posture.bed_zone, self.posture.zone_auto
        self.posture.reset()
        if not keep_zone and auto:
            zone = None
        self.posture.set_zone(zone, auto)
        self.pose.reset()
        self._reset_tracking()
        self.log.add(self.bed_id, "recalibrate")

    # ascii only, cv2 font can't do georgian
    POSTURE_EN = {"lying": "LYING", "reclined": "RECLINED", "sitting": "SITTING", "standing": "STANDING",
                  "floor": "ON FLOOR", "absent": "", "unknown": "TRACKING"}

    def _hud(self):
        parts = [f"{self.fps:.0f} fps", f"{self.delay * 1000:.0f} ms"]
        if self.night:
            parts.append("NIGHT")
        parts.append("AI: pose+breath+face+hands")
        return "  |  ".join(parts)

    def _patient_label(self, patient):
        if patient is None:
            return None
        s = self.POSTURE_EN.get(self.posture.state, "")
        rr = self.breath.rr
        if rr is not None and self.posture.state in IN_BED and self.breath.quality >= 0.25:
            s += f"  RR {rr:.0f}"
        hr = self.snap.get("hr")
        if hr is not None:
            s += f"  HR {hr:.0f}"
        return f"PATIENT  {s}".strip()

    # ---------------- camera settings (dashboard camera panel) ----------------
    def _camera_host(self):
        from .onvif import host_of
        return host_of(self.capture.source) if self.capture.source.startswith(("onvif://", "rtsp://")) else None

    def camera_info(self, probe=False):
        """camera info + health. probe=True asks the camera over onvif"""
        from .onvif import cached_info
        host = self._camera_host()
        info = {"kind": _source_kind(self.capture.source), "host": host,
                "onvif": self.capture.source.startswith("onvif://"),
                "stream": _onvif_stream(self.capture.target()),
                "switching": self.capture._switch_to is not None,
                "privacy": self.privacy, "processing_fps": round(self.fps, 1),
                "delay_s": round(self.delay, 2), "model": os.path.basename(getattr(self.pose, "model_file", "")),
                "hires": self.hands.hi is not None,
                "processed_size": [int(self.shape[1]), int(self.shape[0])], **self.capture.health()}
        if info["onvif"]:
            # cached at connect, asking while streaming kills the conico stream
            info.update(cached_info(host))
            info["ptz"] = True
        return info

    def set_camera(self, stream=None, rotate=None, mirror=None, save=True):
        cap = self.capture
        current = cap.target()
        if stream in ("main", "sub") and current.startswith("onvif://") and stream != _onvif_stream(current):
            cap.switch(_onvif_base(current) + "/" + stream)
        geometry_changed = False
        if rotate is not None and int(rotate) in (0, 90, 180, 270) and int(rotate) != cap.rotate:
            cap.rotate, geometry_changed = int(rotate), True
        if mirror is not None and bool(mirror) != cap.mirror:
            cap.mirror, geometry_changed = bool(mirror), True
        if geometry_changed:
            # rotated -> old bed zone is wrong
            self.recalibrate(keep_zone=False)
        if save:
            self.settings.set(self.bed_id, "camera", {"stream": _onvif_stream(cap.target()),
                                                       "rotate": cap.rotate, "mirror": cap.mirror})
        self.log.add(self.bed_id, "camera_settings", stream=stream, rotate=rotate, mirror=mirror)
        return self.camera_info()

    def ptz(self, pan=0.0, tilt=0.0, zoom=0.0, ms=400):
        """pan/tilt for ms then stop"""
        from .onvif import ptz_move, ptz_stop
        host = self._camera_host()
        if host is None:
            return False
        clamp = lambda v: max(-1.0, min(1.0, float(v)))      # noqa: E731
        ok = ptz_move(host, clamp(pan), clamp(tilt), clamp(zoom))
        if ok:
            threading.Timer(max(0.05, min(2.0, ms / 1000)), ptz_stop, args=(host,)).start()
        return ok

    # ---------------- bedside exams (nurse presses start) ----------------
    def _bedside_exams(self, t, patient):
        last = self.face.last if self.face is not None else None
        r = self.facetest.update(t, last, head_pose(last[1])[0] if last is not None else 0.0)
        if r == "next":
            self.say_to_patient("face_eyes")
        elif r is not None:
            self.log.add(self.bed_id, "face_test", **r)
            if r["asymmetric"]:
                self._alert("face_test_asym", "critical", side=i18n.SIDE_HAND.get(r["side"], ""))
        n = self.chair.update(t, patient, self.cfg.kp_conf)
        if n is not None:
            self.log.add(self.bed_id, "chair_stand", count=n)
            if n < self.cfg.chair_stand_low:
                self._alert("chair_stand_low", "warning", count=n)

    def start_face_test(self):
        self.facetest.start(self._now)
        self.say_to_patient("face_smile")
        self.log.add(self.bed_id, "face_test_start")

    def start_chair_stand(self):
        self.chair.arm(self._now)
        self.say_to_patient("chair_stand")
        self.log.add(self.bed_id, "chair_stand_start")

    def reconnect_camera(self):
        self.capture.reconnect()

    def set_privacy(self, mode):
        if mode in MODES:
            self.privacy = mode

    def set_magnify(self, mode):
        """'breath' | 'pulse' | None"""
        self.magnifier.set_mode(mode)
        self.log.add(self.bed_id, "magnify", mode=self.magnifier.mode)

    def set_zone(self, zone):
        self.posture.set_zone(zone)
        self.settings.set(self.bed_id, "zone", list(zone) if zone else None)

    def mark_visit(self, auto=False):
        self.last_visit = time.time()
        self.rounding_raised = False
        self.alerts.resolve(self.bed_id, "rounding")
        self.log.add(self.bed_id, "visit", auto=auto)

    def validate(self, duration_s, manual_rr):
        """nurse taps breaths for 60s, compare with ours"""
        now = time.time()
        vals = [rr for t, rr, q in self.rr_samples if t >= now - duration_s]
        system = float(np.median(vals)) if vals else None
        err = None if system is None else system - manual_rr
        path = os.path.join(self.cfg.data_dir, "validation.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "bed", "duration_s", "manual_rr", "system_rr", "error"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.bed_id, round(duration_s, 1),
                        round(manual_rr, 1), _r(system), _r(err)])
        return {"manual_rr": round(manual_rr, 1), "system_rr": _r(system), "error": _r(err),
                "pass": err is not None and abs(err) <= 2}

    # ---------------- snapshot ----------------
    def _snapshot(self, t, present, moving):
        active = self.alerts.active(self.bed_id)
        alert_level = "red" if any(a.severity == "critical" for a in active) else "yellow" if active else "green"
        level = max(self.risk.level, alert_level, key=LEVELS.index)
        rr_valid = t - self.rr_valid_t < 15
        if not present and not self.baseline.ready and not active:
            level = "grey"
        hr_shown, hr_status = self.hrest.shown(t) if present else (None, "no_face")
        in_bed = present and self.posture.state in IN_BED
        return {
            "id": self.bed_id, "name": self.name, "simulated": False,
            "level": level, "level_label": i18n.LEVEL[level],
            "score": self.risk.score, "parts": self.risk.parts, "explain": self.risk.explain(),
            "rr": _r(self.breath.rr) if present and self.measurable else None, "rr_valid": rr_valid,
            "breath_view": self.breath_view, "measurable": self.measurable,
            "rr_quality": round(self.breath.quality, 2),
            "trend": self.risk.trend, "trend_label": i18n.TREND[self.risk.trend],
            "slope": round(self.risk.slope, 1),
            "hr": _r(hr_shown, 0), "hr_status": hr_status,
            "hr_quality": round(self.hrest.quality, 2),
            "present": present, "moving": moving,
            "posture": self.posture.state, "posture_label": i18n.POSTURE[self.posture.state],
            "posture_since": self.posture.since,
            "apnea_s": round(self.apnea_s, 1), "apnea_episodes": self.apnea_episodes,
            "baseline": self.baseline.as_dict(t),
            "sleep": self.sleep.state, "sleep_label": i18n.SLEEP[self.sleep.state],
            "asleep_min": round(self.sleep.asleep_total / 60, 1),
            "eyes_open": self.face.eyes_open if self.face and self.face.eye_hist else None,
            "last_visit": self.last_visit, "bed_exits": self.bed_exits,
            "rr_history": self.risk.history_list(),
            "wave": self.breath.waveform(20), "wave_hz": self.cfg.resample_hz,
            "near_falls": self.posture.fastfall.near_falls,
            "facetest_state": self.facetest.state, "facetest": self.facetest.result,
            "chair_state": self.chair.state,
            "chair_count": self.chair.count if self.chair.state == "running" else self.chair.result,
            "chair_history": self.chair.history[-5:],
            "fps": round(self.fps, 1), "privacy": self.privacy, "magnify": self.magnifier.mode,
            "zone": self.posture.bed_zone, "zone_auto": self.posture.zone_auto, "zone_source": self._zone_source,
            "alerts": [a.id for a in active],
            "position_min": round(self.reposition.minutes_in_position(t), 1) if in_bed else None,
            "turn_due_min": round(self.cfg.turn_interval_s / 60, 1),
            "repositions": self.reposition.count,
            "night": self.night, "source_kind": _source_kind(self.source),
            "hand_signal": self.hands.current,
            "hand_label": i18n.HAND_SIGNAL.get(self.hands.current) if self.hands.current is not None else None,
            "staff_min": round(self.staff_seconds / 60, 1),
            "rounding_due_min": round(self.cfg.rounding_interval_s / 60, 1),
            "notes": list(self.notes)[-5:],
            "patient_messages": list(self.messages)[-5:],
            "hidden": self.posture.hidden,
            "floor_s": round(self.posture.floor_seconds(t), 1),
            "features": self.features(t, present, in_bed), "camera_offline": False,
        }

    def features(self, t, present, in_bed):
        """state of every feature: on / wait (+ why) / off. so you can see why something isn't measuring"""
        b, F, W = self.baseline, i18n.FEATURE, i18n.FEATURE_WAIT
        connected = self.capture.health().get("connected", False) if hasattr(self.capture, "health") else True

        def item(key, state, detail=""):
            return {"key": key, "name": F[key]["name"], "what": F[key]["what"], "state": state, "detail": detail}

        if not connected:
            return [item(k, "off", W["camera"]) for k in F]
        view = W.get(self.breath_view) if self.breath_view else None
        learning = W["learning"].format(pct=round(100 * b.progress(t)))
        out = []
        # breathing rate
        if present and self.measurable and self.breath.rr is not None:
            out.append(item("breathing", "on", f"{self.breath.rr:.0f}/min"))
        else:
            out.append(item("breathing", "wait", view or W["measuring"]))
        # apnea
        if not present:
            out.append(item("apnea", "wait", W["no_patient"]))
        elif not in_bed:
            out.append(item("apnea", "wait", W["lie_down"]))
        elif not self.measurable:
            out.append(item("apnea", "wait", view or W["measuring"]))
        elif not b.ready:
            out.append(item("apnea", "wait", learning))
        else:
            out.append(item("apnea", "on", W["watching"]))
        # rate vs own baseline
        if b.ready:
            out.append(item("rr_change", "on", W["normal_rr"].format(rr=round(b.rr_med))))
        else:
            out.append(item("rr_change", "wait", learning if in_bed else W["lie_down"]))
        # heart rate
        hr, why = self.hrest.shown(t) if present else (None, "no_face")
        out.append(item("heart_rate", "on", f"{hr:.0f}/min") if hr is not None
                   else item("heart_rate", "wait", W[why]))
        # falls, bed exit
        if present:
            out.append(item("fall", "on", W["watching"]))
            out.append(item("bed_exit", "on", W["zone_" + (self._zone_source or "none")]))
        else:
            out.append(item("fall", "wait", W["no_patient"]))
            out.append(item("bed_exit", "wait", W["no_patient"]))
        if not present:
            out.append(item("posture", "wait", W["no_patient"]))
        elif self.posture.state == "unknown":
            out.append(item("posture", "wait", W["body_unclear"]))
        else:
            out.append(item("posture", "on", i18n.POSTURE[self.posture.state]))
        # hand signs
        if not self.cfg.hand_signals:
            out.append(item("hand_signs", "off", W["disabled"]))
        elif present:
            out.append(item("hand_signs", "on", i18n.HAND_SIGNAL.get(self.hands.current) or W["show_hand"]))
        else:
            out.append(item("hand_signs", "wait", W["no_patient"]))
        # sleep + turning, in bed only
        if in_bed:
            out.append(item("sleep", "on", i18n.SLEEP[self.sleep.state]))
            out.append(item("turning", "on", W["in_position"].format(m=round(self.reposition.minutes_in_position(t)))))
        else:
            out.append(item("sleep", "wait", W["lie_down"] if present else W["no_patient"]))
            out.append(item("turning", "wait", W["lie_down"] if present else W["no_patient"]))
        out.append(item("visits", "on", W["visit_none"] if self.last_visit is None
                        else W["visit_ago"].format(m=round((time.time() - self.last_visit) / 60))))
        # exams
        busy = self.facetest.state != "idle"
        out.append(item("face_test", "on" if busy else "wait", W["running"] if busy else W["press_start"]))
        busy = self.chair.state in ("armed", "running")
        out.append(item("chair_stand", "on" if busy else "wait", W["running"] if busy else W["press_start"]))
        return out

    def _vision_ai(self, t, seen, patient, shape):
        """bed detection + face model"""
        cfg = self.cfg
        if self.scene is not None:
            try:
                self.scene.update(t, seen, patient, None)
                # detected bed wins over the learned zone
                z = self.scene.bed_zone(shape)
                if z is not None and (self.posture.bed_zone is None or self.posture.zone_auto):
                    if self._zone_source != "bed":
                        self.log.add(self.bed_id, "bed_detected", zone=[round(v, 3) for v in z])
                    self.posture.set_zone(z, auto=True)
                    self._zone_source = "bed"
            except Exception as e:
                print(f"[scene] object detector off: {e}")
                self.scene = None
        self._head_box = None
        if self.face is not None and patient is not None:
            try:
                self._head_box = self.face.head_box(patient, cfg.kp_conf, shape)
                self.face.update(t, seen, patient)
            except Exception as e:
                print(f"[face] face AI off: {e}")
                self.face = None


def _r(v, n=1):
    return None if v is None else round(float(v), n)


def _onvif_base(source):
    s = source.rstrip("/")
    for suffix in ("/sub", "/main"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def _onvif_stream(source):
    """onvif://host -> main, onvif://host/sub -> sub"""
    return "sub" if source.rstrip("/").endswith("/sub") else "main"


def _source_kind(source):
    if source.isdigit():
        return "webcam"
    if source.startswith(("onvif://", "rtsp://")):
        return "ipcam"
    return "phone" if source.startswith(("http://", "https://")) else "video"
