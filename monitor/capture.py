"""Capture thread, always gives you the newest frame.

source = webcam index, video file, http/rtsp url, or onvif://ip.
(a phone with an ip camera app works too: http://<phone>:8080/video)
"""
import os
import platform
import re
import threading
import time
from collections import deque

import cv2

SETTLE_S = 2.5  # drop frames for this long after (re)connect


class Capture:
    def __init__(self, source="0", width=640, height=480, fps=30, loop=True, record_path=""):
        self.source = str(source)
        self.onvif = self.source.startswith("onvif://")
        self.is_stream = self.onvif or self.source.startswith(("http://", "https://", "rtsp://"))
        self.is_file = not self.source.isdigit() and not self.is_stream
        self.loop = loop
        self.width = width
        if self.onvif or self.source.startswith("rtsp://"):
            # udp over wifi = smeared frames
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        try:
            self.cap = self._open(width, height, fps)
        except RuntimeError as e:
            if not self.is_stream:
                raise
            # camera off / not reachable: start anyway and keep retrying
            print(f"[camera] {e} - starting without it, retrying every few seconds")
            self.cap = cv2.VideoCapture()
        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.writer = None
        self.record_path = record_path
        self._cond = threading.Condition()
        self._frame = None
        self._full = None
        self._t = 0.0
        self._seq = 0
        self.restarts = 0  # video file looped
        self.running = True
        self.rotate = 0
        self.mirror = False
        self.reconnects = 0
        self.in_size = None
        self.in_fps = 0.0
        self._switch_to = None
        self._last_in = None
        self._arrivals = deque(maxlen=120)
        self._pts0 = None
        self.lag = 0.0
        self.max_lag_s = 2.5
        self._lag_since = None
        self.backlog_flushes = 0
        self.downgraded = False
        self.settled_out = 0
        self.ok = self.cap.isOpened()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self, width, height, fps):
        if self.is_file or self.is_stream:
            try:
                cap = self._connect()
            except Exception as e:
                safe = re.sub(r"//[^/@]*@", "//***@", str(e))     # never print a camera password
                raise RuntimeError(f"Cannot open video source: {safe}") from None
        else:
            idx = int(self.source)
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            safe = re.sub(r"//[^/@]*@", "//***@", self.source)     # never print a camera password
            raise RuntimeError(f"Cannot open video source {safe!r}")
        return cap

    def _connect(self):
        # timeouts so a frozen stream (conico freezes on pan/tilt) reconnects instead of hanging.
        # tried pyav to drop broken frames, was slower and hung too
        url = self._url()
        self._connected_at = time.time()
        if url.startswith(("rtsp://", "http://", "https://")):
            return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                                                          cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        return cv2.VideoCapture(url)

    def _url(self):
        if not self.onvif:
            return self.source
        from .onvif import resolve
        return resolve(self.source)

    def _run(self):
        period = 1.0 / self.file_fps if self.is_file else 0.0
        next_t = time.time()
        while self.running:
            if self._switch_to is not None:
                self._apply_switch()
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.restarts += 1
                    continue
                self.ok = False
                if self.is_stream:
                    time.sleep(1.0)
                    self.cap.release()
                    self.reconnects += 1
                    self._pts0 = None
                    try:
                        self.cap = self._connect()
                    except Exception:
                        pass
                else:
                    time.sleep(0.05)
                continue
            self.ok = True
            now = time.time()
            self._last_in = now
            self._arrivals.append(now)
            recent = [a for a in self._arrivals if a >= now - 2.0]
            self.in_fps = len(recent) / 2.0 if len(recent) > 1 else 0.0
            stamp = self._stream_time(now) if self.is_stream else now
            self.in_size = (frame.shape[1], frame.shape[0])
            full = None
            if frame.shape[1] > 1.5 * self.width:
                # 1080p cam: everything runs on a 640px copy, keep the full frame for the crops
                full = frame
                frame = cv2.resize(frame, (self.width, int(frame.shape[0] * self.width / frame.shape[1])),
                                   interpolation=cv2.INTER_AREA)
            if self.rotate:
                code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate]
                frame = cv2.rotate(frame, code)
                full = cv2.rotate(full, code) if full is not None else None
            if self.mirror:
                frame = cv2.flip(frame, 1)
                full = cv2.flip(full, 1) if full is not None else None
            if self.is_file:
                # real speed for files
                next_t += period
                delay = next_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.time()
            if self.is_stream and now - self._connected_at < SETTLE_S:
                # until the next keyframe the picture is smeared garbage, skip it
                self.settled_out += 1
                continue
            if self.record_path:
                self._record(frame)
            with self._cond:
                self._frame = frame
                self._full = full
                self._t = stamp if self.is_stream else time.time()
                self._seq += 1
                self._cond.notify_all()

    def switch(self, source):
        """switch stream (main/sub), applied in the reader thread"""
        self._switch_to = str(source)
        self.backlog_flushes, self.downgraded = 0, False

    def _stream_time(self, now):
        """use the stream pts instead of arrival time. wifi cams deliver in bursts
        (3ms apart then 1.7s nothing) which wrecks breathing/fall timing"""
        pts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pts <= 0:
            return now
        # map with the smallest (arrival - pts) seen so far, lag = how much later than that.
        # anchoring on the first frame didn't work, it can itself be late
        offset = now - pts
        if self._pts0 is None or pts < self._pts0[0] or offset < self._pts0[1] - 5.0:
            self._pts0 = (pts, offset)
        elif offset < self._pts0[1]:
            self._pts0 = (self._pts0[0], offset)
        t = pts + self._pts0[1]
        late = now - t
        # backlog builds up on bad wifi (saw 6.7s). reconnect to flush, second time go to the sub stream
        self.lag = 0.8 * self.lag + 0.2 * max(0.0, late)
        if self.lag > self.max_lag_s:
            self._lag_since = self._lag_since or now
            if now - self._lag_since > 5.0:
                self._lag_since, self.lag = None, 0.0
                self.backlog_flushes += 1
                down = self.onvif and self.backlog_flushes >= 2 and not self.source.rstrip("/").endswith("/sub")
                if down:
                    base = self.source.rstrip("/")
                    base = base[: -len("/main")] if base.endswith("/main") else base
                    self.downgraded = True
                    self._switch_to = base + "/sub"
                else:
                    self._switch_to = self.source
        else:
            self._lag_since = None
        return t

    def target(self):
        return self._switch_to or self.source

    def reconnect(self):
        self._switch_to = self.source

    def _apply_switch(self):
        src, self._switch_to = self._switch_to, None
        self.source = src
        self.onvif = src.startswith("onvif://")
        self._pts0 = None
        self.cap.release()
        self.reconnects += 1
        try:
            self.cap = self._connect()
        except Exception:
            pass

    def health(self):
        age = None if self._last_in is None else round(time.time() - self._last_in, 1)
        arr = list(self._arrivals)
        gaps = sorted(b - a for a, b in zip(arr, arr[1:]))
        gap = {"gap_p50_ms": round(gaps[len(gaps) // 2] * 1000) if gaps else None,
               "gap_p95_ms": round(gaps[int(len(gaps) * 0.95)] * 1000) if gaps else None,
               "gap_max_ms": round(gaps[-1] * 1000) if gaps else None}
        return {"connected": bool(self.ok) and age is not None and age < 3, "frame_age_s": age,
                "in_fps": round(self.in_fps, 1), "in_size": self.in_size,
                "lag_s": round(self.lag, 2), "backlog_flushes": self.backlog_flushes, "downgraded": self.downgraded,
                "hidden_broken": self.settled_out,
                "reconnects": self.reconnects, "rotate": self.rotate, "mirror": self.mirror, **gap}

    def _record(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.record_path, fourcc, self.file_fps or 30, (w, h))
        self.writer.write(frame)

    def full_frame(self, seq):
        """full res frame for seq, or None"""
        with self._cond:
            return self._full if self._seq == seq else None

    def read(self, last_seq=0, timeout=1.0):
        """wait for a frame newer than last_seq -> (seq, t, frame) or None"""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq or not self.running, timeout):
                return None
            if self._frame is None:
                return None
            return self._seq, self._t, self._frame

    def release(self):
        self.running = False
        self._thread.join(timeout=2)
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
