"""yolo11n object detector, runs every few seconds. mainly finds the bed for the bed zone.
(cup-at-mouth is still here but the drinking feature got cut)
"""
from collections import deque

import numpy as np

from .pose import NOSE, kp

CLASSES = {"bed": 59, "couch": 57, "chair": 56, "toilet": 61, "cup": 41, "bottle": 39,
           "wine glass": 40, "bowl": 45, "cell phone": 67, "dining table": 60}
DRINK = ("cup", "bottle", "wine glass")
FOOD = ("bowl",)


class SceneDetector:
    def __init__(self, cfg, model="yolo11n.pt", detector=None):
        self.cfg = cfg
        self._model_path = model
        self._detect = detector             # test hook: frame -> [(name, box, conf)]
        self._model = None
        self.objects = []                   # [(name, box, conf)] from the last run
        self.last_run = -1e9
        self.bed_boxes = deque(maxlen=15)
        self.near_mouth = None              # "drink" | "food" | None
        self.near_mouth_since = None
        self.intake_logged_at = -1e9
        self.first_t = None

    def _run_model(self, frame):
        if self._detect is not None:
            return self._detect(frame)
        if self._model is None:
            from ultralytics import YOLO

            from .config import model_path
            self._model = YOLO(model_path(self._model_path, self.cfg.imgsz), task="detect")
        res = self._model.predict(frame, imgsz=self.cfg.imgsz, conf=0.25, verbose=False,
                                  classes=list(CLASSES.values()))[0]
        names = {v: k for k, v in CLASSES.items()}
        out = []
        if res.boxes is not None:
            for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy().astype(int)):
                out.append((names.get(int(k), str(k)), b, float(c)))
        return out

    def update(self, t, frame, patient, scale):
        """Returns events: 'intake' (a cup / bowl held at the mouth for >= 1 s)."""
        events = []
        # search fast for 30s, then slow down. used to search 3x/s forever when there's no bed in
        # view and ate a third of the cpu
        self.first_t = t if self.first_t is None else self.first_t
        if len(self.bed_boxes) >= 3:
            every = max(5.0, self.cfg.scene_every_s)
        elif t - self.first_t < 30:
            every = self.cfg.scene_every_s / 3
        else:
            every = 2 * self.cfg.scene_every_s
        if t - self.last_run >= every:
            self.last_run = t
            self.objects = self._run_model(frame)
            beds = [o for o in self.objects if o[0] == "bed" and o[2] >= 0.35]
            if beds:
                self.bed_boxes.append(max(beds, key=lambda o: _area(o[1]))[1])
            self.near_mouth = self._at_mouth(patient, scale)
        # a cup at the mouth: an object box overlapping the mouth area for >= 1 s (two detector runs)
        if self.near_mouth:
            self.near_mouth_since = self.near_mouth_since or t
            if t - self.near_mouth_since >= 1.0 and t - self.intake_logged_at > 60:
                self.intake_logged_at = t
                events.append("intake")
        else:
            self.near_mouth_since = None
        return events

    def _at_mouth(self, patient, scale):
        if patient is None or scale is None:
            return None
        nose = kp(patient, NOSE, self.cfg.kp_conf)
        if nose is None:
            return None
        mouth = nose + np.array([0, 0.12 * scale])
        for name, box, conf in self.objects:
            if name not in DRINK + FOOD:
                continue
            x1, y1, x2, y2 = box
            pad = 0.15 * scale
            if x1 - pad <= mouth[0] <= x2 + pad and y1 - pad <= mouth[1] <= y2 + pad:
                return "drink" if name in DRINK else "food"
        return None

    def bed_zone(self, shape):
        """Normalised bed box (median of recent detections), or None until the bed was seen 3 times."""
        if len(self.bed_boxes) < 3:
            return None
        h, w = shape[:2]
        x1, y1, x2, y2 = np.median(np.array(self.bed_boxes), axis=0)
        bw, bh = x2 - x1, y2 - y1
        # a little margin: a patient lying on the bed sticks out above the mattress outline
        z = (x1 - 0.05 * bw, y1 - 0.25 * bh, x2 + 0.05 * bw, y2 + 0.05 * bh)
        return tuple(float(np.clip(v / s, 0, 1)) for v, s in zip(z, (w, h, w, h)))

    def overlay(self):
        """Objects worth drawing on the dashboard view."""
        return [(n, b, c) for n, b, c in self.objects if n != "dining table"]


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])
