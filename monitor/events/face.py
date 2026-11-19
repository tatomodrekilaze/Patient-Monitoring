"""face stuff on a head crop, mediapipe face landmarker (478 pts + blendshapes), ~2x/s

eyes open/closed is what's actually used (sleep). pain score and passive droop are still computed
but don't alert anymore - both were too unreliable (pain fired on anger/disgust, passive droop
caught 54% of palsy cases). the guided face test (neuro.py) uses the droop numbers.
"""
from collections import deque

import numpy as np

from ..pose import L_EAR, L_EYE, NOSE, R_EAR, R_EYE, kp

# face mesh ids: eye outer corners, mouth corners, nose tip, cheek edges, chin, forehead
EYE_A, EYE_B, MOUTH_A, MOUTH_B = 33, 263, 61, 291
NOSE_TIP, CHEEK_A, CHEEK_B, CHIN, FOREHEAD = 1, 234, 454, 152, 10


def pain_score(bs):
    """0-10. textbook AU4+AU6/7+AU9/10 scored happy faces highest lol, so brow lowering
    gets most weight and a smile cancels it"""
    m = lambda a, b: (bs.get(a, 0.0) + bs.get(b, 0.0)) / 2          # noqa: E731
    au4 = m("browDownLeft", "browDownRight")
    au7 = m("eyeSquintLeft", "eyeSquintRight")
    au910 = max(m("noseSneerLeft", "noseSneerRight"), m("mouthUpperUpLeft", "mouthUpperUpRight"))
    smile = m("mouthSmileLeft", "mouthSmileRight")
    s = 0.6 * min(1.0, au4 / 0.6) + 0.25 * min(1.0, max(0.0, au7 - 0.35) / 0.35) + 0.15 * min(1.0, au910 / 0.4)
    return float(10.0 * s * max(0.0, 1.0 - smile))


def droop(lm):
    """Signed mouth-corner droop relative to the eye line, in inter-ocular distances.
    lm: (478, 2) points. 0 = symmetric; independent of head roll (measured along the eye line)."""
    a, b = lm[EYE_A], lm[EYE_B]
    axis = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    normal = np.array([-axis[1], axis[0]])
    iod = np.linalg.norm(b - a) + 1e-9
    da = np.dot(lm[MOUTH_A] - a, normal)
    db = np.dot(lm[MOUTH_B] - b, normal)
    return float((da - db) / iod)


def head_pose(lm):
    """(yaw, pitch) proxies from the mesh, roll-independent. yaw: nose offset between the cheek edges,
    -1..1, 0 = facing the camera (|0.1| ~ 10 deg). pitch: nose tip height between eye line and chin,
    0..1; compared with the patient's own baseline, since faces differ."""
    a, b = lm[EYE_A], lm[EYE_B]
    axis = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    normal = np.array([-axis[1], axis[0]])
    la = np.dot(lm[NOSE_TIP] - lm[CHEEK_A], axis)
    lb = np.dot(lm[CHEEK_B] - lm[NOSE_TIP], axis)
    yaw = float((la - lb) / (la + lb + 1e-9))
    eye = (a + b) / 2
    pitch = float(np.dot(lm[NOSE_TIP] - eye, normal) / (np.dot(lm[CHIN] - eye, normal) + 1e-9))
    return yaw, pitch


def whole_face(lm, shape, margin=2):
    """All of the face's outline inside the picture: a face cut by the frame edge gets extrapolated
    landmarks, and a droop measured on invented points is noise."""
    h, w = shape[:2]
    pts = lm[[FOREHEAD, CHIN, CHEEK_A, CHEEK_B, MOUTH_A, MOUTH_B]]
    return bool((pts[:, 0] >= margin).all() and (pts[:, 0] <= w - margin).all()
                and (pts[:, 1] >= margin).all() and (pts[:, 1] <= h - margin).all())


class FaceAnalyzer:
    def __init__(self, cfg, model="face_landmarker.task", landmarker=None):
        self.cfg = cfg
        self._model_path = model
        self._landmarker = landmarker        # test hook: rgb crop -> (landmarks (478,2) px, blendshapes dict) or None
        self.hi = None                       # (full-resolution frame, scale) for this frame, set by the bed
        self.reset()

    def reset(self):
        self.last_run = -1e9
        self.pain = None
        self.pain_hist = deque()
        self.pain_rest = deque(maxlen=600)    # ~5 min of readings: the patient's usual face
        self.pain_since = None
        self.pain_fired = False
        self.eyes_open = None
        self.eye_hist = deque()
        self.droops = deque(maxlen=600)
        self.base_droop = None
        self.base_pitch = None
        self.asym = None
        self.asym_since = None
        self.asym_fired = False
        self.seen = 0
        self.last = None

    def _detect(self, rgb):
        if self._landmarker is not None:
            return self._landmarker(rgb)
        if not hasattr(self, "_mp"):
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            # load as bytes, mediapipe on windows breaks some absolute paths
            with open(self._model_path, "rb") as f:
                model = f.read()
            opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=model),
                                                output_face_blendshapes=True, num_faces=1)
            self._mp = (mp, vision.FaceLandmarker.create_from_options(opts))
        mp, lmk = self._mp
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        return pts, bs

    @staticmethod
    def head_box(person, thr, shape):
        nose = kp(person, NOSE, thr)
        if nose is None:
            return None
        pts = [p for p in (kp(person, k, thr) for k in (L_EYE, R_EYE, L_EAR, R_EAR)) if p is not None]
        if not pts:
            return None
        size = max(float(np.max(np.linalg.norm(np.array(pts) - nose, axis=1))) * 3.2, 48)
        h, w = shape[:2]
        x1, y1 = int(max(0, nose[0] - size / 2)), int(max(0, nose[1] - size / 2))
        x2, y2 = int(min(w, nose[0] + size / 2)), int(min(h, nose[1] + size / 2))
        return (x1, y1, x2, y2) if x2 - x1 >= 40 and y2 - y1 >= 40 else None

    def update(self, t, frame, person):
        """Returns events: 'pain_face', 'face_asymmetry'."""
        if person is None or t - self.last_run < self.cfg.face_every_s:
            return []
        self.last_run = t
        box = self.head_box(person, self.cfg.kp_conf, frame.shape)
        if box is None:
            return []
        x1, y1, x2, y2 = box
        # from the full-resolution picture when there is one: expressions and the mouth-corner
        # droop are a few pixels in a 640 px frame (landmarks only scale with the crop, so the
        # droop, measured in eye distances, is unchanged)
        src, k = self.hi if self.hi is not None else (frame, 1.0)
        crop = np.ascontiguousarray(src[int(y1 * k):int(y2 * k), int(x1 * k):int(x2 * k), ::-1])
        if max(crop.shape[:2]) < 192:          # the landmarker needs a reasonably large face
            import cv2
            s = 192 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        r = self._detect(crop)
        if r is None:
            return []
        lm, bs = r
        self.seen += 1
        h, w = frame.shape[:2]
        # face cut by the camera's frame edge (too close, half out of view): measured, not trusted
        clipped = x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h or not whole_face(lm, crop.shape)
        # face very close to the lens: perspective alone moves one mouth corner (tested on a 3-D face)
        width = np.linalg.norm(lm[CHEEK_B] - lm[CHEEK_A]) / crop.shape[1] * (x2 - x1) / w
        return self._analyse(t, lm, bs, trusted=not clipped and width <= self.cfg.face_max_width)

    def _analyse(self, t, lm, bs, trusted=True):
        events = []
        self.last = (t, lm, bs, trusted)             # for the guided FAST face test
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        self.eye_hist.append((t, blink < 0.5))
        while self.eye_hist and self.eye_hist[0][0] < t - 10:
            self.eye_hist.popleft()
        self.eyes_open = sum(o for _, o in self.eye_hist) >= len(self.eye_hist) / 2

        # pain: smoothed over ~3 s; a single grimace frame is not pain
        self.pain_hist.append((t, pain_score(bs)))
        while self.pain_hist and self.pain_hist[0][0] < t - 3:
            self.pain_hist.popleft()
        self.pain = float(np.median([p for _, p in self.pain_hist]))
        # relative to this patient's resting face: some people's brows are always low
        self.pain_rest.append(self.pain)
        rest = float(np.percentile(self.pain_rest, 25)) if len(self.pain_rest) >= 10 else 0.0
        if self.pain >= self.cfg.pain_thr and self.pain - rest >= 2.5:
            self.pain_since = self.pain_since or t
            if not self.pain_fired and t - self.pain_since >= self.cfg.pain_hold_s:
                self.pain_fired = True
                events.append("pain_face")
        else:
            self.pain_since = None
            if self.pain < 0.6 * self.cfg.pain_thr:
                self.pain_fired = False

        # asymmetry: a 2-D droop changes with head pose (turned, tilted, seen from below), so only
        # frames that face the camera, show the whole face and match the baseline's head pitch count;
        # eyes open and a fairly neutral face (a lopsided smile is not a droop)
        yaw, pitch = head_pose(lm)
        ok = (trusted and blink < 0.5 and abs(yaw) <= self.cfg.face_max_yaw
              and bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0) < 0.6
              and (self.base_pitch is None or abs(pitch - self.base_pitch) <= self.cfg.face_max_dpitch))
        if ok:
            self.droops.append((t, droop(lm), pitch))
            if self.base_droop is None and len(self.droops) >= self.cfg.face_baseline_n:
                self.base_droop = float(np.median([d for _, d, _ in self.droops]))
                self.base_pitch = float(np.median([p for _, _, p in self.droops]))
        recent = [d for tt, d, _ in self.droops if tt >= t - 10]
        # enough good frames in the last 10 s (at 2/s: >= 12), all telling the same story
        if self.base_droop is not None and len(recent) >= 0.6 * 10 / self.cfg.face_every_s:
            self.asym = float(np.median(recent) - self.base_droop)
            spread = float(np.subtract(*np.percentile(recent, [75, 25])))
            if abs(self.asym) >= self.cfg.asym_thr and spread < abs(self.asym):
                self.asym_since = self.asym_since or t
                if not self.asym_fired and t - self.asym_since >= self.cfg.asym_hold_s:
                    self.asym_fired = True
                    events.append("face_asymmetry")
            else:
                self.asym_since = None
        else:
            self.asym = None
            self.asym_since = None              # not seen well: the hold starts again
        return events
