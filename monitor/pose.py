"""YOLO pose estimation + ByteTrack identity, and keypoint geometry helpers."""
from dataclasses import dataclass

import numpy as np

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

SKELETON = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN),
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
]


@dataclass
class Person:
    track_id: int
    box: np.ndarray      # x1, y1, x2, y2
    kps: np.ndarray      # (17, 2) pixels
    conf: np.ndarray     # (17,)
    score: float

    @property
    def center(self):
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


class PoseTracker:
    def __init__(self, model="yolo11n-pose.pt", imgsz=320, conf=0.3, threads=4):
        import torch
        from ultralytics import YOLO

        from .config import model_path
        torch.set_num_threads(threads)
        self.model_file = model_path(model, imgsz)
        self.model = YOLO(self.model_file, task="pose")
        self.imgsz = imgsz
        self.conf = conf

    def infer(self, frame):
        res = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            imgsz=self.imgsz, conf=self.conf, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id
        # ByteTrack may not have confirmed a new track yet: give it a temporary negative id
        ids = ids.int().cpu().numpy() if ids is not None else -(np.arange(len(boxes)) + 1)
        kxy = res.keypoints.xy.cpu().numpy()
        kc = res.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kxy.shape[:2])
        return [Person(int(ids[i]), boxes[i], kxy[i], kc[i], float(scores[i])) for i in range(len(boxes))]

    def reset(self):
        predictor = getattr(self.model, "predictor", None)
        for tr in getattr(predictor, "trackers", None) or []:
            tr.reset()


class PatientSelector:
    """Keeps following the same patient through ByteTrack id changes.

    Tested on UR Fall: when someone falls the tracker often loses them and issues a new id, and a
    naive "nearest person" rule then jumps to a visitor who was in view all along. So people seen
    *at the same time as* the patient are never taken for the patient (unless the patient has been
    gone for a while); a newly appeared person near the patient's last position is preferred.
    """

    def __init__(self, give_up_s=5.0):
        self.give_up_s = give_up_s
        self.reset()

    def reset(self):
        self.patient_id = None
        self.center = None
        self.size = None
        self.last_seen = None
        self.others = set()           # ids seen alongside the patient

    def select(self, people, t):
        if not people:
            return None, []
        p = next((x for x in people if x.track_id == self.patient_id), None)
        if p is None:
            gone_long = self.last_seen is None or t - self.last_seen > self.give_up_s
            fresh = [x for x in people if x.track_id not in self.others]
            pool = fresh if (fresh and not gone_long) else people      # a preference, not a hard rule
            if self.center is None:
                p = max(pool, key=_area)
            else:
                p = min(pool, key=lambda x: np.linalg.norm(x.center - self.center))
            if self.patient_id is not None and p.track_id != self.patient_id:
                self.others.discard(p.track_id)
            self.patient_id = p.track_id
        self.center, self.last_seen = p.center, t
        others = [x for x in people if x is not p]
        self.others.update(x.track_id for x in others)
        return p, others


def _area(p):
    return (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])


# ---------------- geometry ----------------

def kp(person, idx, thr):
    return person.kps[idx] if person.conf[idx] >= thr else None


def mid(person, a, b, thr):
    pts = [p for p in (kp(person, a, thr), kp(person, b, thr)) if p is not None]
    return np.mean(pts, axis=0) if pts else None


def torso(person, thr):
    """(shoulder_mid, hip_mid, torso_length) or None."""
    sh = mid(person, L_SH, R_SH, thr)
    hip = mid(person, L_HIP, R_HIP, thr)
    if sh is None or hip is None:
        return None
    length = float(np.linalg.norm(hip - sh))
    return (sh, hip, length) if length > 5 else None


def torso_view(person, shape, thr=0.5):
    """None if the torso is really visible, else a reason string.
    yolo still puts hips at the frame edge when the body is cut off -> chest box on the face -> fake apnea"""
    h, w = shape[:2]
    # hips under a blanket are less sure: they may be, as long as they are well inside the frame
    hip_thr = max(0.25, thr - 0.25)
    for k in (L_SH, R_SH, L_HIP, R_HIP):
        if person.conf[k] < (thr if k in (L_SH, R_SH) else hip_thr):
            return "torso_hidden"
        x, y = person.kps[k]
        if not (0.02 * w < x < 0.98 * w and 0.02 * h < y < 0.97 * h):
            return "cut_off"
    tr = torso(person, hip_thr)
    if tr is None:
        return "torso_hidden"
    if tr[2] < 0.08 * max(h, w):
        return "too_far"
    if tr[2] > 0.75 * h and tr[2] > 0.5 * w:
        return "too_close"
    return None


def angle_from_vertical(v):
    return float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))


def chest_box(person, thr, frame_shape):
    """Upper-torso box for breathing motion: shoulders down to 40 % of the way to the hips."""
    h, w = frame_shape[:2]
    shs = [p for p in (kp(person, L_SH, thr), kp(person, R_SH, thr)) if p is not None]
    if not shs:
        return None
    t = torso(person, thr)
    if t is not None:
        sh, hip, length = t
        pts = shs + [s + 0.4 * (hip - sh) for s in shs]
        pad = 0.2 * length
    else:
        if len(shs) < 2:
            return None
        width = float(np.linalg.norm(shs[0] - shs[1]))
        pts = shs + [s + np.array([0, 0.8 * width]) for s in shs]
        pad = 0.2 * width
    pts = np.array(pts)
    x1, y1 = pts.min(0) - pad
    x2, y2 = pts.max(0) + pad
    return clip_box((x1, y1, x2, y2), w, h, min_size=24)


def abdomen_box(person, thr, frame_shape):
    """Belly patch: from 55 % of the way shoulders -> hips down to the hips, about as wide as the
    shoulders. Its motion against the chest's tells central from obstructive breathing problems."""
    h, w = frame_shape[:2]
    t = torso(person, thr)
    if t is None:
        return None
    sh, hip, length = t
    ls, rs = kp(person, L_SH, thr), kp(person, R_SH, thr)
    half = 0.5 * float(np.linalg.norm(ls - rs)) if ls is not None and rs is not None else 0.3 * length
    half = max(half, 0.2 * length)
    top, bottom = sh + 0.55 * (hip - sh), hip - 0.05 * (hip - sh)
    axis = (hip - sh) / (length + 1e-9)
    perp = np.array([-axis[1], axis[0]]) * half
    pts = np.array([top + perp, top - perp, bottom + perp, bottom - perp])
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    return clip_box((x1, y1, x2, y2), w, h, min_size=16)


def face_box(person, thr, frame_shape):
    """Cheek / under-eye skin patch for rPPG."""
    h, w = frame_shape[:2]
    nose = kp(person, NOSE, thr)
    le, re = kp(person, L_EYE, thr), kp(person, R_EYE, thr)
    if nose is None or le is None or re is None:
        return None
    eye_d = float(np.linalg.norm(le - re))
    if eye_d < 8:
        return None
    eye_y = (le[1] + re[1]) / 2
    cx = (le[0] + re[0] + nose[0]) / 3
    x1, x2 = cx - 0.9 * eye_d, cx + 0.9 * eye_d
    y1, y2 = eye_y + 0.15 * eye_d, nose[1] + 0.35 * eye_d
    return clip_box((x1, y1, x2, y2), w, h, min_size=10)


def clip_box(box, w, h, min_size=1):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return x1, y1, x2, y2
