"""What the camera view looks like on the dashboard. Frames are never stored.

skeleton - black background, only the body lines (default for the privacy demo)
blur     - heavily pixelated image + skeleton
raw      - real image + skeleton (setup / debugging only)
"""
import cv2
import numpy as np

from .pose import SKELETON

MODES = ("skeleton", "blur", "raw")

COLORS = {"green": (94, 197, 34), "yellow": (21, 184, 245), "red": (68, 68, 239), "grey": (160, 160, 160)}


ZONE_COLORS = {"door": (230, 160, 60), "bathroom": (200, 120, 220), "sanitizer": (120, 220, 120)}
OBJECT_COLOR = (0, 200, 255)


def render(frame, mode, patient=None, others=(), chest=None, face=None, zone=None,
           level="grey", kp_thr=0.4, objects=None, zones=None, exit_score=None, head=None,
           hud=None, label=None, marks=None):
    out = _render(frame, mode, patient, others, chest, face, zone, level, kp_thr)
    h, w = out.shape[:2]
    for x, y, text in marks or ():
        # joint angle (degrees) next to the joint - physiotherapy goniometer
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 230, 255), 1, cv2.LINE_AA)
    if hud:
        # live bar: red dot + LIVE + frame rate / delay - shows the AI working in real time
        bar = out[0:22, :].copy()
        cv2.rectangle(bar, (0, 0), (w, 22), (0, 0, 0), -1)
        out[0:22, :] = cv2.addWeighted(out[0:22, :], 0.35, bar, 0.65, 0)
        cv2.circle(out, (12, 11), 5, (60, 60, 235), -1, cv2.LINE_AA)
        cv2.putText(out, "LIVE", (22, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, hud, (66, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 225, 235), 1, cv2.LINE_AA)
    if label and patient is not None:
        x1, y1 = int(patient.box[0]), int(patient.box[1])
        col = COLORS.get(level, COLORS["grey"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(36, y1 - 14)
        cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 8, y + 3), col, -1)
        cv2.putText(out, label, (x1 + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    for name, (x1, y1, x2, y2) in (zones or {}).items():
        c = ZONE_COLORS.get(name, (200, 200, 200))
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, c, 1, cv2.LINE_AA)
        _label(out, name, (p1[0] + 3, p1[1] + 14), c)
    for name, (x1, y1, x2, y2), conf in objects or ():
        if name == "bed" or mode != "skeleton":           # furniture outlines show even in skeleton mode
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, OBJECT_COLOR, 1, cv2.LINE_AA)
            _label(out, f"{name} {conf:.0%}", (p1[0] + 3, max(12, p1[1] - 4)), OBJECT_COLOR)
    if head is not None and mode != "raw":
        cv2.rectangle(out, head[:2], head[2:], (255, 120, 200), 1, cv2.LINE_AA)
    if patient is not None:
        x1, y1, x2, y2 = (int(v) for v in patient.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS.get(level, COLORS["grey"]), 1, cv2.LINE_AA)
        if exit_score:
            # bed-exit risk bar above the patient
            bw = max(40, x2 - x1)
            cv2.rectangle(out, (x1, y1 - 10), (x1 + bw, y1 - 4), (60, 60, 60), -1)
            col = (68, 68, 239) if exit_score >= 0.55 else (21, 184, 245) if exit_score >= 0.3 else (94, 197, 34)
            cv2.rectangle(out, (x1, y1 - 10), (x1 + int(bw * min(1, exit_score)), y1 - 4), col, -1)
    return out


def _label(img, text, org, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _render(frame, mode, patient, others, chest, face, zone, level, kp_thr):
    h, w = frame.shape[:2]
    if mode == "raw":
        out = frame.copy()
    elif mode == "blur":
        small = cv2.resize(frame, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
        # darken the tiny copy, not the full frame: float maths on every full frame took ~20 % of
        # the app's time (py-spy on the live camera)
        small = cv2.convertScaleAbs(small, alpha=0.6)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        out = np.zeros_like(frame)
        out[:] = (24, 20, 18)

    if zone is not None:
        x1, y1, x2, y2 = zone
        cv2.rectangle(out, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (200, 160, 90), 1, cv2.LINE_AA)
    for p in others:
        _skeleton(out, p, (150, 150, 150), kp_thr)
    if patient is not None:
        _skeleton(out, patient, COLORS.get(level, COLORS["grey"]), kp_thr)
    if chest is not None:
        cv2.rectangle(out, chest[:2], chest[2:], (255, 200, 0), 1, cv2.LINE_AA)
    if face is not None and mode != "raw":
        cv2.rectangle(out, face[:2], face[2:], (200, 120, 255), 1, cv2.LINE_AA)
    return out


def _skeleton(img, person, color, thr):
    pts, conf = person.kps, person.conf
    for a, b in SKELETON:
        if conf[a] >= thr and conf[b] >= thr:
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, 3, cv2.LINE_AA)
    for i in range(len(pts)):
        if conf[i] >= thr:
            cv2.circle(img, tuple(int(v) for v in pts[i]), 4, (255, 255, 255), -1, cv2.LINE_AA)
