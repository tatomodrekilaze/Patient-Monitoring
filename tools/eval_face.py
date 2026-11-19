"""Evaluate the facial-asymmetry (stroke FAST) check on public face videos: palsynet-data
(huggingface.co/datasets/jasir/palsynet-data, CC-BY-4.0; 300x300 face videos from YouTube,
data/affected = Bell's palsy, data/unaffected = healthy).

    python -I tools/eval_face.py <palsynet_dir>

1. False alarms: each healthy video is played through the app's FaceAnalyzer at 2 readings / s (the
   patient's baseline is learned from the start of the video, as in the app). Healthy people talking,
   turning and nodding must give no 'face_asymmetry'. Run with the view gates on and off.
2. Size of a real droop: median |droop| on usable frames, palsy vs healthy faces. A new palsy has to
   move the droop by more than the alert threshold for the alert to be useful.
The videos are already face crops, so the "too close to the lens" gate does not apply here.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from monitor.config import make_config  # noqa: E402
from monitor.events.face import FaceAnalyzer, droop, head_pose, whole_face  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def readings(path, fa, every=0.5):
    """[(t, landmarks, blendshapes, whole)] every `every` seconds of the video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out, i, nxt = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t < nxt:
            continue
        nxt = t + every
        rgb = np.ascontiguousarray(cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC)[:, :, ::-1])
        r = fa._detect(rgb)
        if r is not None:
            out.append((t, r[0], r[1], whole_face(r[0], rgb.shape)))
    return out


def alerts(rs, mode, gates):
    # gates off = the old behaviour: every frame trusted, any head pose
    cfg = make_config(mode) if gates else make_config(mode, face_max_yaw=9.0, face_max_dpitch=9.0)
    fa = FaceAnalyzer(cfg, landmarker=lambda x: None)
    n = 0
    for t, lm, bs, whole in rs:
        n += fa._analyse(t, lm, bs, trusted=whole or not gates).count("face_asymmetry")
    return n


def usable(rs, cfg):
    """Droops of frames the app would measure: whole face, facing the camera, eyes open, no smile."""
    d = []
    for _, lm, bs, whole in rs:
        yaw, _ = head_pose(lm)
        if (whole and abs(yaw) <= cfg.face_max_yaw and (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2 < 0.5
                and bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0) < 0.6):
            d.append(droop(lm))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("palsynet")
    a = ap.parse_args()
    fa = FaceAnalyzer(make_config("demo"), os.path.join(ROOT, "face_landmarker.task"))
    data = {}
    for group in ("unaffected", "affected"):
        for f in sorted(glob.glob(os.path.join(a.palsynet, "data", group, "*.mp4")),
                        key=lambda p: int(os.path.basename(p)[:-4])):
            data[(group, os.path.basename(f))] = readings(f, fa)
            print(f"  read {group}/{os.path.basename(f)}: {len(data[(group, os.path.basename(f))])} faces", flush=True)

    print("\n1. False alarms on healthy faces (talking, turning, nodding)")
    healthy = {k: v for k, v in data.items() if k[0] == "unaffected"}
    minutes = sum(v[-1][0] for v in healthy.values() if v) / 60
    for mode in ("demo", "clinical"):
        cfg = make_config(mode)
        for gates in (False, True):
            per = [alerts(v, mode, gates) for v in healthy.values()]
            n, vids = sum(per), sum(x > 0 for x in per)
            print(f"   {mode:8s} hold {cfg.asym_hold_s:3.0f}s  view gates {'ON ' if gates else 'off'}: "
                  f"{n} alerts in {minutes:.0f} min, {vids}/{len(healthy)} people")

    print("\n2. Droop size on usable frames (inter-ocular distances)")
    cfg = make_config("demo")
    med = {"unaffected": [], "affected": []}
    for (g, name), rs in data.items():
        d = usable(rs, cfg)
        if len(d) >= 5:
            med[g].append(abs(float(np.median(d))))
    for g, v in med.items():
        v = np.array(v)
        print(f"   {g:10s} n={len(v):2d}  median {np.median(v):.3f}  p25 {np.percentile(v, 25):.3f}  p75 {np.percentile(v, 75):.3f}")
    h, p = np.array(med["unaffected"]), np.array(med["affected"])
    auc = np.mean([[1.0 if x > y else 0.5 if x == y else 0.0 for y in h] for x in p])
    print(f"   palsy vs healthy AUC {auc:.2f}; palsy faces with |droop| >= threshold {cfg.asym_thr}: "
          f"{np.mean(p >= cfg.asym_thr):.0%}  (healthy: {np.mean(h >= cfg.asym_thr):.0%})")


if __name__ == "__main__":
    main()
