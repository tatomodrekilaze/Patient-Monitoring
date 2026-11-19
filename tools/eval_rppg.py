"""Evaluate camera heart rate (rPPG) against a pulse oximeter: UBFC-rPPG DATASET_2
(Bobbia et al. 2019; 640x480 30 fps videos with a synchronised finger PPG).

    python -I tools/eval_rppg.py <ubfc_dir> [subject ...]

<ubfc_dir>/<subject>/vid.avi + ground_truth.txt (line 1 PPG, line 2 HR bpm, line 3 time s).
The face region comes from the same pose model as in the app. Every second the estimate is compared
with the oximeter's mean heart rate over the same window: mean absolute error, share within 5 bpm.
(HRV from the camera was tried and dropped: RMSSD error 31 ms when RMSSD itself is 20-60 ms.)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from monitor.config import make_config  # noqa: E402
from monitor.pose import PatientSelector, PoseTracker, face_box  # noqa: E402
from monitor.signals.rppg import HeartRateEstimator  # noqa: E402


def load_gt(path):
    rows = [np.array(line.split(), float) for line in open(path) if line.strip()]
    return rows[0], rows[1], rows[2]           # ppg, hr, time


def run(subject_dir, cfg, pose):
    ppg, hr_gt, tg = load_gt(os.path.join(subject_dir, "ground_truth.txt"))
    cap = cv2.VideoCapture(os.path.join(subject_dir, "vid.avi"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    est = HeartRateEstimator(cfg)
    sel = PatientSelector()
    pose.reset()
    pairs, i, last = [], 0, 0.0
    box = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if i % 3 == 1 or box is None:                 # pose every 3rd frame is plenty for a still face
            p, _ = sel.select(pose.infer(frame), t)
            box = face_box(p, cfg.kp_conf, frame.shape) if p is not None else None
        est.update(t, frame, box, True)
        if t - last >= 1.0:
            last = t
            est.estimate(t)
            if est.hr is not None and est.quality > cfg.hr_quality:
                m = (tg >= t - cfg.hr_window_s) & (tg <= t)
                if m.any():
                    pairs.append((est.hr, float(np.mean(hr_gt[m]))))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ubfc")
    ap.add_argument("subjects", nargs="*")
    a = ap.parse_args()
    cfg = make_config("demo")
    pose = PoseTracker(cfg.pose_model, cfg.imgsz, cfg.pose_conf, cfg.torch_threads)
    subs = a.subjects or sorted(d for d in os.listdir(a.ubfc) if os.path.isfile(os.path.join(a.ubfc, d, "vid.avi")))
    allp = []
    for s in subs:
        pairs = run(os.path.join(a.ubfc, s), cfg, pose)
        allp += pairs
        if not pairs:
            print(f"{s:10s} no confident estimate")
            continue
        e = np.array([p[0] - p[1] for p in pairs])
        print(f"{s:10s} readings {len(pairs):3d}  MAE {np.mean(np.abs(e)):5.1f} bpm  within 5 bpm {np.mean(np.abs(e) <= 5):4.0%}"
              f"  true HR {np.mean([p[1] for p in pairs]):5.1f}")
    if allp:
        e = np.array([p[0] - p[1] for p in allp])
        r = np.corrcoef([p[0] for p in allp], [p[1] for p in allp])[0, 1]
        print(f"\nALL readings {len(allp)}  MAE {np.mean(np.abs(e)):.1f} bpm  RMSE {np.sqrt(np.mean(e ** 2)):.1f}"
              f"  within 5 bpm {np.mean(np.abs(e) <= 5):.0%}  r={r:.2f}")


if __name__ == "__main__":
    main()
