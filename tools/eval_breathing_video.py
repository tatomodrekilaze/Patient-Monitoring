"""Run the breathing estimator on a real video with a fixed chest region (no pose needed).

    python -I tools/eval_breathing_video.py video.mov [--roi 0.2,0.2,0.8,0.8]

Prints the breathing rate every 5 s, signal quality, and any apnea alarm. Used with MPSC-RR
(CC BY 4.0) chest close-ups, which have regular, slow and fast breathing but no published
per-video breath counts - so this checks plausibility and false apnea, not exact accuracy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from monitor.config import make_config  # noqa: E402
from monitor.signals.breathing import BreathingEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--roi", default="0.15,0.15,0.85,0.85")
    ap.add_argument("--process-fps", type=float, default=15.0, help="simulate the live processing rate")
    a = ap.parse_args()
    cfg = make_config("demo")
    est = BreathingEstimator(cfg)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / a.process_fps))
    rx1, ry1, rx2, ry2 = (float(v) for v in a.roi.split(","))
    i, rows, apnea_alarms, in_apnea = 0, [], 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            est.update(t, gray, (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)), moving=False)
            if i % int(fps / 2) == 0:
                est.estimate(t)
                ap_s = est.apnea_seconds(t)
                if ap_s >= cfg.apnea_s and not in_apnea:
                    apnea_alarms += 1
                    print(f"   !! apnea alarm at {t:.0f}s (no breath for {ap_s:.0f}s)")
                in_apnea = ap_s >= cfg.apnea_s
            if i % int(fps * 5) == 0 and est.rr is not None:
                rows.append((t, est.rr, est.quality))
                irr = est.irregularity(t)
                print(f"{t:6.0f}s  RR {est.rr:5.1f}/min  quality {est.quality:4.2f}"
                      + (f"  irregularity {irr:.2f}" if irr is not None else ""))
        i += 1
    rr = np.array([r[1] for r in rows])
    q = np.array([r[2] for r in rows])
    print(f"\n{os.path.basename(a.video)}: {len(rows)} readings, RR range {rr.min():.1f}-{rr.max():.1f}, "
          f"median {np.median(rr):.1f}, good-quality readings {100 * np.mean(q >= 0.25):.0f}%, "
          f"apnea alarms {apnea_alarms}")


if __name__ == "__main__":
    main()
