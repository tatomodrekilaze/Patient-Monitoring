"""Evaluate fall / posture detection on the UR Fall Detection Dataset (Kwolek & Kepski 2014, CC BY-NC-SA 4.0).

    python -I tools/eval_urfall.py <dataset_dir> [--dark 0.25] [--cache <dir>]

<dataset_dir> holds fall-NN-cam0-rgb/ and adl-NN-cam0-rgb/ PNG folders (30 fps).
A fall sequence passes if a 'fall' event fires; a daily-activity (ADL) sequence passes if none does.
--dark scales brightness (e.g. 0.25) to test night mode on the same data.
"""
import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from monitor.bed import enhance_low_light  # noqa: E402
from monitor.config import make_config  # noqa: E402
from monitor.events.posture import PostureTracker, RoomFallWatch  # noqa: E402
from monitor.pose import PatientSelector, PoseTracker  # noqa: E402
from monitor.smooth import KeypointSmoother  # noqa: E402

FPS = 30.0


def detect(folder, pose, dark=None, cfg=None):
    """[(people, shape)] per frame: the pose model's output, cached so rule changes can be re-checked
    on all sequences without the images."""
    pose.reset()
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True)):
        img = cv2.imread(f)
        if dark:
            img = (img * dark).astype(np.uint8)
            if cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < cfg.night_thr:
                img = enhance_low_light(img)
        out.append((pose.infer(img), img.shape))
    return out


def run_sequence(frames, cfg, smooth=False):
    posture = PostureTracker(cfg)
    room = RoomFallWatch(cfg)
    events, states, people_seen = [], [], 0
    selector = PatientSelector()             # same patient-following logic as the live system
    smoother = KeypointSmoother(thr=cfg.kp_conf) if smooth else None
    for i, (people, shape) in enumerate(frames):
        t = i / FPS
        if smoother:
            people = smoother(t, people)
        person, others = selector.select(people, t)
        people_seen += person is not None
        for ev in posture.update(t, person, shape):
            events.append((round(t, 2), ev))
        if room.update(t, others):
            events.append((round(t, 2), "fall"))
        states.append(posture.state)
    return {"frames": len(frames), "detected_pct": round(100 * people_seen / max(1, len(frames))),
            "events": events, "final": states[-1] if states else None,
            "states": sorted(set(states))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--dark", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--model", default=None, help="pose model, e.g. yolo11s-pose.pt")
    ap.add_argument("--pytorch", action="store_true", help="do not use the OpenVINO copy")
    ap.add_argument("--smooth", action="store_true", help="One-Euro keypoint smoothing")
    ap.add_argument("--cache", default=None, help="folder for / with cached pose results (*.pkl)")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="repeat the last frame for this many seconds: the clips end < 1 s after the "
                         "impact, a real patient stays on the floor (reported separately)")
    a = ap.parse_args()
    cfg = make_config("demo")
    if a.pytorch:
        os.environ["SEW_NO_OPENVINO"] = "1"
    pose = PoseTracker(a.model or cfg.pose_model, a.imgsz, cfg.pose_conf, cfg.torch_threads)
    print("model:", os.path.basename(pose.model_file), "imgsz", a.imgsz, "smoothing", a.smooth)
    seqs = sorted(d for d in glob.glob(os.path.join(a.dataset, "*-cam0-rgb")) if os.path.isdir(d))
    if a.cache:                       # sequences only in the cache (images already deleted) count too
        os.makedirs(a.cache, exist_ok=True)
        have = {os.path.basename(x) for x in seqs}
        seqs += sorted(os.path.join(a.dataset, os.path.basename(c)[:-4]) for c in glob.glob(os.path.join(a.cache, "*.pkl"))
                       if os.path.basename(c)[:-4] not in have)
        seqs.sort()
    ok_falls = n_falls = ok_adl = n_adl = 0
    for s in seqs:
        name = os.path.basename(s).replace("-cam0-rgb", "")
        cached = os.path.join(a.cache, os.path.basename(s) + ".pkl") if a.cache and not a.dark else None
        if cached and os.path.exists(cached):
            frames = pickle.load(open(cached, "rb"))
        else:
            frames = detect(s, pose, a.dark, cfg)
            if cached:
                pickle.dump(frames, open(cached, "wb"))
        if a.hold and frames:
            frames = frames + [frames[-1]] * int(a.hold * FPS)
        r = run_sequence(frames, cfg, a.smooth)
        fired = any(ev == "fall" for _, ev in r["events"])
        if name.startswith("fall"):
            n_falls += 1
            ok_falls += fired
            verdict = "PASS" if fired else "MISS"
        else:
            n_adl += 1
            ok_adl += not fired
            verdict = "PASS" if not fired else "FALSE ALARM"
        print(f"{name:8s} {verdict:11s} person {r['detected_pct']:3d}%  final={r['final']:9s} "
              f"states={r['states']} events={r['events']}")
    print(f"\nFalls detected: {ok_falls}/{n_falls}   Daily activities without false alarm: {ok_adl}/{n_adl}")


if __name__ == "__main__":
    main()
