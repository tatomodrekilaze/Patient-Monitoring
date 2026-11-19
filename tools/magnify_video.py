"""Side-by-side Eulerian magnification clip for the pitch: original | magnified.

    python tools/magnify_video.py in.mp4 out.mp4 [--mode breath|pulse] [--box x1,y1,x2,y2]

The box (pixels of the input) is where to magnify - the chest for breath, the face for pulse; without
it the pose model finds the person (breath) or the face (pulse) on the first frames.
Record the input with `python run.py --record-raw clip.mp4`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from monitor.magnify import MODES, Magnifier  # noqa: E402

LABEL = {"breath": "breathing motion x13", "pulse": "pulse colour x36"}


def find_box(frames, mode):
    from monitor.config import make_config
    from monitor.events.face import FaceAnalyzer
    from monitor.pose import PoseTracker
    cfg = make_config("demo")
    pose = PoseTracker(cfg.pose_model, cfg.imgsz, cfg.pose_conf, cfg.torch_threads)
    for f in frames:
        people = pose.infer(f)
        if not people:
            continue
        p = max(people, key=lambda q: (q.box[2] - q.box[0]) * (q.box[3] - q.box[1]))
        if mode == "breath":
            return tuple(int(v) for v in p.box)
        hb = FaceAnalyzer.head_box(p, cfg.kp_conf, f.shape)
        if hb is not None:
            return hb
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=list(MODES), default="breath")
    ap.add_argument("--box", default=None)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > 960:                      # same working size as the live system
            f = cv2.resize(f, (960, int(f.shape[0] * 960 / f.shape[1])), interpolation=cv2.INTER_AREA)
        frames.append(f)
    if not frames:
        sys.exit("no frames in " + a.src)
    box = tuple(int(v) for v in a.box.split(",")) if a.box else find_box(frames[:60], a.mode)
    print("box", box)
    m = Magnifier()
    m.set_mode(a.mode)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(a.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
    for i, f in enumerate(frames):
        g = m.apply(i / fps, f, box)
        pair = np.hstack([f, g])
        for x, text in ((10, "original"), (w + 10, LABEL[a.mode])):
            cv2.putText(pair, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(pair, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
        out.write(pair)
    out.release()
    print(f"wrote {a.dst}: {len(frames)} frames, {fps:.0f} fps")


if __name__ == "__main__":
    main()
