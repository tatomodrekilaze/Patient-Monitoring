"""Evaluate the chair-stand counter (30-s chair stand test) on real people: HMDB51 "stand" (standing up)
and "sit" (sitting down) clips (Kuehne et al. 2011, CC BY 4.0).

    python -I tools/eval_chair.py <hmdb51_dir>

Only full-body clips with a still camera (tags _f_nm_) are used: the counter needs the knees, and a
ward camera does not pan with the patient. A "stand" clip should count
one stand when it starts seated; a "sit" clip must count none (sitting down is not a stand).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from monitor.config import make_config  # noqa: E402
from monitor.events.mobility import ChairStandTest  # noqa: E402
from monitor.pose import PatientSelector, PoseTracker  # noqa: E402


def run(path, pose, cfg):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sel, test = PatientSelector(), ChairStandTest()
    pose.reset()
    test.arm(0.0, window_s=1e9)
    i, seated_seen = 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        p, _ = sel.select(pose.infer(frame), t)
        test.update(t, p, cfg.kp_conf)
        seated_seen |= test.state in ("seated", "running")
    return seated_seen, test.count


def main():
    root = sys.argv[1]
    cfg = make_config("demo")
    pose = PoseTracker(cfg.pose_model, cfg.imgsz, cfg.pose_conf, cfg.torch_threads)
    for cls in ("stand", "sit"):
        files = sorted(glob.glob(os.path.join(root, cls, "*_f_nm_*.avi")))
        res = [run(f, pose, cfg) for f in files]
        counted = sum(c >= 1 for _, c in res)
        extra = sum(c >= 2 for _, c in res)
        if cls == "stand":
            seated = [c for s, c in res if s]
            print(f"stand: {len(files)} full-body clips; started seated (seen by the counter) {len(seated)}; "
                  f"of those counted 1 stand: {sum(c == 1 for c in seated)}/{len(seated)}, "
                  f"counted 2+: {sum(c >= 2 for c in seated)}")
        else:
            print(f"sit:   {len(files)} full-body clips; a stand wrongly counted in {counted} "
                  f"({counted / max(1, len(files)):.0%}), 2+ in {extra}")


if __name__ == "__main__":
    main()
