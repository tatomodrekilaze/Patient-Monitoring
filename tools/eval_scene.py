"""Evaluate the object-detector layer (monitor/scene.py).

    python -I tools/eval_scene.py --hmdb <hmdb51_dir> drink,eat,talk,smile 30
    python -I tools/eval_scene.py --rooms <clip_folder> [...]

--hmdb  : share of clips where a cup / bottle / bowl is held at the patient's mouth (drinking /
          eating should be high, talking / smiling low). HMDB51, Kuehne et al. 2011, CC BY 4.0.
--rooms : UR Fall rooms - is a bed found where there is one, and NOT found where there is none
          (a wrong bed would create a wrong bed zone)?
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from monitor.config import make_config  # noqa: E402
from monitor.pose import PatientSelector, PoseTracker  # noqa: E402
from monitor.scene import SceneDetector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def frames(path, step=1):
    if os.path.isdir(path):
        for i, f in enumerate(sorted(glob.glob(os.path.join(path, "**", "*.png"), recursive=True))):
            if i % step == 0:
                yield i, cv2.imread(f)
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, f
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmdb", nargs=3, metavar=("DIR", "CLASSES", "PER_CLASS"))
    ap.add_argument("--rooms", nargs="+")
    a = ap.parse_args()
    cfg = make_config("demo", scene_every_s=0.33)
    if a.hmdb:
        root, classes, per = a.hmdb[0], a.hmdb[1].split(","), int(a.hmdb[2])
        pose = PoseTracker(os.path.join(ROOT, cfg.pose_model), 320, cfg.pose_conf, cfg.torch_threads)
        for c in classes:
            files = sorted(glob.glob(os.path.join(root, c, "*.avi")))[:per]
            at_mouth = seen_obj = 0
            for f in files:
                s, sel = SceneDetector(cfg, os.path.join(ROOT, cfg.scene_model)), PatientSelector()
                pose.reset()
                hit = obj = False
                for i, img in frames(f):
                    t = i / 30.0
                    p, _ = sel.select(pose.infer(img), t)
                    s.update(t, img, p, None)
                    obj |= any(o[0] in ("cup", "bottle", "wine glass", "bowl") for o in s.objects)
                    hit |= s.near_mouth is not None
                at_mouth += hit
                seen_obj += obj
            n = max(1, len(files))
            print(f"{c:7s} clips {len(files):3d}  cup/bottle/bowl seen {seen_obj / n:4.0%}  held at the mouth {at_mouth / n:4.0%}")
    for room in a.rooms or []:
        s = SceneDetector(make_config("demo"), os.path.join(ROOT, cfg.scene_model))
        n = beds = 0
        names = set()
        for i, img in frames(room, step=15):
            s.last_run = -1e9
            s.update(i / 30.0, img, None, None)
            n += 1
            beds += any(o[0] == "bed" and o[2] >= 0.35 for o in s.objects)
            names |= {o[0] for o in s.objects}
        z = s.bed_zone(img.shape)
        print(f"{os.path.basename(room.rstrip('/'))[:10]:10s} frames {n:3d}  bed found in {beds / max(1, n):4.0%}"
              f"  -> bed zone {None if z is None else tuple(round(v, 2) for v in z)}  objects {sorted(names)}")


if __name__ == "__main__":
    main()
