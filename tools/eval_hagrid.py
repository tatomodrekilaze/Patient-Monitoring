"""Evaluate the finger-count hand signals on HaGRID (Kapitanov et al., CC BY-SA 4.0).

    python -I tools/eval_hagrid.py <hagrid-sample-30k-384p dir> [--per-class 200]

Each photo is cropped around the annotated hand (like the live system crops around the raised
wrist), read by MediaPipe Hands, then by monitor.events.hands.read_hand.
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

from monitor.events.hands import read_hand  # noqa: E402

# what each HaGRID gesture should produce in our signal language
TARGET = {"one": 1, "peace": 2, "peace_inverted": 2, "two_up": 2, "two_up_inverted": 2,
          "three": 3, "three2": 3, "four": 4, "palm": 5, "stop": 5, "stop_inverted": 5, "like": "ok",
          "dislike": "bad"}
# gestures that are NOT signals: anything they produce except 0/None would be a false request
OTHER = ("fist", "ok", "call", "rock", "mute")
REQUEST = {1, 2, 3, 4, 5, "ok", "bad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--only", default="", help="comma-separated gestures to test")
    a = ap.parse_args()
    random.seed(0)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0,
                                     min_detection_confidence=0.5)
    results = {}
    for gesture in list(TARGET) + list(OTHER):
        if a.only and gesture not in a.only.split(","):
            continue
        ann = json.load(open(os.path.join(a.root, "ann_train_val", f"{gesture}.json"), encoding="utf-8"))
        files = sorted(glob.glob(os.path.join(a.root, "hagrid_30k", f"train_val_{gesture}", "*.jpg")))
        random.shuffle(files)
        counts = collections.Counter()
        for f in files[:a.per_class]:
            info = ann.get(os.path.splitext(os.path.basename(f))[0])
            if not info or gesture not in info["labels"]:
                continue
            img = cv2.imread(f)
            H, W = img.shape[:2]
            x, y, w, h = info["bboxes"][info["labels"].index(gesture)]
            cx, cy, side = (x + w / 2) * W, (y + h / 2) * H, max(w * W, h * H) * 1.6
            x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            x2, y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            crop = np.ascontiguousarray(img[y1:y2, x1:x2, ::-1])
            res = hands.process(crop)
            if not res.multi_hand_landmarks:
                counts["no hand"] += 1
                continue
            lm = np.array([(p.x * (x2 - x1), p.y * (y2 - y1)) for p in res.multi_hand_landmarks[0].landmark])
            counts[read_hand(lm)] += 1
        results[gesture] = counts

    print(f"{'gesture':16s} {'expect':6s} {'correct':>8s}  {'hand found':>10s}  outputs")
    ok_total = n_total = 0
    for g, c in results.items():
        n = sum(c.values())
        found = n - c["no hand"]
        if g in TARGET:
            correct = c[TARGET[g]]
            ok_total, n_total = ok_total + correct, n_total + found
            print(f"{g:16s} {str(TARGET[g]):6s} {100 * correct / max(1, found):7.0f}%  {100 * found / max(1, n):9.0f}%  {dict(c)}")
        else:
            false = sum(v for k, v in c.items() if k in REQUEST)
            print(f"{g:16s} {'none':6s} {'':>8s}  {100 * found / max(1, n):9.0f}%  false request {100 * false / max(1, found):.0f}%  {dict(c)}")
    print(f"\nSignal gestures read correctly (when a hand is found): {100 * ok_total / max(1, n_total):.1f}%")


if __name__ == "__main__":
    main()
