"""Download the models (they're not in the repo, too big).

    python tools/get_models.py

- yolo11n-pose.pt, yolo11n.pt  -> ultralytics downloads them on first load
- face_landmarker.task         -> mediapipe model, from google's bucket
Then optionally: python tools/optimize_models.py  (openvino, ~2x faster on intel cpus)
"""
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")

if __name__ == "__main__":
    os.chdir(ROOT)
    face = os.path.join(ROOT, "face_landmarker.task")
    if not os.path.exists(face):
        print("downloading face_landmarker.task ...")
        urllib.request.urlretrieve(FACE_URL, face)
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install -r requirements.txt first")
    for name in ("yolo11n-pose.pt", "yolo11n.pt"):
        if not os.path.exists(name):
            YOLO(name)          # triggers the download into cwd
    print("models ok")
