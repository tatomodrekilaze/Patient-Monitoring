"""Export the YOLO models to OpenVINO for Intel CPUs (once; the app then picks them up by itself).

    pip install openvino
    python tools/optimize_models.py

Measured on this laptop (i5-1235U) inside the full pipeline: pose 25 ms -> 12 ms per frame.
The export is DYNAMIC (<model>_dyn_openvino_model): frames are letterboxed exactly like PyTorch.
A fixed 320x320 export padded 4:3 / 16:9 frames differently and raised 2 extra false falls on UR Fall;
the dynamic one gives PyTorch's results (UR Fall: 12 / 15 falls, 38 / 40 activities without false alarm).
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        import openvino  # noqa: F401
    except ImportError:
        sys.exit("OpenVINO is not installed:  pip install openvino")
    from ultralytics import YOLO
    for name in ("yolo11n-pose.pt", "yolo11n.pt"):
        path = os.path.join(ROOT, name)
        out = YOLO(path).export(format="openvino", imgsz=320, half=False, dynamic=True)
        dst = os.path.join(ROOT, name[:-3] + "_dyn_openvino_model")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.move(out, dst)
        print("ok", dst)
