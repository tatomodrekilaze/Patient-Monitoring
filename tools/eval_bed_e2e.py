"""Run the complete bed pipeline (BedMonitor.process) on a video or PNG-sequence folder and print the
alerts (with the video time they fired) and the bed / hands / walking / vision-AI readouts -
an end-to-end check without a camera.

    python -I tools/eval_bed_e2e.py <video_or_folder> [...] [--reverse] [--fps 30]

--reverse plays the clip backwards (a "lying down" clip becomes "getting out of bed").
--lead N holds the first frame N seconds first (a patient lies in bed a while before getting up).
"""
import argparse
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

import monitor.bed as mb  # noqa: E402
from monitor.alerts import AlertManager, EventLog  # noqa: E402
from monitor.bed import BedMonitor, Settings  # noqa: E402
from monitor.config import make_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _NoCap:
    restarts = 0

    def __init__(self, *a, **k):
        pass

    def release(self):
        pass


mb.Capture = _NoCap


def frames_of(path, reverse=False, lead=0.0, fps=30.0):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.png"), recursive=True))
        files = list(reversed(files)) if reverse else files
        files = [files[0]] * int(lead * fps) + files
        for f in files:
            yield cv2.imread(f)
        return
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if reverse:
            frames.append(f)
        else:
            yield f
    yield from reversed(frames)


def run(path, fps=30.0, reverse=False, lead=0.0, zones=None, **over):
    d = tempfile.mkdtemp()
    cfg = make_config("demo", data_dir=d, patient_prompts=False, pose_model=os.path.join(ROOT, "yolo11n-pose.pt"),
                      scene_model=os.path.join(ROOT, "yolo11n.pt"), face_model=os.path.join(ROOT, "face_landmarker.task"),
                      **over)
    log = EventLog(d)
    alerts = AlertManager(cfg, log)
    bed = BedMonitor("b1", "test", cfg, alerts, log, Settings(d), source="none.mp4")
    for name, z in (zones or {}).items():
        bed.set_room_zone(name, z)
    fired, now = [], [0.0]
    orig = bed._alert

    def alert(kind, severity, **kw):
        fired.append((round(now[0], 1), kind))
        orig(kind, severity, **kw)
    bed._alert = alert
    states, max_exit = [], 0.0
    for i, f in enumerate(frames_of(path, reverse, lead, fps)):
        if f is None:
            continue
        if f.shape[1] > 640:
            f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])))
        now[0] = i / fps
        bed.process(now[0], f)
        max_exit = max(max_exit, bed.exitp.score)
        s = bed.snap["posture"] + ("(hidden)" if bed.snap.get("hidden") else "")
        if not states or states[-1][1] != s:
            states.append((round(i / fps, 1), s))
    return bed.snap, fired, states, max_exit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--lead", type=float, default=0.0)
    ap.add_argument("--zone", action="append", default=[], help="name=x1,y1,x2,y2 (normalised): door / bathroom / sanitizer")
    a = ap.parse_args()
    for p in a.paths:
        zones = {z.split("=")[0]: [float(v) for v in z.split("=")[1].split(",")] for z in a.zone}
        s, fired, states, max_exit = run(p, a.fps, a.reverse, a.lead, zones)
        print(os.path.basename(p.rstrip("/\\"))[:12] + (" (reversed)" if a.reverse else ""), "alerts", fired)
        print("   states", " > ".join(f"{t}:{x}" for t, x in states))
        print(f"   bed zone from: {s['zone_source']}  objects {sorted(set(s['objects']))}  exit score max {max_exit:.2f}"
              f" | fall risk {s['fall_risk']} {s['fall_risk_parts']} | pain {s['pain']} eyes_open {s['eyes_open']}"
              f" | gait {s['gait_mps']} | agit {s['agitation']}")


if __name__ == "__main__":
    main()
