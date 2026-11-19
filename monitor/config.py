"""Settings.

profiles:
  demo      short windows so things react within seconds (default)
  clinical  realistic windows (30 min baseline, 2 h turning etc)
"""
import os
from dataclasses import dataclass, replace


@dataclass
class Config:
    # capture
    source: str = "0"              # webcam index / video file / stream url
    width: int = 640
    height: int = 480
    fps: int = 30
    loop_video: bool = True
    record_raw: str = ""           # save raw camera video here (for making test clips)

    # pose
    pose_model: str = "yolo11n-pose.pt"
    imgsz: int = 320
    pose_conf: float = 0.30
    kp_conf: float = 0.40
    torch_threads: int = 4
    camera_count: int = 1          # threads get split between cameras

    # breathing
    resample_hz: float = 10.0
    breath_band: tuple = (0.1, 0.8)      # Hz, = 6-48 /min
    breath_window_s: float = 30.0
    rr_window_s: float = 20.0
    rr_min_data_s: float = 10.0
    max_rr: float = 50.0
    apnea_s: float = 10.0
    breath_view_conf: float = 0.5        # shoulders+hips need this conf before we measure
    breath_view_hold_s: float = 3.0      # and stay visible this long
    breath_good_q: float = 0.4
    rr_alert_hold_s: float = 15.0        # clinical: 2 min
    flow_roi_px: int = 96
    motion_thr: float = 0.30             # torso lengths/s -> moving

    # heart rate (rPPG)
    hr_enabled: bool = True
    hr_band: tuple = (0.7, 3.0)          # 42-180 bpm
    hr_window_s: float = 12.0
    hr_in_risk: bool = False             # too noisy to score, display only
    hr_quality: float = 0.3
    hr_hold_s: float = 10.0              # keep last good reading on screen

    # posture / events
    state_min_s: float = 0.8
    floor_min_s: float = 1.5
    absent_s: float = 5.0
    help_hold_s: float = 3.0
    visit_min_s: float = 3.0             # 2nd person next to bed this long = visit
    sleep_onset_s: float = 60.0
    camera_offline_s: float = 8.0
    unclear_s: float = 2.0               # after this "unclear" is shown instead of a guess
    turn_interval_s: float = 600.0       # clinical: 2h. 90s was way too annoying
    reposition_thr: float = 0.35
    night_thr: float = 70.0              # mean brightness, below -> CLAHE
    # falls. numbers come from the UR Fall runs, see docs/validation.md
    fall_vanish_drop: float = 1.0        # falls were 1.25-1.44, normal stuff <= 0.61
    fall_vanish_window_s: float = 0.5
    fall_vanish_s: float = 1.5
    nearfall_drop: float = 0.2           # 0.15 gave a false one
    fall_strong_drop: float = 1.0
    fall_strong_wh: float = 1.25
    fall_strong_torso: float = 45.0
    floor_long_s: float = 20.0
    hidden_in_bed_s: float = 60.0        # under a blanket etc, don't call it absent yet
    lite: bool = False

    # scene / face
    scene_ai: bool = True                # yolo11n, only used to find the bed
    scene_model: str = "yolo11n.pt"
    scene_every_s: float = 1.0
    face_ai: bool = True
    face_model: str = "face_landmarker.task"
    face_every_s: float = 0.5
    pain_thr: float = 5.0
    pain_hold_s: float = 3.0
    asym_thr: float = 0.03
    asym_hold_s: float = 20.0
    face_baseline_n: int = 20
    smile_asym_thr: float = 0.035        # healthy p95 was 0.026
    eye_asym_thr: float = 0.25           # healthy p95 0.16
    chair_stand_low: int = 8
    face_max_yaw: float = 0.2            # ~10 deg
    face_max_width: float = 0.25         # face too close -> perspective looks like droop
    face_max_dpitch: float = 0.06
    hires_crops: bool = True             # crop chest/hands/face from the full res frame
    flow_roi_hi_px: int = 128
    kp_smoothing: bool = False           # one-euro, made UR Fall worse so off

    # baseline / risk
    baseline_s: float = 45.0
    trend_window_s: float = 60.0
    trend_thr: float = 4.0
    trend_steep_thr: float = 10.0
    escalate_s: float = 3.0
    deescalate_s: float = 15.0
    forecast_horizon_s: float = 600.0

    # alerts
    alert_cooldown_s: float = 60.0
    escalate_after_s: float = 30.0       # not acked -> head nurse
    ntfy_server: str = "https://ntfy.sh"
    ntfy_topic: str = ""
    ntfy_escalation_topic: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # optional LLM assistant
    ai_provider: str = "auto"            # auto | gemini | claude
    gemini_model: str = "gemini-3.8-flash"
    ai_model: str = "claude-opus-5-5"
    vertex_project: str = ""
    vertex_region: str = "global"

    # hand signs / bedside
    hand_signals: bool = True
    signal_hold_s: float = 2.0
    patient_prompts: bool = True
    rounding_interval_s: float = 1200.0  # clinical 1h

    # cloud relay
    access_code: str = ""
    relay_url: str = ""
    relay_key: str = ""

    # server
    privacy_default: str = "blur"        # skeleton | blur | raw
    stream_fps: float = 20.0
    data_dir: str = "data"
    bed_name: str = "Bed 1"
    sim_beds: int = 0

    profile: str = "demo"


CLINICAL = dict(
    baseline_s=30 * 60.0,
    trend_window_s=15 * 60.0,
    trend_thr=1.0,
    trend_steep_thr=3.0,
    escalate_s=30.0,
    deescalate_s=120.0,
    sleep_onset_s=5 * 60.0,
    alert_cooldown_s=5 * 60.0,
    escalate_after_s=120.0,
    turn_interval_s=2 * 3600.0,
    forecast_horizon_s=3600.0,
    rounding_interval_s=3600.0,
    floor_long_s=60.0,
    rr_alert_hold_s=120.0,
    asym_hold_s=60.0,
)

# for weak laptops. don't drop imgsz below 320, at 256 it missed a fall on UR Fall
LITE = dict(hr_enabled=False, stream_fps=10.0, scene_every_s=3.0, face_every_s=1.0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_path(name, imgsz=320):
    """Use the openvino export of a yolo model if there is one (about 2x faster on intel)."""
    import importlib.util
    path = name if os.path.isabs(name) else os.path.join(ROOT, name)
    if path.endswith(".pt") and not os.environ.get("SEW_NO_OPENVINO") \
            and importlib.util.find_spec("openvino") is not None:
        # dynamic export first. the fixed 320x320 one letterboxes differently and gave
        # 2 extra false falls
        for ov in (f"{path[:-3]}_dyn_openvino_model", f"{path[:-3]}_{imgsz}_openvino_model",
                   path[:-3] + "_openvino_model" if imgsz == 320 else None):
            if ov and os.path.isdir(ov):
                return ov
    return path if os.path.exists(path) else name


def make_config(profile="demo", **overrides):
    cfg = Config(profile=profile)
    if profile == "clinical":
        cfg = replace(cfg, **CLINICAL)
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides.get("lite"):
        overrides = {**LITE, **overrides}
    return replace(cfg, **overrides)
