"""
    python run.py                              webcam
    python run.py --source onvif://<cam-ip>    ip camera
    python run.py --sim-beds 4                 add fake beds to fill the ward view
    python run.py --camera "bed 2=http://<phone-ip>:8080/video"
    python run.py --relay https://... --relay-key ...    mirror to cloud.py
"""
import argparse
import os
import threading
import time

import uvicorn

from monitor.alerts import AlertManager, EventLog
from monitor.bed import BedMonitor, Settings
from monitor.config import make_config
from monitor.relay import RelayClient
from monitor.server import Ward, create_app, lan_urls
from monitor.simulate import PROFILES, SimBed


def keep_cpu():
    """windows only. bump our priority + opt out of power throttling, otherwise with chrome etc
    open we got moved to the e-cores and dropped to 4fps"""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        proc = k32.GetCurrentProcess()
        if not k32.SetPriorityClass(proc, 0x00008000):    # ABOVE_NORMAL_PRIORITY_CLASS
            print(f"[cpu] priority not raised (error {ctypes.get_last_error()})")

        class Throttle(ctypes.Structure):                  # PROCESS_POWER_THROTTLING_STATE
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        st = Throttle(1, 0x1, 0x0)                         # EXECUTION_SPEED controlled, not throttled
        k32.SetProcessInformation(proc, 4, ctypes.byref(st), ctypes.sizeof(st))   # ProcessPowerThrottling
    except Exception as e:                                 # never block the start over this
        print(f"[cpu] could not raise priority: {e}")


def main():
    keep_cpu()
    ap = argparse.ArgumentParser(description="Silent Early Warning - camera patient monitor")
    ap.add_argument("--source", default=None, help="first camera: webcam index (0), video file, http/rtsp stream, or onvif://IP (IP camera, address asked over ONVIF)")
    ap.add_argument("--camera", action="append", default=[], metavar="NAME=SOURCE",
                    help="extra camera, e.g. 'Bed 2=http://phone-ip:8080/video' (repeatable)")
    ap.add_argument("--profile", default="demo", choices=["demo", "clinical"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets phones on the same Wi-Fi connect")
    ap.add_argument("--sim-beds", type=int, default=None, help="extra simulated beds for the ward view")
    ap.add_argument("--bed-name", default=None)
    ap.add_argument("--privacy", default=None, choices=["skeleton", "blur", "raw"])
    ap.add_argument("--imgsz", type=int, default=None, help="pose input size (320 fast, 480 more accurate)")
    ap.add_argument("--no-hr", action="store_true", help="turn off rPPG heart rate (saves CPU)")
    ap.add_argument("--lite", action="store_true", help="weak laptop: no heart rate, slower dashboard video, fewer hand reads")
    ap.add_argument("--record-raw", default=None, help="save the first camera's video to this file (backup demo)")
    ap.add_argument("--access-code", default=os.environ.get("ACCESS_CODE"), help="require a login code")
    ap.add_argument("--ntfy-topic", default=os.environ.get("NTFY_TOPIC"))
    ap.add_argument("--ntfy-escalation-topic", default=os.environ.get("NTFY_ESCALATION_TOPIC"))
    ap.add_argument("--telegram-token", default=os.environ.get("TELEGRAM_TOKEN"))
    ap.add_argument("--telegram-chat", default=os.environ.get("TELEGRAM_CHAT_ID"))
    ap.add_argument("--vertex-project", default=os.environ.get("VERTEX_PROJECT"),
                    help="Google Cloud project for Vertex AI (Gemini assistant + Georgian voice)")
    ap.add_argument("--vertex-region", default=os.environ.get("VERTEX_REGION"))
    ap.add_argument("--ai", default=os.environ.get("AI_PROVIDER"), choices=["auto", "gemini", "claude"])
    ap.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL"))
    ap.add_argument("--no-hand-signals", action="store_true", help="turn off finger-count requests")
    ap.add_argument("--no-patient-voice", action="store_true", help="do not speak to the patient at the bedside")
    ap.add_argument("--relay", default=os.environ.get("RELAY_URL"), help="cloud URL to mirror the ward to")
    ap.add_argument("--relay-key", default=os.environ.get("RELAY_KEY"))
    a = ap.parse_args()

    extra = []
    for c in a.camera:
        name, sep, src = c.partition("=")
        if not sep:
            ap.error(f"--camera needs NAME=SOURCE, got {c!r}")
        extra.append((name.strip(), src.strip()))
    # a camera password is never typed into the command line or a file: write {password} in the
    # address and it is taken from CAMERA_PASSWORD or asked for here (hidden input)
    if any("{password}" in s for s in [a.source or ""] + [s for _, s in extra]):
        import getpass
        from urllib.parse import quote
        pw = quote(os.environ.get("CAMERA_PASSWORD") or getpass.getpass("Camera password: "), safe="")
        a.source = a.source.replace("{password}", pw) if a.source else a.source
        extra = [(n, s.replace("{password}", pw)) for n, s in extra]

    cfg = make_config(
        a.profile, source=a.source, sim_beds=a.sim_beds, bed_name=a.bed_name,
        privacy_default=a.privacy, imgsz=a.imgsz, lite=a.lite or None,
        ntfy_topic=a.ntfy_topic, ntfy_escalation_topic=a.ntfy_escalation_topic,
        telegram_token=a.telegram_token, telegram_chat_id=a.telegram_chat,
        hr_enabled=False if a.no_hr else None, access_code=a.access_code,
        vertex_project=a.vertex_project, vertex_region=a.vertex_region, ai_provider=a.ai,
        gemini_model=a.gemini_model, relay_url=a.relay, relay_key=a.relay_key,
        hand_signals=False if a.no_hand_signals else None,
        patient_prompts=False if a.no_patient_voice else None,
        camera_count=1 + len(extra),
    )
    if cfg.relay_url and not cfg.relay_key:
        ap.error("--relay needs --relay-key (the same RELAY_KEY as the cloud)")
    os.makedirs(cfg.data_dir, exist_ok=True)
    log = EventLog(cfg.data_dir)
    alerts = AlertManager(cfg, log)
    settings = Settings(cfg.data_dir)

    def camera_factory(bed_id, name, source, record_raw=""):
        return BedMonitor(bed_id, name, cfg, alerts, log, settings, source=source, record_raw=record_raw)

    beds = {}
    first = camera_factory("bed1", cfg.bed_name, cfg.source, a.record_raw or "")
    first.start()
    beds["bed1"] = first
    for i, (name, src) in enumerate(extra):
        bed = camera_factory(f"bed{i + 2}", name, src)
        bed.start()
        beds[bed.bed_id] = bed
    for i in range(cfg.sim_beds):
        n = len(beds) + 1
        sb = SimBed(f"sim{n}", f"Bed {n}", cfg, PROFILES[i % len(PROFILES)], seed=i)
        sb.start_thread()
        beds[sb.bed_id] = sb

    ward = Ward(cfg, beds, alerts, log, a.port, camera_factory)

    def escalation_loop():
        while True:
            alerts.tick()
            time.sleep(2)

    threading.Thread(target=escalation_loop, daemon=True).start()
    if cfg.relay_url:
        RelayClient(ward, cfg.relay_url, cfg.relay_key).start()

    print("\n  Ward dashboard:  http://localhost:%d" % a.port)
    print("  Bedside screen:  http://localhost:%d/bedside/bed1   (open on the bedside laptop)" % a.port)
    print("  AI:              %s" % ward.assistant.provider_name +
          (f" on Vertex project {cfg.vertex_project}" if cfg.vertex_project else " (no --vertex-project set)"))
    for u in lan_urls(a.port):
        print("  Nurse phone:     %s/nurse" % u)
    if cfg.ntfy_topic:
        print("  ntfy push topic: %s/%s" % (cfg.ntfy_server, cfg.ntfy_topic))
    if cfg.relay_url:
        print("  Cloud mirror:    %s" % cfg.relay_url)
    print()
    try:
        uvicorn.run(create_app(ward), host=a.host, port=a.port, log_level="warning")
    finally:
        for b in list(beds.values()):
            if hasattr(b, "stop"):
                b.stop()


if __name__ == "__main__":
    main()
