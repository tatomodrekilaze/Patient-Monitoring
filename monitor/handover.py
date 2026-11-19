"""Shift handover text from the event log. rule based, works offline."""
import time

import numpy as np

from .i18n import alert_names

ALERT_NAMES = alert_names()


def _hm(ts):
    return time.strftime("%H:%M", time.localtime(ts))


def bed_summary(bed, log, hours):
    now = time.time()
    since = now - hours * 3600
    snap = bed.snap
    vit = log.query(bed.bed_id, since, {"vitals"})
    alerts = log.query(bed.bed_id, since, {"alert"})
    visits = log.query(bed.bed_id, since, {"visit"})
    lines = [f"# {bed.name}"]

    rrs = [v["rr"] for v in vit if v.get("rr") is not None]
    if rrs:
        lines.append(f"  RR median {np.median(rrs):.0f}/min ({min(rrs):.0f}-{max(rrs):.0f})")
    elif snap.get("rr") is not None:
        lines.append(f"  RR now {snap['rr']:.0f}/min")
    base = snap.get("baseline") or {}
    if base.get("ready"):
        lines.append(f"  baseline {base['rr']:.0f}/min")
    hrs = [v["hr"] for v in vit if v.get("hr") is not None]
    if hrs:
        lines.append(f"  HR median {np.median(hrs):.0f}/min (rppg)")

    lines.append(f"  now: {snap.get('level_label', '-')}, trend {snap.get('trend_label', '-')}, "
                 f"{snap.get('posture_label', '-')}")

    if alerts:
        by_kind = {}
        for a in alerts:
            by_kind.setdefault(a["alert"], []).append(_hm(a["ts"]))
        parts = [f"{ALERT_NAMES.get(k, k)} x{len(v)} ({', '.join(v[-3:])})" for k, v in by_kind.items()]
        lines.append("  alerts: " + "; ".join(parts))
    else:
        lines.append("  alerts: none")

    if snap.get("bed_exits"):
        lines.append(f"  bed exits: {snap['bed_exits']}")
    if snap.get("asleep_min"):
        m = int(snap["asleep_min"])
        lines.append(f"  slept ~{m // 60}h {m % 60}m")
    notes = log.query(bed.bed_id, since, {"note"})
    for n in notes[-3:]:
        lines.append(f"  note {_hm(n['ts'])}: {n.get('summary') or n.get('transcript') or ''}")
    last = snap.get("last_visit") or (visits[-1]["ts"] if visits else None)
    lines.append(f"  last visit: {_hm(last) if last else 'none'}")
    return "\n".join(lines)


def ward_summary(beds, log, hours=8):
    now = time.time()
    head = f"handover {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}, last {hours:g}h\n"
    ranked = sorted(beds, key=lambda b: -["grey", "green", "yellow", "red"].index(b.snap.get("level", "grey")))
    body = []
    for b in ranked:
        if b.snap.get("simulated"):
            body.append(f"# {b.name} (sim): {b.snap.get('level_label', '-')}, "
                        f"RR {b.snap.get('rr', 0):.0f}/min, {b.snap.get('trend_label', '-')}")
        else:
            body.append(bed_summary(b, log, hours))
    return head + "\n" + "\n\n".join(body)
