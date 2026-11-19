"""Numbers for hospital management and for the jury: response times, alert load, accuracy."""
import csv
import os
import time

import numpy as np


def alert_stats(alerts):
    """alerts: list of Alert objects."""
    acked = [a for a in alerts if a.acked and a.ack_ts]
    resp = [a.ack_ts - a.ts for a in acked]
    by_kind, by_bed = {}, {}
    for a in alerts:
        by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
        by_bed[a.bed_name] = by_bed.get(a.bed_name, 0) + 1
    rated = [a for a in alerts if a.feedback in ("true", "false")]
    false_by_kind = {}
    for a in rated:
        t, f = false_by_kind.get(a.kind, (0, 0))
        false_by_kind[a.kind] = (t + (a.feedback == "true"), f + (a.feedback == "false"))
    return {
        "rated": len(rated),
        "false_alarm_pct": round(100 * sum(a.feedback == "false" for a in rated) / len(rated)) if rated else None,
        "precision_by_kind": {k: {"true": t, "false": f, "precision_pct": round(100 * t / (t + f))}
                              for k, (t, f) in false_by_kind.items()},
        "total": len(alerts),
        "critical": sum(a.severity == "critical" for a in alerts),
        "acked": len(acked),
        "escalated": sum(a.escalated for a in alerts),
        "response_median_s": _r(np.median(resp)) if resp else None,
        "response_p90_s": _r(np.percentile(resp, 90)) if resp else None,
        "acked_within_60s_pct": round(100 * sum(r <= 60 for r in resp) / len(resp)) if resp else None,
        "by_kind": by_kind,
        "by_bed": by_bed,
        "response_times": [_r(r) for r in resp[-100:]],
    }


def validation_stats(data_dir):
    """Agreement between the camera and breaths counted by hand (Bland-Altman style)."""
    path = os.path.join(data_dir, "validation.csv")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("system_rr") and r.get("manual_rr"):
                    rows.append((float(r["manual_rr"]), float(r["system_rr"])))
    except OSError:
        pass
    if not rows:
        return {"n": 0, "points": []}
    m = np.array(rows)
    diff = m[:, 1] - m[:, 0]
    mean_pair = m.mean(axis=1)
    bias, sd = float(diff.mean()), float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    return {
        "n": len(rows),
        "mae": _r(np.abs(diff).mean(), 2),
        "bias": _r(bias, 2),
        "loa_low": _r(bias - 1.96 * sd, 2),
        "loa_high": _r(bias + 1.96 * sd, 2),
        "within_2_pct": round(100 * float(np.mean(np.abs(diff) <= 2))),
        "points": [[_r(a, 1), _r(d, 2)] for a, d in zip(mean_pair, diff)],
    }


def sleep_report(log, bed_id, hours=12, step_s=10.0):
    """Overnight report built from the 10-second vitals log (like Oxehealth's clinician sleep report)."""
    since = time.time() - hours * 3600
    vit = log.query(bed_id, since, {"vitals"})
    if not vit:
        return {"bed_id": bed_id, "hours": hours, "samples": 0}
    asleep = [v.get("sleep") == "asleep" for v in vit]
    in_bed = [v.get("posture") in ("lying", "reclined") for v in vit]
    wakeups, longest, run = 0, 0, 0
    for prev, cur in zip([False] + asleep, asleep):
        if prev and not cur:
            wakeups += 1
        run = run + 1 if cur else 0
        longest = max(longest, run)
    rr_sleep = [v["rr"] for v, s in zip(vit, asleep) if s and v.get("rr") is not None]
    apneas = log.query(bed_id, since, {"apnea_episode"})
    exits = [e for e in log.query(bed_id, since, {"posture"}) if e.get("event") == "bed_exit"]
    asleep_h = sum(asleep) * step_s / 3600
    in_bed_n = sum(in_bed)
    return {
        "bed_id": bed_id, "hours": hours, "samples": len(vit),
        "asleep_min": round(sum(asleep) * step_s / 60),
        "in_bed_min": round(in_bed_n * step_s / 60),
        "efficiency_pct": round(100 * sum(asleep) / in_bed_n) if in_bed_n else None,
        "wakeups": wakeups,
        "longest_sleep_min": round(longest * step_s / 60),
        "rr_asleep_median": _r(np.median(rr_sleep)) if rr_sleep else None,
        "rr_asleep_range": [_r(min(rr_sleep)), _r(max(rr_sleep))] if rr_sleep else None,
        "apnea_episodes": len(apneas),
        "pauses_per_sleep_hour": _r(len(apneas) / asleep_h) if asleep_h >= 0.05 else None,
        "bed_exits": len(exits),
        "first": vit[0]["ts"], "last": vit[-1]["ts"],
    }


def clinical_report(log, bed_id, snap, hours=24, step_s=10.0):
    """One-page summary for the doctor: everything the camera saw about this patient in the period.
    Built from the event log (numbers and events only) and the bed's current state."""
    now = time.time()
    since = now - hours * 3600
    q = lambda *k: log.query(bed_id, since, set(k))      # noqa: E731
    vit = q("vitals")
    rr = [(v["ts"], v["rr"]) for v in vit if v.get("rr") is not None]
    rr_vals = [x for _, x in rr]
    in_bed_s = sum(step_s for v in vit if v.get("posture") in ("lying", "reclined"))
    alerts = q("alert")
    by_kind = {}
    for a in alerts:
        by_kind[a["alert"]] = by_kind.get(a["alert"], 0) + 1
    apneas = len(q("apnea_episode"))
    acks = [e for e in q("ack")]
    falls = [a for a in alerts if a["alert"] == "fall"]
    baseline = q("baseline")
    return {
        "bed_id": bed_id, "name": snap.get("name"), "hours": hours, "generated": now,
        "first": vit[0]["ts"] if vit else None, "last": vit[-1]["ts"] if vit else None,
        "monitored_min": round(len(vit) * step_s / 60), "in_bed_min": round(in_bed_s / 60),
        "breathing": {
            "baseline": baseline[-1]["rr"] if baseline else None,
            "median": _r(np.median(rr_vals)) if rr_vals else None,
            "min": _r(min(rr_vals)) if rr_vals else None, "max": _r(max(rr_vals)) if rr_vals else None,
            "pct_outside_12_20": round(100 * np.mean([(x < 12 or x > 20) for x in rr_vals])) if rr_vals else None,
            "series": [[round(t), round(x, 1)] for t, x in rr[-720:]],          # last 2 h at 10 s
            "apneas": apneas,
        },
        "alerts": {"total": len(alerts), "by_kind": by_kind,
                   "critical": sum(1 for a in alerts if a.get("severity") == "critical"),
                   "timeline": [{"ts": a["ts"], "kind": a["alert"], "severity": a.get("severity"),
                                 "message": a.get("message")} for a in alerts[-60:]]},
        "falls": [{"ts": f["ts"]} for f in falls],
        "sleep": sleep_report(log, bed_id, hours, step_s),
        "mobility": {"bed_exits": snap.get("bed_exits"), "near_falls": snap.get("near_falls"),
                     "chair_stands": [{"ts": e["ts"], "count": e.get("count")} for e in q("chair_stand")]},
        "skin": {"repositions": snap.get("repositions"), "position_min": snap.get("position_min")},
        "neuro": {"face_tests": [{"ts": e["ts"], "asymmetric": e.get("asymmetric"), "side": e.get("side")}
                                 for e in q("face_test")]},
        "risk": {"early_warning_level": snap.get("level")},
        "care": {"nurse_acks": len(acks), "visits": len(q("visit")), "staff_min": snap.get("staff_min")},
    }


def _r(v, n=1):
    return None if v is None else round(float(v), n)
