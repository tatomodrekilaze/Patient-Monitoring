"""All user-facing strings."""

LEVEL = {"green": "stable", "yellow": "watch", "red": "alarm", "grey": "no data"}

POSTURE = {
    "lying": "lying",
    "reclined": "reclined",
    "sitting": "sitting",
    "standing": "standing",
    "floor": "on floor",
    "absent": "not visible",
    "unknown": "unclear",
}

TREND = {"up2": "rising fast", "up": "rising", "flat": "flat", "down": "falling", "down2": "falling fast", None: "-"}

SLEEP = {"asleep": "asleep", "awake": "awake", None: "-"}

RISK_PART = {"rr": "resp rate", "deviation": "vs baseline", "trend": "trend", "hr": "heart rate"}

ALERT_MESSAGES = {
    "apnea": "No breathing for {seconds}s",
    "fall": "Fall - patient on floor",
    "fall_other": "Fall - other person in room",
    "floor_long": "On floor for {seconds}s",
    "bed_edge": "Sitting up on bed edge",
    "bed_exit": "Left the bed",
    "absent": "Patient not in view",
    "help": "Hand raised for help",
    "rr_high": "RR high: {rr}/min (baseline {base})",
    "rr_low": "RR low: {rr}/min (baseline {base})",
    "hr_high": "HR high: {hr}/min",
    "hr_low": "HR low: {hr}/min",
    "turn_due": "Same position for {minutes} min - turn due",
    "req_water": "Request: water",
    "req_toilet": "Request: toilet",
    "req_pain": "Request: pain",
    "req_nurse": "Request: nurse",
    "feel_worse": "Patient signals feeling worse",
    "patient_button": "Bedside help button pressed",
    "rounding": "No staff visit for {minutes} min",
    "face_test_asym": "Face test: weak {side} side",
    "camera_offline": "Camera offline ({seconds}s)",
    "chair_stand_low": "Chair stand: {count} in 30s (<8 = fall risk)",
}


ALERT_SHORT = {"apnea": "apnea", "floor_long": "long lie", "turn_due": "turn due", "rounding": "rounding overdue",
               "face_test_asym": "face test asym", "chair_stand_low": "chair stand low", "rr_high": "rr high",
               "rr_low": "rr low", "hr_high": "hr high", "hr_low": "hr low"}


def alert_names():
    """alert kind -> short name, for tables"""
    import re
    return {k: ALERT_SHORT.get(k) or re.sub(r"\s*[:(]?\s*\{[^}]*\}.*$", "", v).strip(" :-") or v
            for k, v in ALERT_MESSAGES.items()}


def alert_message(kind, **kw):
    text = ALERT_MESSAGES.get(kind, kind)
    try:
        return text.format(**kw)
    except (KeyError, IndexError):
        return text


HAND_SIGNAL = {1: "1 water", 2: "2 toilet", 3: "3 pain", 4: "4 nurse", 5: "5 help",
               "ok": "thumbs up", "bad": "thumbs down", 0: "fist"}

# spoken / shown on the bedside screen
PATIENT_PROMPT = {
    "bed_edge": "Please stay in bed. A nurse is coming.",
    "bed_exit": "Please be careful. A nurse is coming.",
    "help": "Got it. A nurse is coming.",
    "req_water": "Water request received.",
    "req_toilet": "Got it. A nurse will help you.",
    "req_pain": "The nurse knows you are in pain.",
    "req_nurse": "A nurse is on the way.",
    "patient_button": "Got it. A nurse is coming.",
    "patient_ok": "Thank you.",
    "feel_worse": "Understood. A nurse will be here soon.",
    "fall": "Help is coming. Please don't get up.",
    "face_smile": "Look at the camera and smile, show your teeth.",
    "face_eyes": "Now close both eyes tightly.",
    "chair_stand": "Cross your arms. Stand up and sit down as many times as you can in 30 seconds.",
}

SIDE_HAND = {"left": "left", "right": "right"}

FEATURE = {
    "breathing": {"name": "resp rate", "what": "chest motion"},
    "apnea": {"name": "apnea", "what": "no breath >= 10s while lying"},
    "rr_change": {"name": "rr vs baseline", "what": "fast/slow vs own baseline"},
    "heart_rate": {"name": "heart rate", "what": "rPPG, face, still only"},
    "fall": {"name": "falls", "what": "fall, long lie, other person"},
    "bed_exit": {"name": "bed exit", "what": "sit up, leave bed"},
    "posture": {"name": "posture", "what": "lying/sitting/standing/floor"},
    "hand_signs": {"name": "hand signs", "what": "1-5 fingers, thumbs"},
    "sleep": {"name": "sleep", "what": "still + eyes closed"},
    "turning": {"name": "turning", "what": "time in one position"},
    "visits": {"name": "visits", "what": "2nd person at bed"},
    "face_test": {"name": "face test", "what": "smile + eyes, guided"},
    "chair_stand": {"name": "chair stand", "what": "30s, guided"},
}

FEATURE_WAIT = {
    "camera": "camera offline",
    "no_patient": "no patient",
    "torso_hidden": "need shoulders + hips",
    "cut_off": "body cut off",
    "too_far": "too far",
    "too_close": "too close",
    "measuring": "measuring",
    "lie_down": "patient must lie down",
    "learning": "baseline {pct}%",
    "watching": "watching",
    "normal_rr": "baseline {rr}/min",
    "no_face": "no face",
    "moving": "moving",
    "body_unclear": "body unclear",
    "zone_bed": "bed detected",
    "zone_learned": "zone learned",
    "zone_manual": "zone set",
    "zone_none": "no bed zone",
    "show_hand": "raise hand",
    "disabled": "off",
    "in_position": "{m} min",
    "visit_none": "none yet",
    "visit_ago": "{m} min ago",
    "running": "running",
    "press_start": "manual start",
}

# one-tap phrases for the bedside speaker
NURSE_PRESETS = {
    "n_coming": "A nurse is coming.",
    "n_stay": "Please stay in bed.",
    "n_howareyou": "How are you? Give a thumbs up if you are OK.",
    "n_doctor": "The doctor will be here soon.",
}

ALERT_TITLE = {"critical": "CRITICAL", "warning": "WARNING"}
