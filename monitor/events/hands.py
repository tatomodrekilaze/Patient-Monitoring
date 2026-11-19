"""hand signs for patients who can't talk / reach the call button

  fingers: 1 water, 2 toilet, 3 pain, 4 nurse, 5 help
  thumbs up = ok (cancels requests), thumbs down = feel worse

pose finds a raised hand, then mediapipe hands reads the fingers on a crop (only while a hand is up)
"""
import numpy as np

from ..pose import L_EL, L_WR, NOSE, R_EL, R_WR, kp, torso

SIGNALS = {1: "req_water", 2: "req_toilet", 3: "req_pain", 4: "req_nurse", 5: "help"}
REQUESTS = ("req_water", "req_toilet", "req_pain", "req_nurse", "help")

# MediaPipe hand landmark ids
WRIST, THUMB_MCP, THUMB_IP, THUMB_TIP, INDEX_MCP, INDEX_TIP, PINKY_MCP = 0, 2, 3, 4, 5, 8, 17
FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))      # (pip, tip) for index, middle, ring, pinky
PALM = (0, 5, 9, 13, 17)


def _thumb_out(lm, size):
    """Thumb extended: abducted away from the palm, or straight and clear of the palm even when held
    alongside the index finger (a flat 'stop' hand). Tuned on HaGRID."""
    d = lambda a, b: np.linalg.norm(lm[a] - lm[b])   # noqa: E731
    abducted = d(THUMB_TIP, PINKY_MCP) > 1.1 * d(THUMB_IP, PINKY_MCP) and d(THUMB_TIP, INDEX_MCP) > 0.6 * size
    straight = d(THUMB_MCP, THUMB_TIP) > 0.9 * (d(THUMB_MCP, THUMB_IP) + d(THUMB_IP, THUMB_TIP))
    clear = np.linalg.norm(lm[THUMB_TIP] - lm[list(PALM)].mean(axis=0)) > 0.75 * size
    return bool(abducted or (straight and clear))


def read_hand(lm):
    """-> 'ok' / 'bad' / 0-5 / None. only normal counting shapes (index first), so rock/call/ok
    gestures don't turn into requests (they did on HaGRID)"""
    w = lm[WRIST]
    size = np.linalg.norm(lm[INDEX_MCP] - w) + 1e-6
    up = tuple(np.linalg.norm(lm[tip] - w) > 1.15 * np.linalg.norm(lm[pip] - w) for pip, tip in FINGERS)
    thumb = _thumb_out(lm, size)
    if np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]) < 0.3 * size and any(up[1:]):
        return None                                      # ok-sign circle
    if not any(up):
        if thumb and lm[THUMB_TIP][1] < min(lm[i][1] for i in (6, 10, 14, 18)) - 0.3 * size:
            return "ok"                                  # thumb clearly above the hand
        if thumb and lm[THUMB_TIP][1] > max(lm[i][1] for i in (6, 10, 14, 18)) + 0.3 * size:
            return "bad"                                 # thumb clearly below the hand
        return None if thumb else 0
    counting = {(1, 0, 0, 0): 1, (1, 1, 0, 0): 2, (1, 1, 1, 0): 3, (1, 1, 1, 1): 4}
    n = counting.get(tuple(int(u) for u in up))
    return None if n is None else n + int(thumb)


class HandSignals:
    def __init__(self, cfg):
        self.cfg = cfg
        self._hands = None
        self._frame_i = 0
        self.hi = None              # (full-resolution frame, scale) for this frame, set by the bed
        self.reset()

    def reset(self):
        self.current = None          # what the hand shows right now
        self.cand, self.cand_since = None, None
        self.fired = set()
        self.last_seen_hand = -1e9

    def _detector(self):
        if self._hands is None:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return self._hands

    def raised_wrist(self, person, thr):
        tr = torso(person, thr)
        if tr is None:
            return None
        sh, _, length = tr
        nose = kp(person, NOSE, thr)
        best = None
        for w_i, e_i in ((L_WR, L_EL), (R_WR, R_EL)):
            w = kp(person, w_i, thr)
            if w is not None and nose is not None and np.linalg.norm(w - nose) < 0.35 * length:
                continue                    # hand at the face (eating, scratching): not a signal
            if w is not None and w[1] < sh[1] - 0.35 * length:
                e = kp(person, e_i, thr)
                if best is None or w[1] < best[0][1]:
                    best = (w, e, length)
        return best

    def update(self, t, frame, person, posture_state):
        """Returns events: request kinds, 'patient_ok', 'feel_worse'."""
        events = []
        if person is None or not self.cfg.hand_signals:
            self.cand = self.current = None
            return events
        thr = self.cfg.kp_conf

        raised = self.raised_wrist(person, thr) if posture_state != "floor" else None
        if raised is None:
            self.current, self.cand = None, None
            self.fired.clear()                      # hand down: the same signal may be given again
            return events
        self._frame_i += 1
        if self._frame_i % (3 if self.cfg.lite else 2):   # every other frame is enough and halves the CPU cost
            return events
        sig = self._read(frame, *raised)
        if sig is None:
            return events
        self.last_seen_hand = t
        self.current = sig
        if sig != self.cand:
            self.cand, self.cand_since = sig, t
        if sig in self.fired or t - self.cand_since < self.cfg.signal_hold_s:
            return events
        self.fired.add(sig)
        if sig == "ok":
            events.append("patient_ok")
        elif sig == "bad":
            events.append("feel_worse")
        elif sig in SIGNALS:
            events.append(SIGNALS[sig])
        return events

    def _read(self, frame, wrist, elbow, length):
        lm = self.landmarks(frame, wrist, elbow, length)
        return None if lm is None else read_hand(lm)

    def fingertips(self, frame, wrist, elbow, length):
        """Mean fingertip position in the frame, or None: where the hand really is (the wrist is far
        from the fingers when holding a cup to the mouth)."""
        if not self.cfg.hand_signals:
            return None
        lm = self.landmarks(frame, wrist, elbow, length, absolute=True)
        return None if lm is None else lm[[8, 12, 16]].mean(axis=0)

    def landmarks(self, frame, wrist, elbow, length, absolute=False):
        h, w = frame.shape[:2]
        # the hand extends beyond the wrist, along the forearm
        direction = (wrist - elbow) if elbow is not None else np.array([0.0, -1.0]) * length
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        center = wrist + direction * 0.25 * length
        half = 0.45 * length
        x1, y1 = int(max(0, center[0] - half)), int(max(0, center[1] - half))
        x2, y2 = int(min(w, center[0] + half)), int(min(h, center[1] + half))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        # crop from full res if we have it, 40px hand at 640 is ~120px at 1080p
        src, k = self.hi if self.hi is not None else (frame, 1.0)
        X1, Y1, X2, Y2 = (int(v * k) for v in (x1, y1, x2, y2))
        crop = np.ascontiguousarray(src[Y1:Y2, X1:X2, ::-1])   # BGR -> RGB
        res = self._detector().process(crop)
        if not res.multi_hand_landmarks:
            return None
        lm = np.array([(p.x * (x2 - x1), p.y * (y2 - y1)) for p in res.multi_hand_landmarks[0].landmark])
        return lm + (x1, y1) if absolute else lm
