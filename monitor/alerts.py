"""Alerts: de-duplication, escalation, acknowledgement, push to phones, event log.

Phone delivery:
  1. the /nurse web page (same Wi-Fi / hotspot) - sound + vibration, no internet needed
  2. ntfy (free, open source push app) - set --ntfy-topic, phone subscribes in the ntfy app
  3. Telegram bot - optional, --telegram-token + --telegram-chat
"""
import hashlib
import itertools
import json
import os
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field

from .i18n import ALERT_TITLE

SEVERITY_RANK = {"warning": 1, "critical": 2}


@dataclass
class Alert:
    id: int
    bed_id: str
    bed_name: str
    kind: str
    severity: str
    message: str
    ts: float = field(default_factory=time.time)
    acked: bool = False
    resolved: bool = False
    ack_ts: float = None
    escalated: bool = False
    resolved_ts: float = None
    feedback: str = None        # "true" / "false": nurse says whether the alert was real
    replay: bool = False        # a skeleton replay of the moments around it is saved (monitor/replay.py)

    @property
    def active(self):
        return not self.acked and not self.resolved


GENESIS = "0" * 64


def _canon(e):
    return json.dumps(e, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _link(prev, e):
    return hashlib.sha256((prev + _canon(e)).encode("utf-8")).hexdigest()


def verify_chain(path):
    """Audit check of events.jsonl: every line carries the SHA-256 of (previous hash + its content),
    so editing, inserting or deleting any past line breaks the chain from that line on."""
    prev, n, legacy = GENESIS, 0, 0
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return {"ok": True, "records": 0, "legacy": 0, "head": GENESIS, "first_bad_line": None}
    with f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                return {"ok": False, "records": n, "legacy": legacy, "head": prev, "first_bad_line": i}
            h, p = rec.pop("_h", None), rec.pop("_prev", None)
            if h is None:
                if n:                       # an unchained line after the chain started = inserted
                    return {"ok": False, "records": n, "legacy": legacy, "head": prev, "first_bad_line": i}
                legacy += 1                 # written before the audit chain existed
                continue
            if p != prev or _link(prev, rec) != h:
                return {"ok": False, "records": n, "legacy": legacy, "head": prev, "first_bad_line": i}
            prev, n = h, n + 1
    return {"ok": True, "records": n, "legacy": legacy, "head": prev, "first_bad_line": None}


class EventLog:
    """Numbers and events only - never images. One JSON line per entry, hash-chained (tamper-evident)."""

    def __init__(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "events.jsonl")
        self.entries = []
        self._lock = threading.Lock()
        self.head = GENESIS
        # the last 48 h survive a restart (the doctor's report and the handover cover a whole shift)
        since = time.time() - 48 * 3600
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    self.head = e.pop("_h", self.head)
                    e.pop("_prev", None)
                    if e.get("ts", 0) >= since:
                        self.entries.append(e)
        except OSError:
            pass

    def add(self, bed_id, kind, **data):
        e = {"ts": time.time(), "bed_id": bed_id, "kind": kind, **data}
        with self._lock:
            self.entries.append(e)
            h = _link(self.head, e)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({**e, "_prev": self.head, "_h": h}, ensure_ascii=False) + "\n")
            self.head = h
        return e

    def query(self, bed_id=None, since=0.0, kinds=None):
        with self._lock:
            return [e for e in self.entries
                    if e["ts"] >= since and (bed_id is None or e["bed_id"] == bed_id)
                    and (kinds is None or e["kind"] in kinds)]


class AlertManager:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log
        self.alerts = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._last_raised = {}
        self.notifiers = []
        if cfg.ntfy_topic:
            self.notifiers.append(NtfyNotifier(cfg.ntfy_server, cfg.ntfy_topic))
        if cfg.telegram_token and cfg.telegram_chat_id:
            self.notifiers.append(TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id))
        # second level of the escalation chain (head nurse / doctor); falls back to the same topic
        topic = cfg.ntfy_escalation_topic or cfg.ntfy_topic
        self.escalation_notifiers = [NtfyNotifier(cfg.ntfy_server, topic)] if topic else []

    def tick(self, now=None):
        """Escalate alerts nobody acknowledged in time. Call every few seconds."""
        now = now or time.time()
        to_send = []
        with self._lock:
            for a in self.alerts:
                if a.active and not a.escalated and now - a.ts >= self.cfg.escalate_after_s:
                    a.escalated = True
                    to_send.append(a)
        for a in to_send:
            self.log.add(a.bed_id, "escalated", alert=a.kind, after_s=round(now - a.ts))
            for n in self.escalation_notifiers:
                threading.Thread(target=n.send, args=(a, True), daemon=True).start()
        return to_send

    def raise_alert(self, bed_id, bed_name, kind, severity, message, notify=True):
        now = time.time()
        with self._lock:
            for a in self.alerts:
                if a.active and a.bed_id == bed_id and a.kind == kind:
                    escalated = SEVERITY_RANK[severity] > SEVERITY_RANK[a.severity]
                    a.message = message
                    if escalated:
                        a.severity, a.ts = severity, now
                        break
                    return a
            else:
                last = self._last_raised.get((bed_id, kind))
                # just acknowledged/resolved: do not nag again immediately (falls always pass)
                if last and now - last < self.cfg.alert_cooldown_s and severity != "critical":
                    return None
                a = Alert(next(self._ids), bed_id, bed_name, kind, severity, message)
                self.alerts.append(a)
                self.alerts = self.alerts[-500:]
            self._last_raised[(bed_id, kind)] = now
        self.log.add(bed_id, "alert", alert=kind, severity=severity, message=message)
        if notify:
            for n in self.notifiers:
                threading.Thread(target=n.send, args=(a,), daemon=True).start()
        return a

    def mark_replay(self, alert_id):
        with self._lock:
            for a in self.alerts:
                if a.id == alert_id:
                    a.replay = True

    def resolve(self, bed_id, kind):
        with self._lock:
            for a in self.alerts:
                if a.active and a.bed_id == bed_id and a.kind == kind:
                    a.resolved, a.resolved_ts = True, time.time()

    def ack(self, alert_id):
        with self._lock:
            for a in self.alerts:
                if a.id == alert_id and not a.acked:
                    a.acked, a.ack_ts = True, time.time()
                    self.log.add(a.bed_id, "ack", alert=a.kind)
                    return True
        return False

    def feedback(self, alert_id, value):
        """Nurse marks an alert real or false: drives the false-alarm rate (alarm fatigue)."""
        if value not in ("true", "false"):
            return False
        with self._lock:
            for a in self.alerts:
                if a.id == alert_id:
                    a.feedback = value
                    if not a.acked:
                        a.acked, a.ack_ts = True, time.time()
                    self.log.add(a.bed_id, "feedback", alert=a.kind, value=value)
                    return True
        return False

    def active(self, bed_id=None):
        with self._lock:
            return [a for a in self.alerts if a.active and (bed_id is None or a.bed_id == bed_id)]

    def recent(self, n=30):
        with self._lock:
            return [asdict(a) | {"active": a.active} for a in self.alerts[-n:]][::-1]

    def all(self):
        with self._lock:
            return list(self.alerts)


class NtfyNotifier:
    def __init__(self, server, topic):
        self.server, self.topic = server.rstrip("/"), topic

    def send(self, alert, escalation=False):
        prefix = "ESCALATED: " if escalation else ""
        body = {
            "topic": self.topic,
            "title": f"{prefix}{ALERT_TITLE[alert.severity]} - {alert.bed_name}",
            "message": alert.message,
            "priority": 5 if alert.severity == "critical" else 4,
            "tags": ["rotating_light" if alert.severity == "critical" else "warning"],
        }
        _post_json(self.server, body)


class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, alert, escalation=False):
        prefix = "ESCALATED: " if escalation else ""
        text = f"{prefix}{ALERT_TITLE[alert.severity]} - {alert.bed_name}\n{alert.message}"
        _post_json(self.url, {"chat_id": self.chat_id, "text": text})


def _post_json(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:  # never let a network problem stop monitoring
        print(f"[alerts] push failed: {e}")
