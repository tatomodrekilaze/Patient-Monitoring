// shared helpers for all pages
const $ = (id) => document.getElementById(id);
const ARROWS = { up2: "++", up: "+", flat: "=", down: "-", down2: "--" };
const LEVEL_COLOR = { green: "#5bb974", yellow: "#e0a93b", red: "#e5534b", grey: "#666" };

function connect(onState, onConn) {
  let ws;
  const open = () => {
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.onopen = () => onConn(true);
    ws.onmessage = (e) => onState(JSON.parse(e.data));
    ws.onclose = (e) => {
      if (e.code === 4401) { location.href = "/login?next=" + encodeURIComponent(location.pathname); return; }
      onConn(false);
      setTimeout(open, 1500);
    };
    ws.onerror = () => ws.close();
  };
  open();
}

function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  }).then((r) => r.json());
}

function ago(ts) {
  if (!ts) return "-";
  const m = Math.floor((Date.now() / 1000 - ts) / 60);
  if (m < 1) return "now";
  if (m < 60) return m + "m";
  return Math.floor(m / 60) + "h" + (m % 60) + "m";
}

const hm = (ts) => new Date(ts * 1000).toTimeString().slice(0, 8);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const n0 = (v, d = "-") => (v == null ? d : Math.round(v));

// only rewrite innerHTML when it changed (keeps hover/focus stable at 2.5 updates/s)
function setHTML(el, html) {
  if (el.dataset.h !== html) { el.innerHTML = html; el.dataset.h = html; }
}

// beeps. browsers need one click first
const Sound = {
  ctx: null,
  pausedUntil: 0,
  enable() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.ctx.resume();
  },
  get enabled() { return !!this.ctx && this.ctx.state === "running"; },
  get paused() { return Date.now() < this.pausedUntil; },
  beep(critical) {
    if (!this.enabled || this.paused) return;
    // IEC 60601-1-8 style: high = 10 pulses, medium = 3
    const pattern = critical ? [0, 0.15, 0.3, 0.55, 0.7, 1.2, 1.35, 1.5, 1.75, 1.9] : [0, 0.2, 0.4];
    for (const off of pattern) {
      const o = this.ctx.createOscillator(), g = this.ctx.createGain();
      o.frequency.value = critical ? 880 : 620;
      o.connect(g); g.connect(this.ctx.destination);
      const t = this.ctx.currentTime + off;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.3, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
      o.start(t); o.stop(t + 0.22);
    }
    if (navigator.vibrate) navigator.vibrate(critical ? [400, 150, 400, 150, 400] : [250, 120, 250]);
  },
};

// spoken alerts, browser tts
const Voice = {
  on: false,
  speak(text) {
    if (!this.on || !window.speechSynthesis || Sound.paused) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    speechSynthesis.speak(u);
  },
  say(alert, escalated) {
    this.speak(`${escalated ? "Escalated. " : ""}${alert.bed_name}. ${alert.message}`);
  },
};

// mic (needs https or localhost)
const Recorder = {
  rec: null, chunks: [],
  supported() { return !!(navigator.mediaDevices && window.MediaRecorder && window.isSecureContext); },
  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"].find((m) => MediaRecorder.isTypeSupported(m)) || "";
    this.chunks = [];
    this.rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.rec.start();
  },
  stop() {
    return new Promise((resolve) => {
      if (!this.rec) return resolve(null);
      this.rec.onstop = () => {
        this.rec.stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(this.chunks, { type: this.rec.mimeType || "audio/webm" });
        this.rec = null;
        resolve(blob);
      };
      this.rec.stop();
    });
  },
};

function holdToRecord(btn, onBlob) {
  const label = btn.textContent;
  btn.addEventListener("pointerdown", async (e) => {
    e.preventDefault();
    if (!Recorder.supported()) { alert("mic needs https or localhost"); return; }
    try { await Recorder.start(); btn.classList.add("rec"); btn.textContent = "REC"; }
    catch (err) { alert("no mic access"); }
  });
  const stop = async () => {
    if (!Recorder.rec) return;
    const blob = await Recorder.stop();
    btn.classList.remove("rec"); btn.textContent = label;
    if (blob && blob.size > 1500) onBlob(blob);
  };
  btn.addEventListener("pointerup", stop);
  btn.addEventListener("pointerleave", stop);
}

function postAudio(url, blob) {
  return fetch(url, { method: "POST", headers: { "Content-Type": blob.type || "audio/webm" }, body: blob }).then((r) => r.json());
}

// onNew(alert, escalated) once per new alert / severity change / escalation
function alertWatcher(onNew) {
  const seen = new Set();
  let first = true;
  return (alerts) => {
    for (const a of alerts) {
      for (const [key, esc] of [[`${a.id}:${a.severity}`, false], [`${a.id}:esc`, true]]) {
        if (esc && !a.escalated) continue;
        if (!seen.has(key)) {
          seen.add(key);
          if (!first && a.active) onNew(a, esc);
        }
      }
    }
    first = false;
  };
}

function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const c = cv.getContext("2d");
  c.scale(dpr, dpr);
  return { c, w, h };
}

function sparkline(cv, points, color) {
  if (!cv) return;
  const { c, w, h } = fitCanvas(cv);
  if (!points || points.length < 2) return;
  const ys = points.map((p) => p[1]);
  const lo = Math.min(...ys, 8), hi = Math.max(...ys, 24);
  const t0 = points[0][0], t1 = points[points.length - 1][0];
  c.strokeStyle = color; c.lineWidth = 1; c.beginPath();
  points.forEach((p, i) => {
    const x = ((p[0] - t0) / (t1 - t0 || 1)) * (w - 2) + 1;
    const y = h - 1 - ((p[1] - lo) / (hi - lo || 1)) * (h - 2);
    i ? c.lineTo(x, y) : c.moveTo(x, y);
  });
  c.stroke();
}

function clock(el) {
  const tick = () => { el.textContent = new Date().toTimeString().slice(0, 8); };
  tick();
  setInterval(tick, 1000);
}
