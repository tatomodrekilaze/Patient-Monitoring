"""AI features: ward assistant (text or spoken question), AI handover, voice notes.

Providers:
  gemini (default when --vertex-project is set) - Gemini on Google Cloud Vertex AI.
          Also does speech: audio in, text out.
  claude  - Claude through the Anthropic API or Vertex AI (text only).

The model only ever sees numbers, events and the nurse's own words - no images, no names.
"""
import json
import threading

SYSTEM = """You are the ward assistant inside "Silent Early Warning", a contactless camera system that \
monitors hospital patients (breathing rate from chest motion, posture, falls, apnea, seizure suspicion, \
time in one position, patient hand signals) and warns nurses early.

You receive live ward data inside <ward_data>. Use only that data; if something is not in it, say you \
don't know. Answer in the language the nurse uses. Be brief and practical: \
a nurse reads this on a phone between tasks - at most 6 short sentences or a short list, most urgent \
bed first. Give bed names and numbers. You support decisions; you do not diagnose or prescribe - \
suggest what to check, and for anything red say to assess the patient now. Beds marked simulated are \
demo data; say so if you mention them."""

HANDOVER_PROMPT = """Write the nursing shift handover from the ward data. One short block per \
bed, most at-risk bed first: current status and trend, what happened this shift (alerts with times), \
nurse voice notes, and what the next shift should watch. End with one line listing beds that need \
attention first."""

VOICE_QUESTION_PROMPT = """The audio is a nurse asking a question. Answer it using \
the ward data. Reply as JSON: {"question": "<what the nurse asked, transcribed>", "answer": "<your answer>"}"""

NOTE_PROMPT = """The audio is a nurse dictating an observation about a patient. \
Reply as JSON with these keys:
"transcript": exact transcription in the original language,
"summary": one-sentence summary,
"observations": list of short items (symptoms, behaviour, intake, wounds...),
"vitals": object with any numbers the nurse said (keys like bp, hr, rr, spo2, temp, pain),
"actions": list of things done or to do,
"urgency": "routine", "soon" or "urgent".
Do not invent anything that was not said."""


class AIError(Exception):
    pass


class GeminiVertex:
    name = "gemini"
    can_hear = True

    def __init__(self, cfg):
        from google import genai
        self.genai = genai
        self.client = genai.Client(vertexai=True, project=cfg.vertex_project, location=cfg.vertex_region)
        self.model = cfg.gemini_model

    def generate(self, system, text, audio=None, mime=None, json_out=False):
        from google.genai import errors, types
        parts = []
        if audio is not None:
            parts.append(types.Part.from_bytes(data=audio, mime_type=mime or "audio/webm"))
        parts.append(text)
        config = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=4000,
            response_mime_type="application/json" if json_out else None)
        try:
            r = self.client.models.generate_content(model=self.model, contents=parts, config=config)
        except errors.ClientError as e:
            raise AIError(f"Vertex AI auth/permission error ({e.code}): {e.message}") from e
        except errors.ServerError as e:
            raise AIError(f"Vertex AI unavailable ({e.code}).") from e
        return (r.text or "").strip()


class Claude:
    name = "claude"
    can_hear = False

    def __init__(self, cfg):
        import anthropic
        self.anthropic = anthropic
        self.vertex = bool(cfg.vertex_project)
        self.client = (anthropic.AnthropicVertex(project_id=cfg.vertex_project, region=cfg.vertex_region)
                       if self.vertex else anthropic.Anthropic())
        self.model = cfg.ai_model

    def generate(self, system, text, audio=None, mime=None, json_out=False):
        if audio is not None:
            raise AIError("voice needs gemini (--ai gemini)")
        a = self.anthropic
        kwargs = dict(model=self.model, max_tokens=4000, system=system,
                      output_config={"effort": "medium"}, messages=[{"role": "user", "content": text}])
        try:
            if self.vertex:
                r = self.client.messages.create(**kwargs)
            else:
                # if a safety classifier declines, the API retries on a fallback model in the same call
                r = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except a.AuthenticationError as e:
            raise AIError("claude: missing or bad api key") from e
        except a.PermissionDeniedError as e:
            raise AIError("claude: permission denied") from e
        except a.RateLimitError as e:
            raise AIError("claude overloaded, try again") from e
        except a.APIStatusError as e:
            raise AIError(f"claude error ({e.status_code})") from e
        except a.APIConnectionError as e:
            raise AIError("no internet") from e
        if r.stop_reason == "refusal":
            raise AIError("no answer")
        return "".join(b.text for b in r.content if b.type == "text").strip()


class WardAssistant:
    def __init__(self, cfg):
        self.cfg = cfg
        self._provider = None
        self._lock = threading.Lock()

    @property
    def provider_name(self):
        p = self.cfg.ai_provider
        if p == "auto":
            p = "gemini" if self.cfg.vertex_project else "claude"
        return p

    def _get(self):
        with self._lock:
            if self._provider is None:
                if self.provider_name == "gemini":
                    if not self.cfg.vertex_project:
                        raise AIError("gemini needs --vertex-project")
                    self._provider = GeminiVertex(self.cfg)
                else:
                    self._provider = Claude(self.cfg)
            return self._provider

    def _call(self, system, text, **kw):
        try:
            return self._get().generate(system, text, **kw)
        except AIError:
            raise
        except ImportError as e:
            raise AIError(f"missing package: {e.name}") from e
        except Exception as e:     # missing Google/Anthropic credentials surface here
            raise AIError(f"ai not configured: {e}") from e

    # ---- text ----
    def ask(self, question, ward_data):
        try:
            return self._call(SYSTEM, f"<ward_data>\n{ward_data}\n</ward_data>\n\n{question}")
        except AIError as e:
            return str(e)

    def handover(self, ward_data):
        try:
            return self._call(SYSTEM, f"<ward_data>\n{ward_data}\n</ward_data>\n\n{HANDOVER_PROMPT}")
        except AIError as e:
            return str(e)

    # ---- audio (Gemini) ----
    def ask_voice(self, audio, mime, ward_data):
        try:
            out = self._call(SYSTEM, f"<ward_data>\n{ward_data}\n</ward_data>\n\n{VOICE_QUESTION_PROMPT}",
                             audio=audio, mime=mime, json_out=True)
            return _json(out, {"question": "", "answer": out})
        except AIError as e:
            return {"question": "", "answer": str(e)}

    def voice_note(self, audio, mime):
        try:
            out = self._call("You transcribe and structure nursing notes.", NOTE_PROMPT,
                             audio=audio, mime=mime, json_out=True)
            note = _json(out, {"transcript": out})
            note.setdefault("summary", note.get("transcript", ""))
            return note
        except AIError as e:
            return {"error": str(e)}


def _json(text, fallback):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else fallback
    except ValueError:
        return fallback
