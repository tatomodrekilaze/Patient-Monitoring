"""Check that Gemini on Vertex AI works before the demo.

    gcloud auth application-default login
    python tools/check_vertex.py --project YOUR_PROJECT_ID [--model gemini-3.8-flash] [--audio note.webm]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.assistant import WardAssistant  # noqa: E402
from monitor.config import make_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--region", default="global")
ap.add_argument("--model", default=None)
ap.add_argument("--audio", help="optional Georgian voice recording to test transcription")
a = ap.parse_args()

cfg = make_config(vertex_project=a.project, vertex_region=a.region, gemini_model=a.model, ai_provider="gemini")
ai = WardAssistant(cfg)
print(f"Model: {cfg.gemini_model}  Project: {a.project}  Region: {a.region}")
print("Text test ->", ai.ask("Are all beds stable? One sentence.",
                             '{"beds_by_risk": [{"name": "Bed 1", "level": "green", "rr": 15}]}'))
if a.audio:
    mime = "audio/webm" if a.audio.endswith(".webm") else "audio/mpeg" if a.audio.endswith(".mp3") else "audio/wav"
    with open(a.audio, "rb") as f:
        print("Voice note ->", ai.voice_note(f.read(), mime))
