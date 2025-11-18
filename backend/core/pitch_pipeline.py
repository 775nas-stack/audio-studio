from __future__ import annotations

from backend.core.pitch_engine import run_primary, run_fallback

ENGINE_PRIORITY = ["primary", "fallback"]


def extract_pitch(audio, sr):
    try:
        pitch, conf = run_primary(audio, sr)
        return pitch, conf
    except Exception:
        pitch, conf = run_fallback(audio, sr)
        return pitch, conf
