"""Routing helpers for pitch extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import load_audio
from .debug import write_debug_file
from .pitch_engine import run_fallback, run_primary
from .pyin_engine import run_pyin
from .types import ModelMissingError, NoMelodyError, PitchTrack

LOGGER = logging.getLogger(__name__)

ENGINE_PRIORITY = ("torchcrepe_full", "pyin", "fallback")
ENGINE_RUNNERS: dict[str, Callable[[np.ndarray, int], PitchTrack]] = {
    "torchcrepe_full": run_primary,
    "pyin": run_pyin,
    "fallback": run_fallback,
}
ENGINE_CHOICES = tuple(ENGINE_RUNNERS.keys())


def _ordered_engines(requested: str | None) -> list[str]:
    if requested is None:
        order = list(ENGINE_PRIORITY)
    else:
        if requested not in ENGINE_RUNNERS:
            raise ValueError(f"Unknown pitch engine '{requested}'.")
        order = [requested]
        order.extend(name for name in ENGINE_PRIORITY if name not in order)
    for name in ENGINE_RUNNERS:
        if name not in order:
            order.append(name)
    return order


def extract_unified_pitch(
    audio: np.ndarray,
    sr: int,
    requested_engine: str | None = None,
    debug_dir: Path | None = None,
) -> PitchTrack:
    """Try engines in priority order until one returns usable frames."""

    attempts: list[dict[str, object]] = []
    last_model_error: ModelMissingError | None = None

    for engine_name in _ordered_engines(requested_engine):
        runner = ENGINE_RUNNERS[engine_name]
        try:
            track = runner(audio, sr)
        except ModelMissingError as exc:
            attempts.append({"engine": engine_name, "status": "missing", "error": str(exc)})
            LOGGER.warning("Engine %s unavailable: %s", engine_name, exc)
            if requested_engine == engine_name:
                write_debug_file(debug_dir, "engine_attempts.json", attempts)
                raise
            last_model_error = exc
            continue

        frames = track.finite_count()
        attempts.append({"engine": engine_name, "status": "ok" if frames else "empty", "frames": frames})
        if frames:
            write_debug_file(debug_dir, "engine_attempts.json", attempts)
            write_debug_file(debug_dir, f"pitch_{engine_name}.json", track.to_payload(include_activation=True))
            LOGGER.info("Selected engine %s (%s frames)", engine_name, frames)
            return track

    write_debug_file(debug_dir, "engine_attempts.json", attempts)

    if last_model_error is not None:
        raise last_model_error

    raise NoMelodyError("No stable monophonic melody detected.")


def extract_pitch_pipeline(audio_path: Path, engine: str | None = None, debug_dir: Path | None = None) -> PitchTrack:
    """Load audio from disk and run the unified pitch extraction pipeline."""

    audio, sr = load_audio(audio_path)
    duration = float(len(audio) / sr) if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    write_debug_file(
        debug_dir,
        "audio_stats.json",
        {"path": str(audio_path), "sample_rate": sr, "duration": duration, "rms": rms},
    )
    return extract_unified_pitch(audio, sr, requested_engine=engine, debug_dir=debug_dir)


__all__ = [
    "ENGINE_CHOICES",
    "ENGINE_PRIORITY",
    "extract_pitch_pipeline",
    "extract_unified_pitch",
]
