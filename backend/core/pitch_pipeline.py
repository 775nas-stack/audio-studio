from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import librosa
import numpy as np

from .crepe_engine import run_crepe_tiny
from .debug import write_debug_file
from .pyin_engine import run_pyin
from .torchcrepe_engine import run_torchcrepe_full
from .types import ModelMissingError, NoMelodyError, PitchTrack

LOGGER = logging.getLogger(__name__)

TARGET_STEP = 0.005  # 5 ms

ENGINE_PRIORITY: tuple[str, ...] = ("torchcrepe_full", "crepe_tiny", "pyin")
ENGINE_RUNNERS: Dict[str, Callable[[np.ndarray, int], PitchTrack]] = {
    "torchcrepe_full": run_torchcrepe_full,
    "crepe_tiny": run_crepe_tiny,
    "pyin": run_pyin,
}
ENGINE_ALIASES: Dict[str, str] = {
    "torchcrepe": "torchcrepe_full",
    "crepe": "crepe_tiny",
}


def _compute_rms(audio: np.ndarray, sr: int, target_time: np.ndarray) -> np.ndarray | None:
    if target_time.size == 0:
        return None
    hop = max(1, int(round(sr * TARGET_STEP)))
    frame_length = max(2048, hop * 4)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop, center=True)[0]
    rms_time = np.arange(rms.shape[0]) * (hop / sr)
    interp = np.interp(target_time, rms_time, rms)
    max_val = np.max(interp)
    if max_val > 0:
        interp = interp / max_val
    return interp


def _normalize_engine_name(name: str | None) -> str | None:
    if not name:
        return None
    lowered = name.strip().lower()
    lowered = ENGINE_ALIASES.get(lowered, lowered)
    if lowered not in ENGINE_RUNNERS:
        LOGGER.warning("Unknown engine '%s' requested; falling back to defaults.", name)
        return None
    return lowered


def _engine_sequence(preferred: str | None) -> Iterable[str]:
    normalized = _normalize_engine_name(preferred)
    if normalized is None:
        return ENGINE_PRIORITY
    ordered: List[str] = [normalized]
    ordered.extend(name for name in ENGINE_PRIORITY if name not in ordered)
    return tuple(ordered)


def _debug_error_payload(engine: str, error: Exception) -> dict:
    return {
        "time": [],
        "frequency": [],
        "confidence": [],
        "engine": engine,
        "error": f"{error.__class__.__name__}: {error}",
    }


def extract_unified_pitch(audio: np.ndarray, sr: int, requested_engine: str | None = None, debug_dir: Path | None = None) -> PitchTrack:
    errors: List[ModelMissingError] = []
    engines = tuple(_engine_sequence(requested_engine))

    for index, engine_name in enumerate(engines):
        runner = ENGINE_RUNNERS.get(engine_name)
        if runner is None:
            continue
        start = time.perf_counter()
        LOGGER.info("[engine-start] %s", engine_name)
        try:
            track = runner(audio, sr)
        except ModelMissingError as exc:
            errors.append(exc)
            LOGGER.warning("[engine-error] %s error=%s", engine_name, exc)
            write_debug_file(debug_dir, f"raw_{engine_name}.json", _debug_error_payload(engine_name, exc))
            continue
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[engine-error] %s error=%s", engine_name, exc)
            LOGGER.exception("Engine %s crashed", engine_name)
            write_debug_file(debug_dir, f"raw_{engine_name}.json", _debug_error_payload(engine_name, exc))
            continue
        duration = time.perf_counter() - start
        LOGGER.info("[engine-success] %s duration=%.2fs", engine_name, duration)
        LOGGER.info("[engine-final] %s", engine_name)

        for skipped in engines[index + 1 :]:
            LOGGER.info("[engine-skipped] %s (previous engine succeeded)", skipped)

        track.engine = engine_name  # normalize legacy names
        loudness = _compute_rms(audio, sr, track.time)
        track.loudness = loudness
        track.sources = np.full(track.time.shape, engine_name, dtype=object)

        write_debug_file(debug_dir, f"raw_{engine_name}.json", track.to_payload(include_activation=(engine_name == "crepe_tiny")))

        return track

    if errors:
        raise errors[0]
    raise NoMelodyError("No stable monophonic melody detected.")
