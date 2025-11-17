from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from ..vendor import crepe
from .types import ModelMissingError, PitchTrack

LOGGER = logging.getLogger(__name__)

STEP_SIZE_MS = 5
MODEL_MESSAGE = (
    "Download model-full.h5 from https://github.com/marl/crepe/raw/master/assets/model-full.h5 "
    "and place it at backend/vendor/crepe/model-full.h5"
)


def _candidate_paths() -> list[Path]:
    base_dir = Path(__file__).resolve().parent.parent
    repo_root = base_dir.parent
    return [
        base_dir / "vendor" / "crepe" / "model-full.h5",
        base_dir / "vendor" / "crepe" / "model.h5",
        repo_root / "models" / "crepe" / "model-full.h5",
        repo_root / "models" / "crepe" / "model.h5",
    ]


def _ensure_model_path() -> Path:
    for candidate in _candidate_paths():
        if candidate.exists():
            os.environ["CREPE_MODEL_PATH"] = str(candidate)
            return candidate
    raise ModelMissingError("crepe", MODEL_MESSAGE)


def run_crepe(audio: np.ndarray, sr: int) -> PitchTrack:
    """Run the CREPE full-capacity model with a high resolution hop size."""

    try:
        model_path = _ensure_model_path()
        LOGGER.debug("Using CREPE weights at %s", model_path)
        time, frequency, confidence, activation = crepe.predict(
            audio,
            sr=sr,
            model_capacity="full",
            step_size=STEP_SIZE_MS,
            verbose=0,
            center=True,
        )
    except ModelMissingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("CREPE inference failed: %s", exc)
        time = np.array([], dtype=float)
        frequency = np.array([], dtype=float)
        confidence = np.array([], dtype=float)
        activation = np.zeros((0, 0), dtype=float)

    return PitchTrack(
        time=time,
        frequency=frequency,
        confidence=confidence,
        activation=activation,
        engine="crepe",
    )
