from __future__ import annotations

import logging
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from .types import ModelMissingError, PitchTrack

LOGGER = logging.getLogger(__name__)

STEP_SIZE_MS = 10
MODEL_CAPACITY = "tiny"
MODEL_MESSAGE = (
    "Download model-tiny.h5 from https://github.com/marl/crepe/raw/master/assets/model-tiny.h5 "
    "and place it at backend/vendor/crepe/model-tiny.h5"
)


def _candidate_paths() -> list[Path]:
    backend_dir = Path(__file__).resolve().parent.parent
    repo_root = backend_dir.parent
    return [
        backend_dir / "vendor" / "crepe" / "model-tiny.h5",
        backend_dir / "vendor" / "crepe" / "model.h5",
        backend_dir / "vendor" / "crepe" / "model-full.h5",
        repo_root / "models" / "crepe" / "model-tiny.h5",
        repo_root / "models" / "crepe" / "model.h5",
        repo_root / "models" / "crepe" / "model-full.h5",
    ]


def _ensure_model_path() -> Path:
    for candidate in _candidate_paths():
        if candidate.exists():
            os.environ["CREPE_MODEL_PATH"] = str(candidate)
            return candidate
    raise ModelMissingError("crepe_tiny", MODEL_MESSAGE)


def run_crepe_tiny(audio: np.ndarray, sr: int) -> PitchTrack:
    """Run the lightweight CREPE tiny model using vendored weights."""

    try:
        from ..vendor import crepe
        model_path = _ensure_model_path()
        LOGGER.debug("Using CREPE tiny weights at %s", model_path)
        time, frequency, confidence, activation = crepe.predict(
            audio,
            sr=sr,
            model_capacity=MODEL_CAPACITY,
            step_size=STEP_SIZE_MS,
            verbose=0,
            center=True,
        )
    except ModelMissingError:
        raise
    except ImportError as exc:
        instructions = "Install the optional CREPE dependencies (tensorflow, hmmlearn)."
        raise ModelMissingError("crepe_tiny", instructions) from exc
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("CREPE tiny inference failed: %s", exc)
        time = np.array([], dtype=float)
        frequency = np.array([], dtype=float)
        confidence = np.array([], dtype=float)
        activation = np.zeros((0, 0), dtype=float)

    return PitchTrack(
        time=time,
        frequency=frequency,
        confidence=confidence,
        activation=activation,
        engine="crepe_tiny",
    )
