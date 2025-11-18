"""Fallback MiniPitch model definition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

MODEL_FILENAME = "model.safetensors"


@dataclass
class MiniPitchFallbackModel:
    """A minimal CPU-only MiniPitch model.

    The model does not perform any real inference. Instead, it returns arrays of
    zeros for both pitch and confidence. This allows the rest of the pipeline to
    interact with a predictable object when the real model is unavailable.
    """

    model_path: Path

    def __post_init__(self) -> None:
        self.device = "cpu"
        self._ensure_placeholder_exists()

    def _ensure_placeholder_exists(self) -> None:
        """Ensure that a placeholder model file exists on disk."""
        placeholder_path = self.model_path / MODEL_FILENAME
        if not placeholder_path.exists():
            placeholder_path.write_text("placeholder", encoding="utf-8")

    def predict(self, audio: np.ndarray, sample_rate: int | float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero-valued pitch and confidence arrays.

        Args:
            audio: A 1-D numpy array of audio samples.
            sample_rate: Included for API compatibility. It is unused because the
                fallback model does not run inference.

        Returns:
            A tuple ``(pitch_hz, confidence)`` where both arrays are zeros with a
            length matching the provided audio.
        """

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        num_samples = audio.shape[0]
        pitch_hz = np.zeros(num_samples, dtype=np.float32)
        confidence = np.zeros(num_samples, dtype=np.float32)
        return pitch_hz, confidence


def load_model(model_dir: str | Path | None = None) -> MiniPitchFallbackModel:
    """Load the fallback MiniPitch model from ``model_dir``.

    Args:
        model_dir: Directory that should contain the placeholder ``model.safetensors``
            file. If ``None``, the directory containing this file is used.

    Returns:
        An instance of :class:`MiniPitchFallbackModel` configured to run on CPU.
    """

    if model_dir is None:
        model_dir = Path(__file__).parent
    else:
        model_dir = Path(model_dir)
    return MiniPitchFallbackModel(model_dir)
