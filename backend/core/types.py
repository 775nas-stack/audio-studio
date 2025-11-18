from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


class NoMelodyError(RuntimeError):
    """Raised when no stable melody can be produced."""


class ModelMissingError(RuntimeError):
    """Raised when a required model file is missing from disk."""

    def __init__(self, model_name: str, instructions: str | None = None):
        message = f"Model missing: {model_name}"
        if instructions:
            message = f"{message}. {instructions}"
        super().__init__(message)
        self.model_name = model_name
        self.instructions = instructions


@dataclass
class PitchTrack:
    time: np.ndarray
    frequency: np.ndarray
    confidence: np.ndarray
    engine: Literal[
        "crepe",
        "crepe_tiny",
        "torchcrepe",
        "torchcrepe_full",
        "pyin",
        "hybrid",
        "unknown",
    ]
    activation: np.ndarray | None = None
    loudness: np.ndarray | None = None
    sources: np.ndarray | None = None

    def finite_mask(self) -> np.ndarray:
        return np.isfinite(self.frequency)

    def finite_count(self) -> int:
        return int(self.finite_mask().sum())

    def to_payload(self, include_activation: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "time": self.time.tolist(),
            "frequency": self.frequency.tolist(),
            "confidence": self.confidence.tolist(),
            "engine": self.engine,
        }
        if self.loudness is not None:
            payload["loudness"] = self.loudness.tolist()
        if self.sources is not None:
            payload["sources"] = self.sources.tolist()
        if include_activation and self.activation is not None:
            payload["activation"] = self.activation.tolist()
        return payload
