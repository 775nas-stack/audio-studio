from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


class NoMelodyError(RuntimeError):
    """Raised when no stable melody can be produced."""


@dataclass
class PitchTrack:
    time: np.ndarray
    frequency: np.ndarray
    confidence: np.ndarray
    engine: Literal["crepe", "pyin"]

    def finite_mask(self) -> np.ndarray:
        return np.isfinite(self.frequency)

    def finite_count(self) -> int:
        return int(self.finite_mask().sum())
