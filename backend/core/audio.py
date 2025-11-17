from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
PEAK_TARGET = 0.9


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 at 16 kHz with gentle normalization."""

    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    peak = np.max(np.abs(data))
    if peak > 0:
        data = data * (PEAK_TARGET / peak)

    return data.astype(np.float32), sr
