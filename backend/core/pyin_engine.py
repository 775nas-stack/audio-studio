from __future__ import annotations

import logging

import librosa
import numpy as np

from .types import PitchTrack

LOGGER = logging.getLogger(__name__)


FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C6")
FRAME_LENGTH = 1024
HOP_LENGTH = 160  # 10 ms at 16 kHz
FRAME_DURATION = HOP_LENGTH / 16000.0


def run_pyin(audio: np.ndarray, sr: int) -> PitchTrack:
    try:
        f0, voiced_flag, prob = librosa.pyin(
            audio,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("PYIN inference failed: %s", exc)
        return PitchTrack(
            time=np.array([], dtype=float),
            frequency=np.array([], dtype=float),
            confidence=np.array([], dtype=float),
            engine="pyin",
        )

    if prob is None:
        prob = voiced_flag.astype(float)

    time = np.arange(len(f0)) * FRAME_DURATION
    voiced = voiced_flag.astype(bool)
    freq = np.where(voiced, f0, np.nan)
    conf = np.where(voiced, prob, 0.0)

    return PitchTrack(time=time, frequency=freq, confidence=conf, engine="pyin")
