"""Audio helpers for loading, resampling and saving waveforms."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

try:  # Optional dependency guarding to provide clearer errors.
    import librosa
except Exception as exc:  # pragma: no cover - handled during runtime
    raise RuntimeError("librosa is required for audio processing") from exc


TARGET_SAMPLE_RATE = 16_000


def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio)) or 1.0
    return audio / max_val


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return librosa.to_mono(audio.T if audio.shape[0] < audio.shape[1] else audio)


def load_audio_bytes(data: bytes, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio from raw bytes and resample to the target sample rate."""

    with sf.SoundFile(io.BytesIO(data)) as sound_file:
        audio = sound_file.read(always_2d=True)
        sr = sound_file.samplerate

    audio = _to_mono(audio)
    audio = _resample(audio, sr, target_sr)
    audio = _ensure_float32(audio)
    return audio, target_sr


def load_audio_file(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file from disk."""

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return _ensure_float32(audio), sr


def save_wav(path: str | Path, audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def trim_leading_trailing_silence(audio: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    start = np.argmax(mask)
    end = len(audio) - np.argmax(mask[::-1])
    return audio[start:end]
