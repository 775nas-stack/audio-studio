"""Audio helpers for predictable mono, 16 kHz loading and saving."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

try:  # Provide a clear error if librosa is unavailable at runtime.
    import librosa
except Exception as exc:  # pragma: no cover - handled dynamically
    raise RuntimeError("librosa is required for audio processing") from exc


TARGET_SAMPLE_RATE = 16_000
_PEAK_TARGET = 0.9


def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.float32:
        return audio
    return audio.astype(np.float32)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2 and audio.shape[1] == 1:
        return audio[:, 0]
    return np.mean(audio, axis=1)


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or audio.size == 0:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)


def _normalize(audio: np.ndarray, peak: float = _PEAK_TARGET) -> np.ndarray:
    if audio.size == 0:
        return audio
    current_peak = float(np.max(np.abs(audio)))
    if current_peak <= 0:
        return audio
    if peak <= 0:
        return audio
    scale = peak / current_peak
    return (audio * scale).astype(np.float32)


def _finalize(audio: np.ndarray, sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    mono = _to_mono(audio)
    mono = mono.squeeze()
    mono = _ensure_float32(mono)
    mono = _resample(mono, sr, target_sr)
    mono = _normalize(mono)
    return mono.astype(np.float32), target_sr


def load_audio_mono_16k(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load audio from disk, convert to mono and resample to 16 kHz."""

    path = Path(path)
    with sf.SoundFile(path) as sound_file:
        audio = sound_file.read(always_2d=True, dtype="float32")
        sr = sound_file.samplerate
    return _finalize(audio, sr)


def load_audio_bytes(data: bytes, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio from raw bytes using the same mono-16k path."""

    with sf.SoundFile(io.BytesIO(data)) as sound_file:
        audio = sound_file.read(always_2d=True, dtype="float32")
        sr = sound_file.samplerate
    audio, sr = _finalize(audio, sr, target_sr)
    return audio, sr


def load_audio_file(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Backward-compatible helper that mirrors librosa.load behaviour."""

    audio, sr = load_audio_mono_16k(path)
    if target_sr != TARGET_SAMPLE_RATE and sr != target_sr:
        audio = _resample(audio, TARGET_SAMPLE_RATE, target_sr)
        sr = target_sr
    return audio, sr


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


def preprocess_pitch_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Legacy helper kept for compatibility with existing modules."""

    return _finalize(audio, sr, TARGET_SAMPLE_RATE)[0]
