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
    """Cast the buffer to float32 without altering its dynamics."""

    if audio.dtype == np.float32:
        return audio
    return audio.astype(np.float32)


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert arbitrary channel layouts to mono without shrinking length."""

    if audio.ndim == 1:
        return audio

    # SoundFile returns arrays shaped (frames, channels); transpose so librosa
    # receives the (channels, frames) layout it expects, regardless of frame
    # count, to avoid collapsing long clips to a single sample.
    return librosa.to_mono(audio.T)


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


def _db_to_amplitude(db_value: float) -> float:
    return float(np.power(10.0, db_value / 20.0))


def _high_pass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 55.0) -> np.ndarray:
    """Apply a simple first-order high-pass filter."""

    if audio.size == 0:
        return audio
    if cutoff_hz <= 0:
        return audio
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / max(sr, 1)
    alpha = rc / (rc + dt)
    filtered = np.empty_like(audio)
    filtered[0] = audio[0]
    prev_output = filtered[0]
    prev_input = audio[0]
    for idx in range(1, audio.size):
        current = audio[idx]
        prev_output = alpha * (prev_output + current - prev_input)
        filtered[idx] = prev_output
        prev_input = current
    return filtered


def normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize audio so its peak matches the requested dBFS."""

    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        return audio
    target_amp = _db_to_amplitude(target_db)
    if target_amp <= 0:
        return audio
    scale = target_amp / peak
    return audio * scale


def preprocess_pitch_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Minimal preprocessing: convert to mono, resample, keep raw dynamics."""

    if audio.ndim > 1:
        audio = _to_mono(audio)

    if sr != TARGET_SAMPLE_RATE:
        audio = _resample(audio, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    return _ensure_float32(audio)
