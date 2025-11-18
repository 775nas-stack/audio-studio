"""TorchCREPE inference helpers."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchcrepe
from torchcrepe import core as torchcrepe_core

from .types import ModelMissingError, PitchTrack
from .utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    FMAX,
    FMIN,
    FRAME_HOP,
    time_axis_for_frames,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "full"
MODEL_MESSAGE = (
    "Download full.pth from https://github.com/maxrmorrison/torchcrepe/raw/main/torchcrepe/assets/full.pth "
    "and place it at backend/vendor/torchcrepe/full.pth"
)

_MODEL_CACHE: tuple[torch.nn.Module, torch.device] | None = None

_HQ_DEFAULT_BATCH = 128
_HQ_DEFAULT_CHUNK_SECONDS = 20.0
_HQ_OVERSAMPLE_FLAG = "false"
_HQ_BATCH_SIZE = _env_int("AUDIO_STUDIO_TORCHCREPE_HQ_BATCH", _HQ_DEFAULT_BATCH)
_HQ_CHUNK_SECONDS = _env_float(
    "AUDIO_STUDIO_TORCHCREPE_HQ_CHUNK_SECONDS",
    _HQ_DEFAULT_CHUNK_SECONDS,
)
_HQ_OVERSAMPLE = _env_flag("AUDIO_STUDIO_TORCHCREPE_HQ_OVERSAMPLE", _HQ_OVERSAMPLE_FLAG)


def _env_flag(name: str, default: str = "false") -> bool:
    value = os.getenv(name)
    if value is None:
        value = default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _candidate_paths() -> list[Path]:
    backend_dir = Path(__file__).resolve().parent.parent
    repo_root = backend_dir.parent
    return [
        backend_dir / "vendor" / "torchcrepe" / "full.pth",
        backend_dir / "vendor" / "torchcrepe" / "torchcrepe-full.pth",
        repo_root / "models" / "torchcrepe" / "full.pth",
    ]


def _load_model(device: torch.device) -> torch.nn.Module:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        cached_model, cached_device = _MODEL_CACHE
        if cached_device == device:
            return cached_model

    for candidate in _candidate_paths():
        if candidate.exists():
            weights_path = candidate
            break
    else:  # pragma: no cover - file system fallback
        raise ModelMissingError("torchcrepe_full", MODEL_MESSAGE)

    LOGGER.info("[torchcrepe-load] Using weights: %s", weights_path)

    load_kwargs = {"map_location": device}
    try:
        state = torch.load(weights_path, weights_only=True, **load_kwargs)
    except TypeError:  # pragma: no cover - backwards compatibility
        state = torch.load(weights_path, **load_kwargs)

    model = torchcrepe.Crepe(MODEL_NAME)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    torchcrepe_core.infer.model = model
    torchcrepe_core.infer.capacity = MODEL_NAME

    _MODEL_CACHE = (model, device)
    return model


def _hq_hop_length(sample_rate: int, oversample: bool) -> int:
    hop_ms = 5.0 if oversample else 10.0
    hop = int(round(sample_rate * (hop_ms / 1000.0)))
    return max(1, hop)


def _hq_chunk_samples(sample_rate: int, hop_length: int) -> int:
    base = int(sample_rate * _HQ_CHUNK_SECONDS)
    min_chunk = hop_length * 128
    return max(min_chunk, base)


def _match_length(data: np.ndarray, target: int) -> np.ndarray:
    if data.shape[0] == target:
        return data
    if data.shape[0] == 0:
        return np.zeros(target, dtype=data.dtype)
    if data.shape[0] > target:
        return data[:target]
    pad_value = data[-1]
    pad = np.full(target - data.shape[0], pad_value, dtype=data.dtype)
    return np.concatenate([data, pad])


def _compute_rms_track(audio_tensor: torch.Tensor, hop_length: int, sample_rate: int) -> np.ndarray:
    window = max(int(sample_rate * 0.04), hop_length * 2)
    samples = audio_tensor.unsqueeze(1)  # (batch, 1, n)
    pad = window // 2
    padded = F.pad(samples, (pad, pad), mode="reflect")
    energy = F.avg_pool1d(padded.pow(2), kernel_size=window, stride=hop_length)
    rms = torch.sqrt(torch.clamp(energy, min=1e-12)).squeeze(0).squeeze(0)
    return rms.detach().cpu().numpy()


def _predict_hq(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    total_samples = audio_tensor.shape[-1]
    chunk_samples = _hq_chunk_samples(sample_rate, hop_length)
    frequencies: list[np.ndarray] = []
    periodicities: list[np.ndarray] = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio_tensor[:, start:end]
        if chunk.shape[-1] == 0:
            continue
        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(
                chunk,
                sample_rate,
                hop_length=hop_length,
                fmin=FMIN,
                fmax=FMAX,
                model=MODEL_NAME,
                batch_size=batch_size,
                device=device,
                return_periodicity=True,
                pad=True,
            )

        frequencies.append(pitch.squeeze(0).squeeze(0).detach().cpu().numpy())
        periodicities.append(periodicity.squeeze(0).squeeze(0).detach().cpu().numpy())

    if not frequencies:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    frequency = np.concatenate(frequencies)
    periodicity = np.concatenate(periodicities)
    return frequency, periodicity


def run_torchcrepe_hq(
    audio: np.ndarray,
    sr: int,
    *,
    batch_size: int | None = None,
    oversample: bool | None = None,
) -> PitchTrack:
    """High-fidelity TorchCREPE inference with tight hop size and diagnostics."""

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if oversample is None:
        oversample = _HQ_OVERSAMPLE
    hop_length = _hq_hop_length(sr, oversample)
    effective_batch = batch_size or _HQ_BATCH_SIZE

    try:
        _load_model(device)
    except ModelMissingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("TorchCREPE failed to load: %s", exc)
        raise

    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    try:
        start = time.perf_counter()
        frequency, periodicity = _predict_hq(
            audio_tensor,
            sr,
            hop_length=hop_length,
            batch_size=effective_batch,
            device=device,
        )
        duration = time.perf_counter() - start
        LOGGER.info(
            "[engine-success-detail] engine=torchcrepe_hq device=%s hop=%d batch=%d duration=%.2fs",
            device_name,
            hop_length,
            effective_batch,
            duration,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("TorchCREPE HQ inference failed: %s", exc)
        raise

    confidence = periodicity.copy()
    low_conf = confidence < DEFAULT_CONFIDENCE_THRESHOLD
    frequency = frequency.astype(float, copy=True)
    frequency[low_conf] = np.nan
    confidence[low_conf] = 0.0

    time_axis = np.arange(frequency.shape[0], dtype=float) * (hop_length / sr)
    loudness = _match_length(_compute_rms_track(audio_tensor, hop_length, sr), frequency.shape[0])
    voicing = periodicity

    return PitchTrack(
        time=time_axis,
        frequency=frequency,
        confidence=confidence,
        engine="torchcrepe_hq",
        loudness=loudness,
        voicing=voicing,
    )


def run_torchcrepe_full(audio: np.ndarray, sr: int) -> PitchTrack:
    """Run TorchCREPE full model and return a rich PitchTrack."""

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    try:
        _load_model(device)
    except ModelMissingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("TorchCREPE failed to load: %s", exc)
        raise

    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    try:
        start = time.perf_counter()
        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(
                audio_tensor,
                sr,
                hop_length=FRAME_HOP,
                fmin=FMIN,
                fmax=FMAX,
                model=MODEL_NAME,
                batch_size=64,
                device=device,
                return_periodicity=True,
                pad=True,
            )
        duration = time.perf_counter() - start
        LOGGER.info(
            "[engine-success-detail] engine=torchcrepe_full device=%s duration=%.2fs",
            device_name,
            duration,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("TorchCREPE inference failed: %s", exc)
        raise

    frequency = pitch.squeeze(0).squeeze(0).detach().cpu().numpy()
    confidence = periodicity.squeeze(0).squeeze(0).detach().cpu().numpy()

    low_conf = confidence < DEFAULT_CONFIDENCE_THRESHOLD
    frequency[low_conf] = np.nan
    confidence[low_conf] = 0.0

    time_axis = time_axis_for_frames(frequency.shape[0])
    return PitchTrack(time=time_axis, frequency=frequency, confidence=confidence, engine="torchcrepe_full")


__all__ = ["run_torchcrepe_full", "run_torchcrepe_hq"]
