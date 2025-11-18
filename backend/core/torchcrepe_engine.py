from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchcrepe
from torchcrepe import core as torchcrepe_core

from .types import ModelMissingError, PitchTrack

LOGGER = logging.getLogger(__name__)

HOP_LENGTH = 160  # 10 ms @ 16 kHz
FMIN = 50.0
FMAX = 1200.0
MODEL_NAME = "full"
MODEL_MESSAGE = (
    "Download full.pth from https://github.com/maxrmorrison/torchcrepe/raw/main/torchcrepe/assets/full.pth "
    "and place it at backend/vendor/torchcrepe/full.pth"
)

_MODEL_CACHE: tuple[torch.nn.Module, torch.device] | None = None


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
        model, cached_device = _MODEL_CACHE
        if cached_device == device:
            return model

    for candidate in _candidate_paths():
        if candidate.exists():
            weights_path = candidate
            break
    else:
        raise ModelMissingError("torchcrepe_full", MODEL_MESSAGE)

    LOGGER.info(
        "[torchcrepe-load] Using weights: %s (exists=%s)",
        weights_path,
        os.path.exists(weights_path),
    )

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


def run_torchcrepe_full(audio: np.ndarray, sr: int) -> PitchTrack:
    """Run TorchCREPE offline with the vendored full model on the best available device."""

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    try:
        _load_model(device)
    except ModelMissingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("TorchCREPE failed to load: %s", exc)
        return PitchTrack(
            time=np.array([], dtype=float),
            frequency=np.array([], dtype=float),
            confidence=np.array([], dtype=float),
            engine="torchcrepe_full",
        )

    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    try:
        start = time.perf_counter()
        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(
                audio_tensor,
                sr,
                hop_length=HOP_LENGTH,
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
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("TorchCREPE inference failed: %s", exc)
        return PitchTrack(
            time=np.array([], dtype=float),
            frequency=np.array([], dtype=float),
            confidence=np.array([], dtype=float),
            engine="torchcrepe_full",
        )

    frequency = pitch.squeeze(0).squeeze(0).detach().cpu().numpy()
    confidence = periodicity.squeeze(0).squeeze(0).detach().cpu().numpy()
    time = np.arange(frequency.shape[0], dtype=float) * (HOP_LENGTH / sr)

    return PitchTrack(time=time, frequency=frequency, confidence=confidence, engine="torchcrepe_full")
