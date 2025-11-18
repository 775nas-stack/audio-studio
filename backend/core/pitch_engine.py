"""Pitch estimation engines and fallback handling."""

from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import load_file

from backend.models.minipitch import MODEL_FILENAME, TinyMiniPitchNet, get_model_dir

from .torchcrepe_engine import run_torchcrepe_full
from .types import ModelMissingError, PitchTrack
from .utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    FMAX,
    FMIN,
    FRAME_HOP,
    time_axis_for_frames,
)

_FALLBACK_MODEL: TinyMiniPitchNet | None = None


def run_primary(audio: np.ndarray, sample_rate: int) -> PitchTrack:
    """Run the TorchCREPE full-capacity network on the best available device."""

    return run_torchcrepe_full(audio, sample_rate)


def _frame_audio(audio: np.ndarray) -> np.ndarray:
    """Split audio into non-overlapping frames used by the fallback model."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        samples = np.zeros(1, dtype=np.float32)
    hop = FRAME_HOP
    total_frames = int(np.ceil(samples.size / hop))
    total_samples = total_frames * hop
    if samples.size < total_samples:
        pad = np.zeros(total_samples - samples.size, dtype=np.float32)
        samples = np.concatenate([samples, pad])
    return samples.reshape(total_frames, hop)


def _load_fallback_model() -> TinyMiniPitchNet:
    """Load TinyMiniPitchNet weights once and cache the module."""

    global _FALLBACK_MODEL, _FALLBACK_WEIGHTS
    if _FALLBACK_MODEL is not None:
        return _FALLBACK_MODEL

    model_dir = get_model_dir()
    model_path = model_dir / MODEL_FILENAME
    if not model_path.exists():
        instructions = "Run `python -m backend.models.minipitch.generate_fallback_model`."
        raise ModelMissingError("minipitch_fallback", instructions)

    state_dict = load_file(str(model_path))
    model = TinyMiniPitchNet()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(torch.device("cpu"))

    _FALLBACK_MODEL = model
    _FALLBACK_WEIGHTS = model_path
    return model


def run_fallback(audio: np.ndarray, sample_rate: int | float | None = None) -> PitchTrack:
    """Run the TinyMiniPitchNet fallback model on CPU only."""

    del sample_rate  # The fallback always operates at the unified CREPE sample rate.

    model = _load_fallback_model()
    device = torch.device("cpu")

    frames = _frame_audio(audio)
    frame_tensor = torch.from_numpy(frames).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(frame_tensor)
        pitch_logits = outputs[:, 0]
        confidence_logits = outputs[:, 1]
        pitch = torch.sigmoid(pitch_logits) * (FMAX - FMIN) + FMIN
        confidence = torch.sigmoid(confidence_logits)

    pitch_np = pitch.cpu().numpy()
    confidence_np = confidence.cpu().numpy()

    low_conf_mask = confidence_np < DEFAULT_CONFIDENCE_THRESHOLD
    pitch_np[low_conf_mask] = np.nan
    confidence_np[low_conf_mask] = 0.0

    time_axis = time_axis_for_frames(pitch_np.shape[0])
    return PitchTrack(time=time_axis, frequency=pitch_np, confidence=confidence_np, engine="fallback")


__all__ = ["run_primary", "run_fallback"]
