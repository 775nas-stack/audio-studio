"""Pitch estimation entry points."""

from __future__ import annotations

import numpy as np
import torch
import torchcrepe


SAMPLE_RATE = 16000
HOP_LENGTH = 160
BATCH_SIZE = 1024
MODEL_NAME = "full"
CONFIDENCE_THRESHOLD = 0.45


def _to_audio_tensor(audio: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a 1-D numpy array of audio samples into a torch tensor."""

    audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=device)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    return audio_tensor


def run_primary(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run TorchCREPE full model and return (pitch_hz, confidence)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_tensor = _to_audio_tensor(audio, device)

    with torch.no_grad():
        pitch, confidence = torchcrepe.predict(
            audio_tensor,
            SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            model=MODEL_NAME,
            batch_size=BATCH_SIZE,
            device=device,
            return_periodicity=True,
            pad=True,
        )

    pitch = pitch.squeeze(0).squeeze(0).detach().cpu().numpy()
    confidence = confidence.squeeze(0).squeeze(0).detach().cpu().numpy()

    pitch[confidence < CONFIDENCE_THRESHOLD] = 0.0
    return pitch, confidence


def run_fallback(num_frames: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Placeholder fallback implementation that currently returns zeros."""

    return np.zeros(num_frames, dtype=np.float32), np.zeros(num_frames, dtype=np.float32)
