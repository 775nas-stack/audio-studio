"""Pitch estimation entry points."""

from __future__ import annotations

from math import ceil

import numpy as np
import torch
import torchcrepe
import torch.nn.functional as F
from safetensors.torch import load_file

from backend.models.minipitch.model_definition import (
    MODEL_FILENAME,
    TinyMiniPitchNet,
    get_model_dir,
)


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


def _frame_audio(audio: np.ndarray, hop_length: int) -> np.ndarray:
    """Split audio into hop-sized frames (zero-padded as needed)."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    num_samples = audio.shape[0]
    if num_samples == 0:
        audio = np.zeros(1, dtype=np.float32)
        num_samples = 1
    num_frames = max(1, ceil(num_samples / hop_length))
    total_samples = num_frames * hop_length
    pad_amount = total_samples - num_samples
    if pad_amount:
        audio = np.pad(audio, (0, pad_amount))
    frames = audio.reshape(num_frames, hop_length)
    return frames


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


def run_fallback(
    audio: np.ndarray, sample_rate: int | float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run the TinyMiniPitchNet fallback model on CPU."""

    del sample_rate  # The fallback always operates at the CREPE sample rate.
    model_dir = get_model_dir()
    model_path = model_dir / MODEL_FILENAME
    if not model_path.exists():
        msg = "Fallback model missing — run generate_fallback_model.py"
        raise FileNotFoundError(msg)

    state_dict = load_file(str(model_path))
    model = TinyMiniPitchNet()
    model.load_state_dict(state_dict)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    frames = _frame_audio(audio, HOP_LENGTH)
    frame_tensor = torch.from_numpy(frames).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(frame_tensor)
        pitch = F.softplus(outputs[:, 0])
        confidence = torch.sigmoid(outputs[:, 1])

    return pitch.cpu().numpy(), confidence.cpu().numpy()
