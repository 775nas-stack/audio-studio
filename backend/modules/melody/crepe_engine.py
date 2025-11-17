"""Melody extraction using the CREPE ONNX model.

The routines defined here perform the low level pitch extraction from an
uploaded waveform.  The expected workflow is:

1.  Load and resample the audio to 16 kHz mono.
2.  Slice the audio into CREPE compatible frames (1024 samples, 10 ms hop).
3.  Run the ``crepe_full.onnx`` model with ``onnxruntime``.
4.  Convert the model logits to a frequency track and keep only confident
    frames (confidence >= 0.6).
5.  Persist the raw melody track to ``melody_raw.json`` inside the provided
    project directory.

The public ``extract_melody`` function returns the same data structure that is
stored on disk so the subsequent smoothing and MIDI generation steps can
operate on it directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

try:  # Optional import – raise a friendly error if not installed when used.
    import librosa
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "librosa is required for melody extraction but is not installed"
    ) from exc


CREPE_SAMPLE_RATE = 16_000
CREPE_FRAME_SIZE = 1024
CREPE_HOP_LENGTH = int(CREPE_SAMPLE_RATE * 0.01)  # 10 ms
CREPE_MIN_FREQUENCY = 32.703195662574764  # C1
CREPE_BINS = 360
CONFIDENCE_THRESHOLD = 0.6


class MelodyExtractionError(RuntimeError):
    """Raised when melody extraction fails."""


def _softmax(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _crepe_frequency_lookup() -> np.ndarray:
    # CREPE uses 20-cent spacing across 6 octaves.
    bins = np.arange(CREPE_BINS, dtype=np.float32)
    return CREPE_MIN_FREQUENCY * np.power(2.0, bins / 60.0)


CREPE_FREQUENCIES = _crepe_frequency_lookup()


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _prepare_audio_frames(audio: np.ndarray) -> np.ndarray:
    if audio.size < CREPE_FRAME_SIZE:
        pad_amount = CREPE_FRAME_SIZE - audio.size
        audio = np.pad(audio, (0, pad_amount), mode="constant")
    frames = librosa.util.frame(
        audio, frame_length=CREPE_FRAME_SIZE, hop_length=CREPE_HOP_LENGTH
    ).T
    return frames.astype(np.float32)


class CrepeOnnxEngine:
    """Wraps the CREPE ONNX model inference session."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        base_path = Path(__file__).resolve().parents[4]
        default_model = base_path / "models" / "melody" / "crepe_full.onnx"
        self.model_path = Path(model_path) if model_path else default_model
        if not self.model_path.exists():
            raise MelodyExtractionError(
                f"CREPE model not found at {self.model_path}. Please ensure the "
                "onnx file is available."
            )
        self._session = None

    def _session_inputs(self):  # pragma: no cover - helper
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - handled at runtime
                raise MelodyExtractionError(
                    "onnxruntime is required to run the CREPE model"
                ) from exc

            try:
                self._session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                raise MelodyExtractionError(
                    f"Failed to initialize CREPE model: {exc}"
                ) from exc

        return self._session, self._session.get_inputs()[0].name

    def _run_model(self, frames: np.ndarray) -> np.ndarray:
        session, input_name = self._session_inputs()
        # Match the required dimensionality automatically.
        model_input = frames
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) == 3 and model_input.ndim == 2:
            model_input = model_input[:, :, np.newaxis]
        elif len(input_shape) == 4 and model_input.ndim == 2:
            model_input = model_input[:, np.newaxis, :, np.newaxis]
        try:
            outputs = session.run(None, {input_name: model_input})
        except Exception as exc:
            raise MelodyExtractionError(f"CREPE inference failed: {exc}") from exc
        return outputs[0]


def _load_audio(audio_path: str) -> np.ndarray:
    if not os.path.exists(audio_path):
        raise MelodyExtractionError(f"Audio file not found: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=CREPE_SAMPLE_RATE, mono=True)
    if not np.any(np.abs(audio) > 1e-6):
        raise MelodyExtractionError("Uploaded audio appears to be silent.")
    audio = audio.astype(np.float32)
    audio /= max(np.max(np.abs(audio)), 1e-6)
    return audio


def extract_melody(audio_path: str, project_dir: str) -> Dict[str, List[float]]:
    """Run CREPE on the provided audio file and persist the raw melody track."""

    engine = CrepeOnnxEngine()
    audio = _load_audio(audio_path)
    frames = _prepare_audio_frames(audio)
    logits = engine._run_model(frames)
    probabilities = _softmax(logits)
    best_idx = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    frequency = CREPE_FREQUENCIES[best_idx]
    times = np.arange(len(frequency)) * (CREPE_HOP_LENGTH / CREPE_SAMPLE_RATE)

    mask = confidence >= CONFIDENCE_THRESHOLD
    if not np.any(mask):
        raise MelodyExtractionError("No confident melody frames detected in audio.")

    result = {
        "time": times[mask].astype(float).tolist(),
        "frequency": frequency[mask].astype(float).tolist(),
        "confidence": confidence[mask].astype(float).tolist(),
        "sr": CREPE_SAMPLE_RATE,
    }

    project_path = Path(project_dir)
    _ensure_directory(project_path)
    raw_path = project_path / "melody_raw.json"
    with raw_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    return result


__all__ = ["extract_melody", "MelodyExtractionError"]
