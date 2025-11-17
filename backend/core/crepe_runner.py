"""TensorFlow CREPE runner with optional PYIN fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from backend.core import audio_utils

try:
    import librosa
except Exception as exc:  # pragma: no cover - handled dynamically
    raise RuntimeError("librosa is required for pitch extraction") from exc


CREPE_SAMPLE_RATE = audio_utils.TARGET_SAMPLE_RATE
CREPE_FRAME_SIZE = 1024
CREPE_STEP_SIZE_MS = 10
CREPE_BINS = 360
CREPE_MIN_FREQUENCY = 32.703195662574764  # C1


def _softmax(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _frequency_lookup() -> np.ndarray:
    bins = np.arange(CREPE_BINS, dtype=np.float32)
    return CREPE_MIN_FREQUENCY * np.power(2.0, bins / 60.0)


CREPE_FREQUENCIES = _frequency_lookup()


class CREPERunner:
    """Runs the bundled TensorFlow CREPE model or an explicit PYIN fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        use_pyin_fallback: bool = False,
        step_size_ms: float = CREPE_STEP_SIZE_MS,
    ) -> None:
        base = Path(__file__).resolve().parents[2]
        default_model = base / "models" / "melody" / "model.h5"
        model_path = Path(model_path) if model_path else default_model
        if not default_model.exists():
            raise FileNotFoundError(f"CREPE model not found at {default_model}")
        if model_path != default_model and not model_path.exists():
            raise FileNotFoundError(f"CREPE model not found at {model_path}")
        self.model_path = default_model
        self.use_pyin_fallback = use_pyin_fallback
        self.step_size_ms = step_size_ms
        hop = int(round(CREPE_SAMPLE_RATE * (self.step_size_ms / 1000.0)))
        self.hop_length = max(1, hop)
        self._model = None

    def _load_model(self):  # pragma: no cover - heavy dependency
        if self._model is None:
            try:
                from tensorflow import keras
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise RuntimeError(
                    "TensorFlow is required to run the bundled CREPE model."
                ) from exc
            self._model = keras.models.load_model(str(self.model_path), compile=False)
        return self._model

    def _prepare_frames(self, audio: np.ndarray) -> np.ndarray:
        if audio.size < CREPE_FRAME_SIZE:
            pad = CREPE_FRAME_SIZE - audio.size
            audio = np.pad(audio, (0, pad), mode="constant")
        frames = librosa.util.frame(
            audio, frame_length=CREPE_FRAME_SIZE, hop_length=self.hop_length
        ).T
        return frames.astype(np.float32)

    def _crepe_predict(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        model = self._load_model()
        frames = self._prepare_frames(audio)
        model_input = frames[:, :, np.newaxis]
        try:
            logits = model.predict(model_input, verbose=0)
        except Exception as exc:  # pragma: no cover - depends on tensorflow
            raise RuntimeError(f"CREPE inference failed: {exc}") from exc
        if logits.ndim == 4:
            logits = np.squeeze(logits, axis=(1, 3))
        elif logits.ndim == 3:
            logits = np.squeeze(logits, axis=2)
        probabilities = _softmax(logits)
        best_idx = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        frequency = CREPE_FREQUENCIES[best_idx]
        times = np.arange(len(frequency)) * (self.hop_length / sr)
        return {
            "time": times.astype(float).tolist(),
            "frequency": frequency.astype(float).tolist(),
            "confidence": confidence.astype(float).tolist(),
            "sr": sr,
        }

    def _pyin_fallback(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        frame_length = 2048
        hop_length = frame_length // 4
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=138.0,
            fmax=2000.0,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        mask = ~np.isnan(f0)
        f0 = np.nan_to_num(f0, nan=0.0)
        return {
            "time": times[mask].astype(float).tolist(),
            "frequency": f0[mask].astype(float).tolist(),
            "confidence": voiced_prob[mask].astype(float).tolist(),
            "sr": sr,
        }

    def process_audio(self, audio_path: str | Path, *, use_pyin_fallback: bool | None = None) -> Dict[str, List[float]]:
        """Load audio and run CREPE (or optional PYIN) to extract melody."""

        target_sr = audio_utils.TARGET_SAMPLE_RATE
        audio, _ = audio_utils.load_audio_file(audio_path, target_sr=target_sr)

        fallback = self.use_pyin_fallback if use_pyin_fallback is None else use_pyin_fallback
        if not fallback:
            return self._crepe_predict(audio, target_sr)

        return self._pyin_fallback(audio, target_sr)

    def export_raw_track(self, track: Dict[str, List[float]], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fp:
            json.dump(track, fp, indent=2)
