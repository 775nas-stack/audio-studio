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


class PitchExtractionError(RuntimeError):
    """Raised when both CREPE and PYIN fail to detect a melody."""


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
        if not model_path.exists():
            raise FileNotFoundError(f"CREPE model not found at {model_path}")
        self.model_path = model_path
        self.use_pyin_fallback = use_pyin_fallback
        self.step_size_ms = step_size_ms
        hop = int(round(CREPE_SAMPLE_RATE * (self.step_size_ms / 1000.0)))
        self.hop_length = max(1, hop)
        self._model = None
        self._voicing_threshold = 0.5

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
        frequency = np.nan_to_num(frequency, nan=0.0, posinf=0.0, neginf=0.0)
        times = np.arange(len(frequency)) * (self.hop_length / sr)
        return {
            "time": times.astype(float).tolist(),
            "frequency": frequency.astype(float).tolist(),
            "confidence": confidence.astype(float).tolist(),
            "sr": sr,
        }

    def _pyin_humming(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        frame_length = 2048
        hop_length = max(1, int(round(sr * 0.01)))
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=50.0,
            fmax=800.0,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            fill_na=None,
        )
        times = librosa.times_like(
            f0,
            sr=sr,
            hop_length=hop_length,
            n_fft=frame_length,
            center=True,
        )
        frequencies = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        if voiced_prob is None:
            voiced_prob = np.zeros_like(frequencies)
        confidence = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)
        confidence[frequencies <= 0] = 0.0
        if voiced_flag is not None:
            voiced_flag = np.asarray(voiced_flag, dtype=bool)
        else:
            voiced_flag = frequencies > 0
        return {
            "time": times.astype(float).tolist(),
            "frequency": frequencies.astype(float).tolist(),
            "confidence": confidence.astype(float).tolist(),
            "sr": sr,
            "voiced_flag": voiced_flag.astype(bool).tolist(),
            "humming_mode": True,
        }

    def _has_voiced_frames(
        self,
        track: Dict[str, List[float]],
        *,
        min_frames: int,
        use_confidence: bool,
    ) -> bool:
        freqs = np.asarray(track.get("frequency", []), dtype=float)
        if freqs.size == 0:
            return False
        if use_confidence:
            confidence = np.asarray(track.get("confidence", []), dtype=float)
            if confidence.size != freqs.size:
                return False
            mask = confidence >= self._voicing_threshold
        else:
            mask = freqs > 0
        return int(np.count_nonzero(mask)) >= min_frames

    def _crepe_failed(self, track: Dict[str, List[float]]) -> bool:
        freqs = np.asarray(track.get("frequency", []), dtype=float)
        if freqs.size == 0 or np.isnan(freqs).any():
            return True
        return not self._has_voiced_frames(track, min_frames=30, use_confidence=True)

    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio_utils.preprocess_pitch_audio(audio, sr)

    def process_audio(
        self,
        audio_path: str | Path,
        *,
        use_pyin_fallback: bool | None = None,
    ) -> Dict[str, List[float]]:
        """Load audio and run CREPE (or optional PYIN) to extract melody."""

        target_sr = audio_utils.TARGET_SAMPLE_RATE
        audio, _ = audio_utils.load_audio_file(audio_path, target_sr=target_sr)
        audio = self._preprocess_audio(audio, target_sr)
        print(f"[CREPERunner] Loaded audio '{audio_path}' with {audio.size} samples")

        force_pyin = self.use_pyin_fallback if use_pyin_fallback is None else use_pyin_fallback
        if force_pyin:
            print("CREPE failed → PYIN humming mode used")
            pyin_track = self._pyin_humming(audio, target_sr)
            if self._has_voiced_frames(pyin_track, min_frames=1, use_confidence=False):
                return pyin_track
            raise PitchExtractionError("No stable pitch detected in the audio.")

        crepe_track = None
        try:
            crepe_track = self._crepe_predict(audio, target_sr)
        except Exception:
            crepe_track = None

        if crepe_track and not self._crepe_failed(crepe_track):
            print("CREPE used")
            crepe_track["humming_mode"] = False
            return crepe_track

        print("CREPE failed → PYIN humming mode used")
        try:
            pyin_track = self._pyin_humming(audio, target_sr)
        except Exception as exc:  # pragma: no cover - depends on librosa internals
            raise PitchExtractionError("No stable pitch detected in the audio.") from exc

        if self._has_voiced_frames(pyin_track, min_frames=1, use_confidence=False):
            return pyin_track

        raise PitchExtractionError("No stable pitch detected in the audio.")

    def export_raw_track(self, track: Dict[str, List[float]], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fp:
            json.dump(track, fp, indent=2)
