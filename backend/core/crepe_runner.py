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
        f0, _, _ = librosa.pyin(
            audio,
            fmin=50.0,
            fmax=800.0,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            fill_na=None,
        )
        if f0 is None:
            f0 = np.array([], dtype=float)
        frame_times = np.arange(len(f0), dtype=float) * (hop_length / sr)
        mask = np.isfinite(f0) & (f0 > 0)
        times = frame_times[mask].astype(float)
        frequencies = f0[mask].astype(np.float32)
        confidence = np.full_like(frequencies, 0.9, dtype=np.float32)
        voiced_flag = np.ones_like(frequencies, dtype=bool)
        return {
            "time": times.tolist(),
            "frequency": frequencies.astype(float).tolist(),
            "confidence": confidence.astype(float).tolist(),
            "sr": sr,
            "voiced_flag": voiced_flag.tolist(),
            "humming_mode": True,
        }

    def _count_finite_frames(self, track: Dict[str, List[float]]) -> int:
        freqs = np.asarray(track.get("frequency", []), dtype=float)
        if freqs.size == 0:
            return 0
        mask = np.isfinite(freqs) & (freqs > 0)
        return int(np.count_nonzero(mask))

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
            pyin_count = self._count_finite_frames(pyin_track)
            print(f"[CREPERunner] PYIN finite frames: {pyin_count}")
            if pyin_count > 0:
                return pyin_track
            raise PitchExtractionError("No stable pitch detected in the audio.")

        crepe_track = None
        try:
            crepe_track = self._crepe_predict(audio, target_sr)
        except Exception:
            crepe_track = None

        if crepe_track:
            crepe_count = self._count_finite_frames(crepe_track)
            print(f"[CREPERunner] CREPE finite frames: {crepe_count}")
            if crepe_count > 0:
                print("CREPE used")
                crepe_track["humming_mode"] = False
                return crepe_track

        print("CREPE failed → PYIN humming mode used")
        try:
            pyin_track = self._pyin_humming(audio, target_sr)
        except Exception as exc:  # pragma: no cover - depends on librosa internals
            raise PitchExtractionError("No stable pitch detected in the audio.") from exc

        pyin_count = self._count_finite_frames(pyin_track)
        print(f"[CREPERunner] PYIN finite frames: {pyin_count}")
        if pyin_count > 0:
            return pyin_track

        raise PitchExtractionError("No stable pitch detected in the audio.")

    def export_raw_track(self, track: Dict[str, List[float]], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fp:
            json.dump(track, fp, indent=2)
