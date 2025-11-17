"""Pitch extraction pipeline using CREPE with a PYIN fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.core import audio_utils

try:  # Guard heavy dependencies so failures surface clearly.
    import librosa
except Exception as exc:  # pragma: no cover - handled dynamically
    raise RuntimeError("librosa is required for pitch extraction") from exc


CREPE_SAMPLE_RATE = audio_utils.TARGET_SAMPLE_RATE
CREPE_FRAME_SIZE = 1024
CREPE_STEP_SIZE_MS = 10.0
CREPE_BINS = 360
CREPE_MIN_FREQUENCY = 32.703195662574764  # C1
MIN_USABLE_FRAMES = 20
CONFIDENCE_THRESHOLD = 0.25


def _softmax(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _frequency_lookup() -> np.ndarray:
    bins = np.arange(CREPE_BINS, dtype=np.float32)
    return CREPE_MIN_FREQUENCY * np.power(2.0, bins / 60.0)


CREPE_FREQUENCIES = _frequency_lookup()


class PitchExtractionError(RuntimeError):
    """Raised when no monophonic melody contour can be estimated."""


@dataclass
class PitchTrack:
    time: np.ndarray
    frequency: np.ndarray
    confidence: np.ndarray
    source: str
    sr: int

    def finite_mask(self) -> np.ndarray:
        freq_mask = np.isfinite(self.frequency) & (self.frequency > 0)
        conf_mask = self.confidence > 0
        return freq_mask & conf_mask

    def finite_count(self) -> int:
        return int(np.count_nonzero(self.finite_mask()))

    def stats(self) -> Dict[str, float]:
        mask = self.finite_mask()
        if not np.any(mask):
            return {
                "finite_frames": 0,
                "median_confidence": float(np.median(self.confidence)) if self.confidence.size else 0.0,
                "median_frequency": 0.0,
                "min_frequency": 0.0,
                "max_frequency": 0.0,
            }
        freqs = self.frequency[mask]
        conf = self.confidence[mask]
        return {
            "finite_frames": int(mask.sum()),
            "median_confidence": float(np.median(conf)) if conf.size else 0.0,
            "median_frequency": float(np.median(freqs)),
            "min_frequency": float(np.min(freqs)),
            "max_frequency": float(np.max(freqs)),
        }

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "time": self.time.astype(float).tolist(),
            "frequency": self.frequency.astype(float).tolist(),
            "confidence": self.confidence.astype(float).tolist(),
            "sr": self.sr,
            "source": self.source,
        }


class CREPERunner:
    """Runs the bundled TensorFlow CREPE model with a musical PYIN fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        step_size_ms: float = CREPE_STEP_SIZE_MS,
    ) -> None:
        base = Path(__file__).resolve().parents[2]
        default_model = base / "models" / "melody" / "model.h5"
        self.model_path = Path(model_path) if model_path else default_model
        if not self.model_path.exists():
            raise FileNotFoundError(f"CREPE model not found at {self.model_path}")
        hop = int(round(CREPE_SAMPLE_RATE * (step_size_ms / 1000.0)))
        self.hop_length = max(1, hop)
        self.step_size_ms = step_size_ms
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_audio(self, audio_path: str | Path) -> Dict[str, List[float]]:
        audio, sr = audio_utils.load_audio_mono_16k(audio_path)
        print(f"[CREPERunner] Loaded audio '{audio_path}' with {audio.size} samples")
        track = self.process(audio, sr)
        track_dict = track.to_dict()
        track_dict["humming_mode"] = track.source == "pyin"
        return track_dict

    def process(self, audio: np.ndarray, sr: int) -> PitchTrack:
        crepe_track = self._crepe_predict(audio, sr)
        crepe_stats = crepe_track.stats()
        self._log_track_stats("CREPE", crepe_stats)

        pyin_track: Optional[PitchTrack] = None
        if crepe_stats["finite_frames"] < MIN_USABLE_FRAMES or crepe_stats["median_confidence"] < CONFIDENCE_THRESHOLD:
            pyin_track = self._pyin_predict(audio, sr)
        else:
            # CREPE looked okay, but keep PYIN as a fallback if it still fails downstream.
            if crepe_stats["finite_frames"] == 0:
                pyin_track = self._pyin_predict(audio, sr)

        if pyin_track is not None:
            pyin_stats = pyin_track.stats()
            self._log_track_stats("PYIN", pyin_stats)
            track = self._choose_track(crepe_track, pyin_track)
        else:
            track = crepe_track

        if track.finite_count() == 0:
            raise PitchExtractionError(
                "No stable monophonic melody detected in the audio."
            )

        return track

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_model(self):  # pragma: no cover - heavy dependency
        if self._model is None:
            try:
                from tensorflow import keras
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("TensorFlow is required to run CREPE") from exc
            self._model = keras.models.load_model(str(self.model_path), compile=False)
        return self._model

    def _prepare_frames(self, audio: np.ndarray) -> np.ndarray:
        if audio.size < CREPE_FRAME_SIZE:
            pad = CREPE_FRAME_SIZE - audio.size
            audio = np.pad(audio, (0, pad), mode="constant")
        frames = librosa.util.frame(audio, frame_length=CREPE_FRAME_SIZE, hop_length=self.hop_length)
        return frames.T.astype(np.float32)

    def _crepe_predict(self, audio: np.ndarray, sr: int) -> PitchTrack:
        model = self._load_model()
        frames = self._prepare_frames(audio)
        model_input = frames[:, :, np.newaxis]
        logits = model.predict(model_input, verbose=0)
        if logits.ndim == 4:
            logits = np.squeeze(logits, axis=(1, 3))
        elif logits.ndim == 3:
            logits = np.squeeze(logits, axis=2)
        probabilities = _softmax(logits)
        best_idx = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1).astype(np.float32)
        frequency = CREPE_FREQUENCIES[best_idx].astype(np.float32)
        times = np.arange(len(frequency), dtype=np.float32) * (self.hop_length / sr)
        return PitchTrack(
            time=times,
            frequency=frequency,
            confidence=confidence,
            source="crepe",
            sr=sr,
        )

    def _pyin_predict(self, audio: np.ndarray, sr: int) -> PitchTrack:
        frame_length = 2048
        hop_length = max(1, int(round(sr * (self.step_size_ms / 1000.0))))
        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz("C6")
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            fill_na=np.nan,
        )
        if f0 is None:
            f0 = np.array([], dtype=np.float32)
        if voiced_flag is None:
            voiced_flag = np.zeros_like(f0, dtype=bool)
        if voiced_prob is None:
            voiced_prob = np.zeros_like(f0, dtype=np.float32)
        time = np.arange(len(f0), dtype=np.float32) * (hop_length / sr)
        confidence = voiced_prob if voiced_prob is not None else None
        if confidence is None or confidence.size == 0:
            confidence = voiced_flag.astype(np.float32)
        else:
            confidence = np.nan_to_num(confidence, nan=0.0).astype(np.float32)
        freq = np.array(f0, dtype=np.float32)
        freq[~np.asarray(voiced_flag, dtype=bool)] = np.nan
        return PitchTrack(
            time=time,
            frequency=freq,
            confidence=confidence,
            source="pyin",
            sr=sr,
        )

    def _choose_track(self, crepe_track: PitchTrack, pyin_track: PitchTrack) -> PitchTrack:
        if crepe_track.finite_count() == 0 and pyin_track.finite_count() > 0:
            print("[CREPERunner] CREPE unusable – switching to PYIN")
            return pyin_track
        if pyin_track.finite_count() == 0:
            return crepe_track
        crepe_stats = crepe_track.stats()
        pyin_stats = pyin_track.stats()
        if pyin_stats["finite_frames"] > crepe_stats["finite_frames"] * 1.2:
            print("[CREPERunner] PYIN selected (more finite frames)")
            return pyin_track
        if pyin_stats["median_confidence"] > crepe_stats["median_confidence"] + 0.1:
            print("[CREPERunner] PYIN selected (higher confidence)")
            return pyin_track
        print("[CREPERunner] CREPE selected")
        return crepe_track

    def _log_track_stats(self, label: str, stats: Dict[str, float]) -> None:
        print(
            f"[CREPERunner] {label} finite={stats['finite_frames']} "
            f"median_conf={stats['median_confidence']:.2f} "
            f"freq_range={stats['min_frequency']:.1f}-{stats['max_frequency']:.1f}"
        )
