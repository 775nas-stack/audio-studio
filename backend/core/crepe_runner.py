"""Wrapper around the CREPE model for offline pitch extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from backend.core import audio_utils

try:
    import librosa
except Exception as exc:  # pragma: no cover - handled dynamically
    raise RuntimeError("librosa is required for pitch extraction") from exc

try:  # Optional – if crepe is missing we fall back to PYIN.
    import crepe
except Exception:  # pragma: no cover - fallback handled below
    crepe = None


class CREPERunner:
    """Runs CREPE on audio and returns a dictionary with time/frequency/confidence."""

    def __init__(self, model_path: str | Path | None = None, *, use_pyin_fallback: bool = False) -> None:
        base = Path(__file__).resolve().parents[2]
        default_model = base / "models" / "melody" / "model.h5"
        # Always prefer the bundled TensorFlow CREPE model.
        self.model_path = Path(model_path) if model_path else default_model
        if self.model_path != default_model and not self.model_path.exists():
            raise FileNotFoundError(f"CREPE model not found at {self.model_path}")
        if not default_model.exists():
            raise FileNotFoundError(f"CREPE model not found at {default_model}")
        self.model_path = default_model
        self.use_pyin_fallback = use_pyin_fallback

    def _crepe_predict(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        os.environ.setdefault("CREPE_MODEL", str(self.model_path))
        os.environ.setdefault("CREPE_CACHE_DIR", str(self.model_path.parent))
        time, frequency, confidence, _ = crepe.predict(
            audio,
            sr,
            model_capacity="full",
            step_size=10,
            viterbi=True,
            verbose=0,
        )
        return {
            "time": time.astype(float).tolist(),
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
            if crepe is None:
                raise RuntimeError(
                    "CREPE library is not available. Enable PYIN fallback explicitly to continue."
                )
            return self._crepe_predict(audio, target_sr)

        return self._pyin_fallback(audio, target_sr)

    def export_raw_track(self, track: Dict[str, List[float]], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fp:
            json.dump(track, fp, indent=2)
