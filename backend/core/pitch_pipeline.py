from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np

from .crepe_engine import run_crepe
from .pyin_engine import run_pyin
from .torchcrepe_engine import run_torchcrepe
from .debug import write_debug_file
from .types import ModelMissingError, NoMelodyError, PitchTrack

LOGGER = logging.getLogger(__name__)

TARGET_STEP = 0.005  # 5 ms grid
MIN_FRAMES = 32
ENGINE_PRIORITY = ("crepe", "torchcrepe", "pyin")
ENGINE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "crepe": {"confidence": 0.35, "max_cents": 250.0},
    "torchcrepe": {"confidence": 0.3, "max_cents": 280.0},
    "pyin": {"confidence": 0.2, "max_cents": 320.0},
}


@dataclass
class EngineTrack:
    name: str
    track: PitchTrack
    reliable: bool
    diagnostics: Dict[str, float]


def _hz_to_cents(freq: np.ndarray) -> np.ndarray:
    cents = np.full_like(freq, np.nan, dtype=float)
    mask = freq > 0
    cents[mask] = 1200.0 * np.log2(freq[mask] / 10.0)
    return cents


def _stability_mask(freq: np.ndarray, cents_limit: float) -> np.ndarray:
    mask = np.ones_like(freq, dtype=bool)
    finite_idx = np.where(np.isfinite(freq))[0]
    if finite_idx.size < 2:
        return mask
    cents = _hz_to_cents(freq)
    prev = finite_idx[0]
    for idx in finite_idx[1:]:
        jump = abs(cents[idx] - cents[prev])
        if jump > cents_limit:
            mask[idx] = False
        else:
            prev = idx
    return mask


def _median_confidence(track: PitchTrack) -> float:
    finite_mask = track.finite_mask()
    if not finite_mask.any():
        return 0.0
    return float(np.nanmedian(track.confidence[finite_mask]))


def _median_jump(track: PitchTrack) -> float:
    mask = track.finite_mask()
    freq = track.frequency[mask]
    if freq.size < 2:
        return 0.0
    cents = _hz_to_cents(freq)
    diffs = np.abs(np.diff(cents))
    return float(np.nanmedian(diffs))


def _score_track(name: str, track: PitchTrack) -> EngineTrack:
    conf = _median_confidence(track)
    jump = _median_jump(track)
    reliable = track.finite_count() >= MIN_FRAMES and conf >= ENGINE_THRESHOLDS[name]["confidence"]
    diagnostics = {
        "frames": float(track.finite_count()),
        "median_confidence": conf,
        "median_jump_cents": jump,
    }
    return EngineTrack(name=name, track=track, reliable=reliable, diagnostics=diagnostics)


def _resample(values_time: np.ndarray, values: np.ndarray, target_time: np.ndarray, fill: float = np.nan) -> np.ndarray:
    if values_time.size == 0:
        return np.full(target_time.shape, fill, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return np.full(target_time.shape, fill, dtype=float)
    ref_time = values_time[valid]
    ref_values = values[valid]
    interp = np.interp(target_time, ref_time, ref_values)
    outside = (target_time < ref_time[0]) | (target_time > ref_time[-1])
    interp[outside] = fill
    return interp


def _resample_frequency(track: PitchTrack, target_time: np.ndarray) -> np.ndarray:
    freq = track.frequency.copy()
    return _resample(track.time, freq, target_time)


def _resample_confidence(track: PitchTrack, target_time: np.ndarray) -> np.ndarray:
    conf = track.confidence.copy()
    if conf.size == 0:
        return np.zeros_like(target_time)
    valid = np.isfinite(conf)
    if not valid.any():
        conf = np.zeros_like(conf)
        valid = np.ones_like(conf, dtype=bool)
    ref_time = track.time[valid]
    ref_values = conf[valid]
    interp = np.interp(target_time, ref_time, ref_values)
    outside = (target_time < ref_time[0]) | (target_time > ref_time[-1])
    interp[outside] = 0.0
    return interp


def _interpolate_nan_runs(data: np.ndarray, max_gap: int = 3) -> np.ndarray:
    result = data.copy()
    isnan = ~np.isfinite(result)
    idx = 0
    n = len(result)
    while idx < n:
        if not isnan[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and isnan[idx]:
            idx += 1
        end = idx
        gap = end - start
        if gap == 0 or gap > max_gap:
            continue
        left = start - 1
        right = end
        if left < 0 or right >= n:
            continue
        if not (np.isfinite(result[left]) and np.isfinite(result[right])):
            continue
        span = gap + 1
        for offset, pos in enumerate(range(start, end), start=1):
            ratio = offset / span
            result[pos] = (1 - ratio) * result[left] + ratio * result[right]
    return result


def _compute_rms(audio: np.ndarray, sr: int, target_time: np.ndarray) -> np.ndarray:
    hop = max(1, int(round(sr * TARGET_STEP)))
    frame_length = max(2048, hop * 4)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop, center=True)[0]
    rms_time = np.arange(rms.shape[0]) * (hop / sr)
    interp = np.interp(target_time, rms_time, rms)
    if np.max(interp) > 0:
        interp = interp / np.max(interp)
    return interp


def _combine_tracks(audio: np.ndarray, sr: int, tracks: List[EngineTrack], debug_dir: Path | None) -> PitchTrack:
    if not tracks:
        raise NoMelodyError("No pitch sources available.")

    duration = len(audio) / float(sr)
    target_time = np.arange(0.0, duration + TARGET_STEP, TARGET_STEP)
    combined_freq = np.full(target_time.shape, np.nan, dtype=float)
    combined_conf = np.zeros(target_time.shape, dtype=float)
    source = np.full(target_time.shape, "unassigned", dtype=object)

    for engine in tracks:
        threshold = ENGINE_THRESHOLDS[engine.name]
        freq = _resample_frequency(engine.track, target_time)
        freq[freq <= 0] = np.nan
        conf = _resample_confidence(engine.track, target_time)
        stability = _stability_mask(freq, threshold["max_cents"])
        mask = stability & (conf >= threshold["confidence"]) & np.isfinite(freq)
        replace = mask & ((combined_conf < threshold["confidence"]) | np.isnan(combined_freq))
        combined_freq[replace] = freq[replace]
        combined_conf[replace] = conf[replace]
        source[replace] = engine.name

    combined_freq = _interpolate_nan_runs(combined_freq, max_gap=4)
    loudness = _compute_rms(audio, sr, target_time)

    primary_engine = tracks[0].name if tracks else "unknown"
    crepe_track = next((t.track for t in tracks if t.name == "crepe" and t.track.activation is not None), None)
    activation = crepe_track.activation if crepe_track is not None else None

    sources_array = np.array(source, dtype=object)
    routed = PitchTrack(
        time=target_time,
        frequency=combined_freq,
        confidence=combined_conf,
        engine=primary_engine,
        activation=activation,
        loudness=loudness,
        sources=sources_array,
    )

    if routed.finite_count() < MIN_FRAMES:
        raise NoMelodyError("No stable monophonic melody detected.")

    write_debug_file(debug_dir, "confidence.json", {
        "time": target_time.tolist(),
        "confidence": combined_conf.tolist(),
        "sources": sources_array.tolist(),
    })

    return routed


def extract_unified_pitch(audio: np.ndarray, sr: int, debug_dir: Path | None = None) -> PitchTrack:
    tracks: List[EngineTrack] = []
    errors: List[ModelMissingError] = []

    for name, runner in (("crepe", run_crepe), ("torchcrepe", run_torchcrepe)):
        try:
            track = runner(audio, sr)
        except ModelMissingError as exc:
            errors.append(exc)
            write_debug_file(
                debug_dir,
                "raw_crepe.json" if name == "crepe" else "raw_torchcrepe.json",
                {
                    "time": [],
                    "frequency": [],
                    "confidence": [],
                    "engine": name,
                    "error": str(exc),
                },
            )
            continue
        tracks.append(_score_track(name, track))
        filename = "raw_crepe.json" if name == "crepe" else "raw_torchcrepe.json"
        write_debug_file(debug_dir, filename, track.to_payload(include_activation=(name == "crepe")))

    pyin_track = run_pyin(audio, sr)
    tracks.append(_score_track("pyin", pyin_track))

    reliable_tracks = [t for t in tracks if t.reliable]
    ordered: List[EngineTrack] = []
    for name in ENGINE_PRIORITY:
        match = next((t for t in reliable_tracks if t.name == name), None)
        if match:
            ordered.append(match)
    if not ordered:
        # Fall back to the best scoring track even if unreliable.
        tracks_sorted = sorted(
            tracks,
            key=lambda t: (
                -t.diagnostics["median_confidence"],
                t.diagnostics["median_jump_cents"],
            ),
        )
        ordered = [tracks_sorted[0]] if tracks_sorted else []

    if not ordered:
        if errors:
            raise errors[0]
        raise NoMelodyError("No stable monophonic melody detected.")

    LOGGER.info(
        "Routing pitch using %s (frames=%d, conf=%.3f)",
        ordered[0].name,
        ordered[0].diagnostics["frames"],
        ordered[0].diagnostics["median_confidence"],
    )

    return _combine_tracks(audio, sr, ordered, debug_dir)
