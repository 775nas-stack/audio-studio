"""Utilities for converting smoothed melody tracks into MIDI files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pretty_midi

MIN_ALLOWED_FREQ = 65.0
MAX_ALLOWED_FREQ = 1000.0
HUMMING_MAX_ALLOWED_FREQ = 1500.0
MAX_INTERP_GAP_SECONDS = 0.1
HUMMING_MAX_INTERP_GAP_SECONDS = 0.05
MIN_NOTE_DURATION_SECONDS = 0.08
HUMMING_MIN_NOTE_DURATION_SECONDS = 0.03
MERGE_GAP_SECONDS = 0.1
HUMMING_MERGE_GAP_SECONDS = 0.05
MEDIAN_WINDOW = 5
MEAN_WINDOW = 5
JUMP_LIMIT_CENTS = 300.0


def _hz_to_midi(freq: float) -> int:
    return int(round(pretty_midi.hz_to_note_number(max(freq, 1e-6))))


def _quantize_time(value: float, quantum: float) -> float:
    if quantum <= 0:
        return value
    return round(value / quantum) * quantum


def _estimate_frame_duration(time: np.ndarray) -> float:
    if time.size < 2:
        return 0.0
    diffs = np.diff(time)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _median_filter(values: np.ndarray, window: int = MEDIAN_WINDOW) -> np.ndarray:
    if window <= 1 or values.size < 2:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filtered = np.empty_like(values)
    for idx in range(values.size):
        filtered[idx] = np.median(padded[idx : idx + window])
    return filtered


def _moving_average(values: np.ndarray, window: int = MEAN_WINDOW) -> np.ndarray:
    if window <= 1 or values.size < 2:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(values, kernel, mode="same")
    return smoothed.astype(values.dtype)


def _limit_jumps(values: np.ndarray, cents_threshold: float = JUMP_LIMIT_CENTS) -> np.ndarray:
    if values.size == 0:
        return values
    stabilized = values.copy()
    last = stabilized[0]
    for idx in range(1, values.size):
        current = stabilized[idx]
        if last <= 0 or current <= 0:
            last = current
            continue
        cents = 1200.0 * np.log2(current / last)
        if abs(cents) > cents_threshold:
            stabilized[idx] = last
        else:
            last = current
    return stabilized


def _interpolate_small_gaps(
    time: np.ndarray,
    freq: np.ndarray,
    mask: np.ndarray,
    max_gap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if time.size == 0:
        return freq, mask

    filled = freq.copy()
    valid = mask.copy()
    frame_duration = _estimate_frame_duration(time)
    if frame_duration <= 0:
        frame_duration = max_gap

    idx = 0
    total = len(freq)
    while idx < total:
        if valid[idx]:
            idx += 1
            continue
        start = idx
        while idx < total and not valid[idx]:
            idx += 1
        end = idx

        gap_frames = end - start
        prev_idx = start - 1
        next_idx = end
        approx_duration = gap_frames * frame_duration

        if (
            approx_duration <= max_gap
            and prev_idx >= 0
            and next_idx < total
            and valid[prev_idx]
            and valid[next_idx]
        ):
            start_time = time[prev_idx]
            end_time = time[next_idx]
            if end_time > start_time:
                filled[start:end] = np.interp(
                    time[start:end],
                    [start_time, end_time],
                    [filled[prev_idx], filled[next_idx]],
                )
                valid[start:end] = True

    return filled, valid


def _clean_pitch_track(
    track: Dict[str, Sequence[float]],
    *,
    humming_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float], int]:
    raw_time = np.asarray(track.get("time", []), dtype=float)
    raw_freq = np.asarray(track.get("frequency", []), dtype=float)
    if raw_time.size == 0 or raw_freq.size == 0:
        valid = int(np.count_nonzero(np.isfinite(raw_freq) & (raw_freq > 0)))
        bounds = (
            float(raw_time[0]) if raw_time.size else 0.0,
            float(raw_time[-1]) if raw_time.size else 0.0,
        )
        return np.array([], dtype=float), np.array([], dtype=float), 0.0, bounds, valid

    length = min(raw_time.size, raw_freq.size)
    time = raw_time[:length]
    freq = np.nan_to_num(raw_freq[:length], nan=0.0, posinf=0.0, neginf=0.0)

    order = np.argsort(time)
    time = time[order]
    freq = freq[order]

    initial_mask = np.isfinite(freq) & (freq > 0)
    initial_valid = int(np.count_nonzero(initial_mask))

    mask = initial_mask.copy()
    voiced_flag = track.get("voiced_flag")
    if voiced_flag is not None:
        voiced_arr = np.asarray(voiced_flag, dtype=bool)
        if voiced_arr.size >= length:
            voiced_arr = voiced_arr[:length]
        else:
            voiced_arr = np.pad(voiced_arr, (0, length - voiced_arr.size), constant_values=False)
        voiced_arr = voiced_arr[order]
        candidate = mask & voiced_arr
        if np.any(candidate):
            mask = candidate

    max_freq = HUMMING_MAX_ALLOWED_FREQ if humming_mode else MAX_ALLOWED_FREQ
    freq = np.clip(freq, a_min=0.0, a_max=max_freq)

    max_gap = HUMMING_MAX_INTERP_GAP_SECONDS if humming_mode else MAX_INTERP_GAP_SECONDS
    freq, mask = _interpolate_small_gaps(time, freq, mask, max_gap)

    freq = _median_filter(freq)
    freq = _moving_average(freq)
    freq = _limit_jumps(freq)

    min_freq = MIN_ALLOWED_FREQ
    mask &= freq >= min_freq
    mask &= freq <= max_freq

    if not np.any(mask) and np.any(freq > 0):
        mask = freq > 0

    frame_duration = _estimate_frame_duration(time)
    if frame_duration <= 0:
        frame_duration = 0.01

    bounds = (float(time[0]), float(time[-1])) if time.size else (0.0, 0.0)
    cleaned_time = time[mask]
    cleaned_freq = freq[mask]

    if cleaned_time.size == 0 and initial_valid > 0:
        fallback_freq = float(np.median(freq[initial_mask]))
        fallback_freq = float(np.clip(fallback_freq, MIN_ALLOWED_FREQ, max_freq))
        start = bounds[0]
        end = bounds[1] if bounds[1] > start else start + max(frame_duration, 0.5)
        cleaned_time = np.array([start, end], dtype=float)
        cleaned_freq = np.array([fallback_freq, fallback_freq], dtype=float)

    return cleaned_time, cleaned_freq, frame_duration, bounds, initial_valid


def _merge_midi_frames(
    time: np.ndarray,
    midi_pitch: np.ndarray,
    frame_duration: float,
    merge_gap: float,
) -> List[Tuple[float, float, int]]:
    if midi_pitch.size == 0:
        return []

    step = max(frame_duration, 1e-3)
    segments: List[Tuple[float, float, int]] = []
    segment_start = float(time[0])
    segment_end = segment_start + step
    current_pitch = int(midi_pitch[0])
    prev_time = float(time[0])

    for idx in range(1, midi_pitch.size):
        t = float(time[idx])
        same_pitch = abs(int(midi_pitch[idx]) - current_pitch) <= 1
        contiguous = (t - prev_time) <= (frame_duration * 1.5 if frame_duration > 0 else merge_gap)
        if contiguous and same_pitch:
            segment_end = t + step
        else:
            segments.append((segment_start, segment_end, current_pitch))
            segment_start = t
            segment_end = t + step
            current_pitch = int(midi_pitch[idx])
        prev_time = t

    segments.append((segment_start, segment_end, current_pitch))

    merged: List[Tuple[float, float, int]] = []
    for start, end, pitch in segments:
        if not merged:
            merged.append((start, end, pitch))
            continue
        prev_start, prev_end, prev_pitch = merged[-1]
        gap = start - prev_end
        if gap <= merge_gap and abs(prev_pitch - pitch) <= 1:
            merged[-1] = (prev_start, max(prev_end, end), prev_pitch)
        else:
            merged.append((start, end, pitch))
    return merged


def _filter_short_segments(
    segments: Iterable[Tuple[float, float, int]], min_duration: float
) -> List[Tuple[float, float, int]]:
    filtered: List[Tuple[float, float, int]] = []
    for start, end, pitch in segments:
        duration = end - start
        if duration < min_duration or duration <= 0:
            continue
        filtered.append((start, end, pitch))
    return filtered


def build_midi_from_track(
    track: Dict[str, List[float]],
    output_path: Path,
    tempo: float = 120.0,
    *,
    humming_mode: bool = False,
) -> Tuple[Path, int]:
    """Convert a smoothed melody track into a quantized PrettyMIDI file."""

    raw_freq = np.asarray(track.get("frequency", []), dtype=float)
    raw_valid_freq = raw_freq[np.isfinite(raw_freq) & (raw_freq > 0)]
    total_frames = len(track.get("time", []))
    print(
        f"[midi_builder] Cleaning track with humming_mode={humming_mode} and {total_frames} frames"
    )
    time, freq, frame_duration, bounds, initial_valid = _clean_pitch_track(
        track, humming_mode=humming_mode
    )
    if time.size == 0 or freq.size == 0:
        if initial_valid == 0:
            raise ValueError("No melody, audio is effectively silent")
        print("[midi_builder] No cleaned frames but raw pitch existed – synthesizing fallback")
        start = bounds[0]
        end = bounds[1] if bounds[1] > start else start + 0.5
        fallback_freq = (
            float(
                np.clip(
                    np.median(raw_valid_freq),
                    MIN_ALLOWED_FREQ,
                    HUMMING_MAX_ALLOWED_FREQ if humming_mode else MAX_ALLOWED_FREQ,
                )
            )
            if raw_valid_freq.size
            else MIN_ALLOWED_FREQ
        )
        freq = np.array([fallback_freq, fallback_freq], dtype=float)
        time = np.array([start, end], dtype=float)

    midi_pitch = np.array([_hz_to_midi(hz) for hz in freq], dtype=int)
    merge_gap = HUMMING_MERGE_GAP_SECONDS if humming_mode else MERGE_GAP_SECONDS
    segments = _merge_midi_frames(time, midi_pitch, frame_duration, merge_gap)
    min_duration = (
        HUMMING_MIN_NOTE_DURATION_SECONDS if humming_mode else MIN_NOTE_DURATION_SECONDS
    )
    filtered_segments = _filter_short_segments(segments, min_duration)
    if not filtered_segments and segments:
        filtered_segments = segments

    if not filtered_segments:
        start = bounds[0]
        end = bounds[1] + max(frame_duration, min_duration, 0.2)
        if end <= start:
            end = start + max(frame_duration, min_duration, 0.2)
        fallback_pitch = int(np.clip(int(round(np.median(midi_pitch))), 0, 127))
        filtered_segments = [(start, end, fallback_pitch)]
        print("[midi_builder] Synthesized fallback note from noisy frames")

    sixteenth = 60.0 / tempo / 4.0

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0, name="Melody")

    for start, end, midi_note in filtered_segments:
        start_q = _quantize_time(start, sixteenth)
        end_q = _quantize_time(end, sixteenth)
        if end_q <= start_q:
            end_q = start_q + max(sixteenth, 1e-3)
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(midi_note),
            start=start_q,
            end=end_q,
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    if instrument.notes:
        coverage = f"{instrument.notes[0].start:.2f}s→{instrument.notes[-1].end:.2f}s"
    else:
        coverage = "0.00s"
    note_count = len(instrument.notes)
    print(f"[midi_builder] Wrote {note_count} notes covering {coverage}")
    return output_path, note_count


def melody_to_midi(
    track: Dict[str, List[float]],
    output_path: Path,
    tempo: float = 120.0,
    *,
    humming_mode: bool = False,
) -> Tuple[Path, int]:
    """Backward-compatible wrapper for callers relying on the old name."""

    return build_midi_from_track(
        track,
        output_path,
        tempo=tempo,
        humming_mode=humming_mode,
    )
