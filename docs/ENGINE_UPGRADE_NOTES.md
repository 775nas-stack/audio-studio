# Engine Upgrade Notes

## Pipeline Overview

1. Audio is normalized to 16 kHz mono.
2. Three pitch engines run offline:
   - CREPE full-capacity (5 ms hop) for the highest resolution contour.
   - TorchCREPE (1024/160 hop) for a second neural opinion.
   - PYIN as the traditional fallback.
3. `backend/core/pitch_pipeline.py` resamples all contours onto a 5 ms grid, evaluates
   their confidence and stability, and routes each frame according to engine priority
   (CREPE → TorchCREPE → PYIN).
4. Advanced smoothing (`backend/core/smoothing_advanced.py`) applies median filtering,
   Savitzky–Golay smoothing, octave correction, outlier rejection, and confidence-aware blending.
5. The upgraded MIDI engine (`backend/core/midi_advanced.py`) performs dynamic thresholding,
   a lightweight state machine for onsets/sustains, octave heuristics, velocity
   estimation from per-frame RMS, and optional pitch bend generation.

## Routing Logic

- Each engine is scored by median confidence and pitch stability.
- Only engines meeting the minimum threshold are considered primary; others are used
  to fill gaps where higher-priority engines report low confidence.
- Frames inherit their origin engine, enabling downstream debugging and velocity heuristics.

## Accuracy Improvements

- Smaller CREPE hop size plus TorchCREPE cross-check significantly reduce octave and
  transition errors.
- Smoothing preserves expressive bends while eliminating isolated spikes.
- Segmentation enforces minimum duration and adjusts octaves relative to recent notes.
- Velocity now follows RMS energy, and pitch bends capture continuous slides when
  the contour deviates from the quantized pitch.

## Debugging & Diagnostics

- Enable or disable debug output via the `AUDIO_STUDIO_DEBUG` environment variable
  (defaults to `True`).
- Each project stores diagnostics under `data/projects/<id>/debug/`:
  - `raw_crepe.json`, `raw_torchcrepe.json`: raw neural contours.
  - `confidence.json`: routed confidence curve with source engine tags.
  - `smoothed_curve.json`: post-smoothing contour.
  - `segmentation.json` and `final_notes.json`: MIDI segmentation results.
- Missing models raise `Model missing: <engine>` errors with installation hints.

## Example Outputs

1. Upload → `/extract_midi` creates the contour, smoothing curve, and debug JSONs.
2. `/make_midi` exports `melody.mid` plus segmentation artifacts for inspection.
