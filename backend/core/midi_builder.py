from __future__ import annotations

from pathlib import Path

from .debug import write_debug_file
from .midi_advanced import MidiExportResult, NoteEvent, build_midi_advanced
from .types import PitchTrack


def build_midi(track: PitchTrack, destination: Path, debug_dir: Path | None = None) -> MidiExportResult:
    result = build_midi_advanced(track, destination)

    notes_payload = [
        {"pitch": note.pitch, "start": note.start, "end": note.end, "velocity": note.velocity, "source": note.source}
        for note in result.notes
    ]
    write_debug_file(debug_dir, "segmentation.json", result.segmentation)
    write_debug_file(debug_dir, "final_notes.json", notes_payload)

    return result


__all__ = ["build_midi", "MidiExportResult", "NoteEvent"]
