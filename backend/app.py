from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core.debug import ensure_debug_dir, write_debug_file
from .core.melody_postprocess import MelodyNote
from .core.midi_builder import build_midi
from .core.pitch_pipeline import ENGINE_CHOICES, extract_pitch_pipeline
from .core.smoothing import smooth_track
from .core.types import ModelMissingError, NoMelodyError, PitchTrack

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
FRONTEND_DIR = BASE_DIR / "frontend"

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Offline Audio Studio")
app.mount("/projects", StaticFiles(directory=PROJECTS_DIR), name="projects")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ProjectRequest(BaseModel):
    project_id: str


class ExtractRequest(ProjectRequest):
    engine: str | None = None


@app.exception_handler(NoMelodyError)
async def melody_error_handler(_, exc: NoMelodyError):  # pragma: no cover - FastAPI integration
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(ModelMissingError)
async def model_missing_handler(_, exc: ModelMissingError):  # pragma: no cover - FastAPI integration
    return JSONResponse(status_code=500, content={"error": str(exc), "missing_model": exc.model_name})


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):  # pragma: no cover - FastAPI integration
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("error") or detail.get("detail") or str(detail)
    else:
        message = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"error": message})


@app.get("/")
def index():  # pragma: no cover - FastAPI integration
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    project_id = uuid.uuid4().hex
    project_path = PROJECTS_DIR / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    target = project_path / "input.wav"
    contents = await file.read()
    target.write_bytes(contents)
    LOGGER.info("Uploaded audio for project %s", project_id)
    return {"project_id": project_id, "message": "Audio uploaded."}


def _run_extraction(request: ExtractRequest) -> dict:
    project_path = PROJECTS_DIR / request.project_id
    audio_path = project_path / "input.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail={"error": "Audio file not found."})

    engine_choice = request.engine
    if engine_choice in (None, "", "auto"):
        engine_choice = None
    elif engine_choice not in ENGINE_CHOICES:
        raise HTTPException(status_code=400, detail={"error": f"Unknown engine '{engine_choice}'."})

    debug_dir = project_path / "debug"
    ensure_debug_dir(debug_dir)

    try:
        result = extract_pitch_pipeline(
            audio_path=audio_path,
            engine=engine_choice,
            debug_dir=debug_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    track = result.track
    smoothed = smooth_track(track)

    write_debug_file(debug_dir, "smoothed_curve.json", smoothed.to_payload())

    contour = {
        "time": smoothed.time.tolist(),
        "frequency": smoothed.frequency.tolist(),
        "confidence": smoothed.confidence.tolist(),
        "engine": smoothed.engine,
        "loudness": smoothed.loudness.tolist() if smoothed.loudness is not None else None,
        "sources": smoothed.sources.tolist() if smoothed.sources is not None else None,
    }
    contour_path = project_path / "contour.json"
    contour_path.write_text(json.dumps(contour))

    notes_payload = [note.to_payload() for note in result.notes]
    notes_path = project_path / "melody_notes.json"
    notes_path.write_text(json.dumps(notes_payload))
    write_debug_file(debug_dir, "melody_notes.json", notes_payload)

    return {
        "project_id": request.project_id,
        "engine": smoothed.engine,
        "frames": smoothed.finite_count(),
        "notes": len(result.notes),
    }


@app.post("/extract_melody")
async def extract_melody(request: ExtractRequest):  # pragma: no cover - FastAPI integration
    return _run_extraction(request)


@app.post("/extract_midi")
async def extract_midi(request: ExtractRequest):  # pragma: no cover - FastAPI integration
    return _run_extraction(request)


def _load_contour(project_id: str) -> PitchTrack:
    project_path = PROJECTS_DIR / project_id
    contour_path = project_path / "contour.json"
    if not contour_path.exists():
        raise HTTPException(status_code=404, detail={"error": "No contour found. Run extraction first."})

    payload = json.loads(contour_path.read_text())
    return PitchTrack(
        time=np.array(payload["time"], dtype=float),
        frequency=np.array(payload["frequency"], dtype=float),
        confidence=np.array(payload["confidence"], dtype=float),
        engine=payload.get("engine", "crepe"),
        loudness=(np.array(payload["loudness"], dtype=float) if payload.get("loudness") is not None else None),
        sources=(np.array(payload["sources"], dtype=object) if payload.get("sources") is not None else None),
    )


def _load_notes(project_id: str) -> list[MelodyNote] | None:
    project_path = PROJECTS_DIR / project_id
    notes_path = project_path / "melody_notes.json"
    if not notes_path.exists():
        return None
    payload = json.loads(notes_path.read_text())
    return [MelodyNote.from_payload(item) for item in payload]


@app.post("/make_midi")
async def make_midi(request: ProjectRequest):
    track = _load_contour(request.project_id)
    notes = _load_notes(request.project_id)
    midi_path = (PROJECTS_DIR / request.project_id) / "melody.mid"
    debug_dir = (PROJECTS_DIR / request.project_id) / "debug"
    ensure_debug_dir(debug_dir)
    build_midi(track, midi_path, debug_dir=debug_dir, notes=notes)
    return {
        "project_id": request.project_id,
        "midi_url": f"/projects/{request.project_id}/melody.mid",
    }


@app.post("/cleanup_project")
async def cleanup_project(request: ProjectRequest):
    project_path = PROJECTS_DIR / request.project_id
    if project_path.exists():
        shutil.rmtree(project_path)
    return {"project_id": request.project_id, "status": "removed"}
