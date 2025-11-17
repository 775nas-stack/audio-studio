from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core.audio import load_audio
from .core.extractor import extract_pitch
from .core.midi_builder import build_midi
from .core.smoothing import smooth_track
from .core.types import NoMelodyError, PitchTrack

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


@app.exception_handler(NoMelodyError)
async def melody_error_handler(_, exc: NoMelodyError):  # pragma: no cover - FastAPI integration
    return JSONResponse(status_code=400, content={"error": str(exc)})


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


@app.post("/extract_midi")
async def extract_midi(request: ProjectRequest):
    project_path = PROJECTS_DIR / request.project_id
    audio_path = project_path / "input.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail={"error": "Audio file not found."})

    audio, sr = load_audio(audio_path)
    track = extract_pitch(audio, sr)
    smoothed = smooth_track(track)

    contour = {
        "time": smoothed.time.tolist(),
        "frequency": smoothed.frequency.tolist(),
        "confidence": smoothed.confidence.tolist(),
        "engine": smoothed.engine,
    }
    contour_path = project_path / "contour.json"
    contour_path.write_text(json.dumps(contour))

    return {
        "project_id": request.project_id,
        "engine": smoothed.engine,
        "frames": smoothed.finite_count(),
    }


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
    )


@app.post("/make_midi")
async def make_midi(request: ProjectRequest):
    track = _load_contour(request.project_id)
    midi_path = (PROJECTS_DIR / request.project_id) / "melody.mid"
    build_midi(track, midi_path)
    return {
        "project_id": request.project_id,
        "midi_url": f"/projects/{request.project_id}/melody.mid",
    }
