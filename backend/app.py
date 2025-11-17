"""FastAPI backend for the offline Audio Studio phase 1 prototype."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core import audio_utils, crepe_runner, midi_utils, smooth_pitch


BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "data"
FRONTEND_DIR = REPO_ROOT / "frontend"
PROJECTS_DIR = DATA_DIR / "projects"
MODEL_PATH = REPO_ROOT / "models" / "melody" / "model.h5"

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


class ProjectRequest(BaseModel):
    project_id: str


class ChatRequest(BaseModel):
    message: str


def _project_path(project_id: str) -> Path:
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return project_dir


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


app = FastAPI(title="Audio Studio Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept an audio file, normalize it and create a new project."""

    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".wav", ".mp3"}:
        raise HTTPException(status_code=400, detail="Only mp3 and wav files are supported")

    project_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    audio, sr = audio_utils.load_audio_bytes(data, target_sr=audio_utils.TARGET_SAMPLE_RATE)
    output_path = project_dir / "uploaded.wav"
    audio_utils.save_wav(output_path, audio, sr)

    meta = {
        "project_id": project_id,
        "created_at": datetime.utcnow().isoformat(),
        "source_name": file.filename,
        "sample_rate": sr,
    }
    _write_json(project_dir / "meta.json", meta)

    return {"project_id": project_id, "message": "Audio uploaded", "meta": meta}


@app.post("/extract_midi")
async def extract_midi(request: ProjectRequest) -> Dict[str, Any]:
    """Run CREPE on the uploaded audio and smooth the detected melody."""

    project_dir = _project_path(request.project_id)
    audio_path = project_dir / "uploaded.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=400, detail="Upload audio before extracting MIDI")

    runner = crepe_runner.CREPERunner(model_path=MODEL_PATH)
    raw_track = runner.process_audio(str(audio_path))
    _write_json(project_dir / "melody_raw.json", raw_track)

    smooth_track = smooth_pitch.smooth_pitch_track(raw_track)
    _write_json(project_dir / "melody_smooth.json", smooth_track)

    return {
        "project_id": request.project_id,
        "frames": len(smooth_track["time"]),
        "message": "Melody extracted",
    }


@app.post("/make_midi")
async def make_midi(request: ProjectRequest) -> Dict[str, Any]:
    """Convert the smoothed melody to a MIDI file."""

    project_dir = _project_path(request.project_id)
    smooth_path = project_dir / "melody_smooth.json"
    if not smooth_path.exists():
        raise HTTPException(status_code=400, detail="Run extract_midi before creating MIDI")

    with smooth_path.open("r", encoding="utf-8") as fp:
        track = json.load(fp)

    midi_path = project_dir / "melody.mid"
    midi_utils.melody_to_midi(track, midi_path)

    return {
        "project_id": request.project_id,
        "midi_path": f"projects/{request.project_id}/melody.mid",
        "message": "MIDI file created",
    }


@app.post("/chat")
async def mini_chat(request: ChatRequest) -> Dict[str, Any]:
    """Simple rule-based chat endpoint used for placeholder UI."""

    text = (request.message or "").strip().lower()
    if not text:
        reply = "Please type a message about your project."
    elif "midi" in text:
        reply = "To build a MIDI file, upload audio, extract melody, then hit Convert."
    elif "hello" in text or "hi" in text:
        reply = "Hello! I'm your offline studio helper."
    elif "thanks" in text:
        reply = "You're welcome! Let me know if you need another conversion."
    else:
        reply = "Phase 1 chat is simple – try asking about MIDI or upload steps."

    return {"response": reply}


app.mount("/projects", StaticFiles(directory=PROJECTS_DIR), name="projects")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
