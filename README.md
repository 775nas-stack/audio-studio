# Offline AI Audio Studio

This repository contains the initial skeleton for the Offline AI Audio Studio project. Implementation details will be added in future phases.

## Local startup

1. Install dependencies with `pip install -r requirements.txt`.
2. From the repository root, run the backend and bundled UI on the Techloq-friendly port:

   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 7860
   ```

3. Open [http://localhost:7860](http://localhost:7860) in your browser to access the UI that is served directly from FastAPI.

### File locations

- Uploaded audio and generated MIDI are stored under `data/projects/<timestamp>`.
- The CREPE model is loaded from the manual weight file stored at `backend/vendor/crepe/model-full.h5`.
- Generated MIDI files are exposed at `/projects/<project-id>/melody.mid` so they can be downloaded straight from the UI.

## Installing Model Files (Manual)

Both neural pitch engines require manually downloaded weights. GitHub rejects these large binaries, so place them locally before
starting the backend.

1. **CREPE-full**
   - Download `model-full.h5` from [the official repository](https://github.com/marl/crepe/raw/master/assets/model-full.h5).
   - Copy the file to `backend/vendor/crepe/model-full.h5` (you can also mirror it to `models/crepe/model-full.h5`).
   - Re-start the backend. If the file is missing you will receive `Model missing: crepe` from the API.
2. **TorchCREPE**
   - Download `full.pth` from [the TorchCREPE assets](https://github.com/maxrmorrison/torchcrepe/raw/main/torchcrepe/assets/full.pth).
   - Place the file at `backend/vendor/torchcrepe/full.pth` (optionally mirror to `models/torchcrepe/full.pth`).
   - The backend never downloads weights automatically, so the API will return `Model missing: torchcrepe` until the file exists.
