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
- The CREPE model is loaded from `models/melody/model.h5` (download the model into that path before running the backend).
- Generated MIDI files are exposed at `/projects/<project-id>/melody.mid` so they can be downloaded straight from the UI.
