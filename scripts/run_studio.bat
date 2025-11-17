@echo off
setlocal
cd /d %~dp0..
uvicorn backend.app:app --host 0.0.0.0 --port 7860
