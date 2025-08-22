@echo off
REM tools/start_api.bat
setlocal
cd /d %~dp0..
echo Starting API on http://127.0.0.1:8000
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
endlocal
