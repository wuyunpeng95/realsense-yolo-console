@echo off
cd /d "%~dp0"
echo DepthSight http://127.0.0.1:8000
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else (
  python app.py
)
pause
