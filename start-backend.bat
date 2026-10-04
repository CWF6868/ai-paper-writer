@echo off
chcp 65001 >nul
cd /d "%~dp0backend"

if not exist "venv\Scripts\uvicorn.exe" (
  echo [ERROR] backend venv not found: %~dp0backend\venv
  echo         Create it first:
  echo             python -m venv venv
  echo             venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist ".env" (
  echo [ERROR] backend\.env not found. Copy .env.example to .env and fill in the API keys.
  pause
  exit /b 1
)

echo ============================================================
echo  Backend  : http://127.0.0.1:8000
echo  API docs : http://127.0.0.1:8000/docs
echo  Press Ctrl+C to stop.
echo ============================================================
echo.
venv\Scripts\uvicorn.exe app.main:app --reload --port 8000
pause
