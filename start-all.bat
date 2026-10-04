@echo off
chcp 65001 >nul
echo Starting backend and frontend in two separate windows...
start "ai-paper-writer backend"  cmd /k "%~dp0start-backend.bat"
timeout /t 3 /nobreak >nul
start "ai-paper-writer frontend" cmd /k "%~dp0start-frontend.bat"
echo.
echo Backend  window: http://127.0.0.1:8000/docs
echo Frontend window: http://localhost:5173
echo Close each window (or Ctrl+C inside it) to stop the service.
