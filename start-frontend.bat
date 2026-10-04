@echo off
chcp 65001 >nul
cd /d "%~dp0frontend"

if not exist "package.json" (
  echo [ERROR] frontend\package.json not found.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo [INFO] node_modules not found, running npm install first...
  call npm install
  if errorlevel 1 (
    echo [ERROR] npm install failed.
    pause
    exit /b 1
  )
)

echo ============================================================
echo  Frontend : http://localhost:5173
echo  /api is proxied to backend at http://127.0.0.1:8000
echo  Make sure the backend is running first.
echo  Press Ctrl+C to stop.
echo ============================================================
echo.
call npm run dev
pause
