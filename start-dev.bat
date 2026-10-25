@echo off
setlocal EnableExtensions EnableDelayedExpansion
title ISL Avatar - Demo Launcher

cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo ========================================
echo  ISL Avatar - Hackathon Demo Launcher
echo ========================================
echo.

REM ---- prerequisites -------------------------------------------------------
if not exist "%ROOT%\backend\.venv\Scripts\python.exe" (
  echo [error] backend\.venv not found. Create it once with:
  echo.
  echo     python -m venv backend\.venv
  echo     backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
  echo.
  goto :end
)

REM Node is often installed per-user or via a version manager even when it
REM is not on the machine PATH.
if exist "%ProgramFiles%\nodejs\npm.cmd" set "PATH=%ProgramFiles%\nodejs;%PATH%"
if exist "%ProgramFiles(x86)%\nodejs\npm.cmd" set "PATH=%ProgramFiles(x86)%\nodejs;%PATH%"
if exist "%LOCALAPPDATA%\Programs\nodejs\npm.cmd" set "PATH=%LOCALAPPDATA%\Programs\nodejs;%PATH%"
if exist "%LOCALAPPDATA%\Volta\bin\npm.cmd" set "PATH=%LOCALAPPDATA%\Volta\bin;%PATH%"
if exist "%APPDATA%\nvm" for /d %%D in ("%APPDATA%\nvm\v*") do if exist "%%D\npm.cmd" set "PATH=%%D;%PATH%"
where npm >nul 2>&1
if errorlevel 1 (
  echo [error] Node.js/npm was not found.
  echo         Install Node.js 22 or newer from https://nodejs.org and re-run.
  echo.
  goto :end
)

if not exist "%ROOT%\frontend\node_modules" (
  echo [setup] Installing frontend dependencies once...
  pushd "%ROOT%\frontend"
  call npm install
  popd
)

REM ---- pick a port ---------------------------------------------------------
REM 8000 is the documented default; fall back to 8010 when another service
REM (this machine has one) already owns it. Vite's proxy follows the URL.
netstat -ano | findstr ":5173" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo [error] Port 5173 is already in use. Close the other frontend window and re-run.
  echo.
  goto :end
)

set "PORT=8000"
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
  set "PORT=8010"
  netstat -ano | findstr ":8010" | findstr "LISTENING" >nul 2>&1
  if not errorlevel 1 (
    echo [error] Ports 8000 and 8010 are both in use. Stop one of them and re-run.
    echo.
    goto :end
  )
  echo [port] 8000 is busy - using 8010 instead.
)

REM ---- backend -------------------------------------------------------------
REM Set the backend env here, not inline after `&&`: `set VAR=0 && ...`
REM stores "0 " (trailing space), and main.py compares against "0" exactly,
REM so the prefetch would run and block startup for minutes.
set "PYTHONUNBUFFERED=1"
set "ISL_WARMUP_MODELS=0"
echo [1/3] Backend  http://127.0.0.1:!PORT!
pushd "%ROOT%\backend"
start "ISL Backend" cmd /k ""%ROOT%\backend\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port !PORT!"
popd

set /a TRIES=0
:waitloop
set /a TRIES+=1
curl -s --max-time 10 -o nul "http://127.0.0.1:!PORT!/api/health" >nul 2>&1
if not errorlevel 1 goto :warm
if !TRIES! GEQ 45 (
  echo [warn] Backend did not answer after 90 s - starting the UI anyway.
  goto :frontend
)
ping -n 3 127.0.0.1 >nul
goto :waitloop

:warm
REM Load what the demo touches before the browser opens: a curated sentence
REM (RAG), a novel sentence (T5) and a Hindi greeting (OPUS-MT). First run
REM downloads/loads them; later runs are seconds.
echo [2/3] Warming models (first run can take ~40 s; it may download weights)...
curl -s --max-time 120 -o nul -X POST -H "Content-Type: application/json" -d "{\"text\":\"good morning\",\"use_rag\":true}" "http://127.0.0.1:!PORT!/api/translate" >nul 2>&1
curl -s --max-time 120 -o nul -X POST -H "Content-Type: application/json" -d "{\"text\":\"i need a doctor\",\"use_rag\":true}" "http://127.0.0.1:!PORT!/api/translate" >nul 2>&1
curl -s --max-time 120 -o nul -X POST -H "Content-Type: application/json" -d "{\"text\":\"\u0938\u0941\u092a\u094d\u0930\u092d\u093e\u0924\",\"use_rag\":true}" "http://127.0.0.1:!PORT!/api/translate" >nul 2>&1

:frontend
echo [3/3] Frontend http://localhost:5173
set "ISL_BACKEND_URL=http://127.0.0.1:!PORT!"
pushd "%ROOT%\frontend"
start "ISL Frontend" cmd /k "npm run dev"
popd
ping -n 7 127.0.0.1 >nul
start "" "http://localhost:5173"

echo.
echo Ready. One-click demo sentences are under the text box:
echo   hi how are you  /  help me  /  do not worry  /  are you free today
echo   can you repeat that please  /  congratulations  /  Hindi greeting
echo.
echo API docs: http://127.0.0.1:!PORT!/docs
echo Close the "ISL Backend" and "ISL Frontend" windows to stop.
echo.

:end
if not defined ISL_NO_PAUSE pause
