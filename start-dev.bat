@echo off
echo ========================================
echo  ISL Avatar - Development Startup
echo ========================================
echo.

REM Prefer the project virtualenv so installs do not leak into the system Python.
set "PY=backend\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/3] Starting Backend (%PY%)...
cd backend
start "ISL Backend" cmd /c "set PYTHONUNBUFFERED=1 && "%PY%" -m pip install -r requirements.txt && "%PY%" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --access-log"
cd ..

echo [2/3] Starting Frontend...
cd frontend
start "ISL Frontend" cmd /c "npm run dev"
cd ..

echo [3/3] Done!
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Open http://localhost:5173 and watch the "ISL Backend" window for request logs.
pause