@echo off
echo ========================================
echo  ISL Avatar - Development Startup
echo ========================================
echo.

echo [1/3] Starting Backend...
cd backend
start "ISL Backend" cmd /c "pip install -r requirements.txt && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
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
pause
