@echo off
echo Starting ATC Visualizer Setup...
echo.

echo ========================================
echo 1. Installing Backend Dependencies
echo ========================================
cd web\atc_visualizer\backend
pip install -r requirements.txt

echo.
echo ========================================
echo 2. Installing Frontend Dependencies
echo ========================================
cd ..\frontend
call npm install

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo.
echo 1. Start Backend (Terminal 1):
echo    cd web\atc_visualizer\backend
echo    python api.py
echo.
echo    Or use uvicorn directly:
echo    uvicorn api:app --host 0.0.0.0 --port 5000 --reload
echo.
echo 2. Start Frontend (Terminal 2):
echo    cd web\atc_visualizer\frontend
echo    npm run dev
echo.
echo Then open: http://localhost:5173
echo API Docs: http://localhost:5000/docs
echo.

cd ..\..\..
pause
