@echo off
echo ========================================
echo  Testing Backend Server
echo ========================================
echo.
echo Starting backend at http://localhost:5000
echo.

cd web\atc_visualizer\backend
python api.py
