@echo off
echo ========================================
echo     ATC Visualizer - Quick Start
echo ========================================
echo.

python run_atc_visualizer.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error occurred. Press any key to exit...
    pause >nul
)
