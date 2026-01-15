@echo off
echo ========================================
echo  Kill Ports -^> Start ATC Visualizer
echo ========================================
echo.

echo [1/2] Cleaning up ports (5000, 5173-5176)...
python kill_ports.py

echo.
echo [2/2] Starting ATC Visualizer...
echo.
python run_atc_visualizer.py --skip-npm-check
