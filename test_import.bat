@echo off
echo ========================================
echo  Testing Backend Import
echo ========================================
echo.

cd web\atc_visualizer\backend
python test_import.py

echo.
echo ========================================
echo  If SUCCESS above, backend should work!
echo  Run: test_backend.bat
echo ========================================
pause
