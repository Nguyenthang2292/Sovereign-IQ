@echo off
echo Cleaning Python cache...
cd web\atc_visualizer\backend
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo Cache cleaned!
echo.
echo Starting backend...
echo.
python api.py
