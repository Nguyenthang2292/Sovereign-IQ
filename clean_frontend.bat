@echo off
echo ========================================
echo  Cleaning Frontend Dependencies
echo ========================================
echo.

cd web\atc_visualizer\frontend

if exist node_modules (
    echo Removing node_modules...
    rmdir /s /q node_modules
)

if exist package-lock.json (
    echo Removing package-lock.json...
    del /f /q package-lock.json
)

if exist dist (
    echo Removing dist...
    rmdir /s /q dist
)

echo.
echo Cleaning Vite cache...
if exist .vite (
    rmdir /s /q .vite
)

echo.
echo ✅ Clean complete!
echo.
echo You can now run: start_visualizer_clean.bat
pause
