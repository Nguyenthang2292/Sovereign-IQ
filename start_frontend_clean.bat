@echo off
echo ========================================
echo  Kill Ports and Start Frontend Only
echo ========================================
echo.

echo [1/2] Cleaning ports 5173-5176...
for %%p in (5173 5174 5175 5176) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p') do (
        taskkill /F /PID %%a 2>nul
    )
)

echo.
echo [2/2] Starting frontend...
cd web\atc_visualizer\frontend
npm run dev
