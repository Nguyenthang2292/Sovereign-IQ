@echo off
echo ========================================
echo  Killing Processes on Ports 5000, 5173
echo ========================================
echo.

echo Checking port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    echo Killing process %%a on port 5000...
    taskkill /F /PID %%a 2>nul
)

echo.
echo Checking port 5173...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173') do (
    echo Killing process %%a on port 5173...
    taskkill /F /PID %%a 2>nul
)

echo.
echo ========================================
echo  Cleanup Complete!
echo ========================================
echo.
pause
