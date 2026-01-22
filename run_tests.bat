@echo off
REM Batch script to run tests with venv activated
REM Usage: run_tests.bat [pytest arguments]

echo.
echo ========================================
echo 🚀 Activating venv and running pytest...
echo ========================================
echo.

REM Check if venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ ERROR: venv not found at .venv
    echo Please create venv first: python -m venv .venv
    exit /b 1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Verify Python from venv
echo 🐍 Using Python: %VIRTUAL_ENV%
python --version

REM Check if pytest is installed
python -m pytest --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: pytest not installed in venv
    echo Installing pytest...
    python -m pip install pytest pytest-xdist pytest-cov pytest-timeout
)

echo.
echo ========================================
echo 🧪 Running tests...
echo ========================================
echo.

REM Run pytest with all arguments passed to this script
if "%*"=="" (
    REM Default: run all tests
    python -m pytest tests -v --tb=short
) else (
    REM Run with custom arguments
    python -m pytest %*
)

set TEST_EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %TEST_EXIT_CODE%==0 (
    echo ✅ Tests completed successfully!
) else (
    echo ❌ Tests failed with exit code: %TEST_EXIT_CODE%
)
echo ========================================

exit /b %TEST_EXIT_CODE%
