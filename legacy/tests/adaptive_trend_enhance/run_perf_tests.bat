@echo off
REM Quick Performance Test Runner for Windows
REM Usage: run_perf_tests.bat [mode] [suite]
REM   mode: fast (default) | full | ci
REM   suite: all (default) | regression | comparison

setlocal

set MODE=%1
set SUITE=%2
if "%MODE%"=="" set MODE=fast
if "%SUITE%"=="" set SUITE=all

echo ============================================
echo Performance Test Runner
echo ============================================
echo.

REM Determine test files to run
if "%SUITE%"=="all" (
    set TEST_FILES=tests\adaptive_trend_enhance\test_performance_regression.py tests\adaptive_trend_enhance\test_performance.py
) else if "%SUITE%"=="regression" (
    set TEST_FILES=tests\adaptive_trend_enhance\test_performance_regression.py
) else if "%SUITE%"=="comparison" (
    set TEST_FILES=tests\adaptive_trend_enhance\test_performance.py
) else (
    echo Unknown suite: %SUITE%
    echo.
    echo Available suites:
    echo   all        - Run both test files (default)
    echo   regression - Run only test_performance_regression.py
    echo   comparison - Run only test_performance.py
    echo.
    echo Example: run_perf_tests.bat fast all
    exit /b 1
)

echo Suite: %SUITE%
echo.

if "%MODE%"=="fast" (
    echo Mode: FAST DEVELOPMENT
    echo - Iterations: 3
    echo - Skip slow tests
    echo - Single threaded
    echo.
    pytest %TEST_FILES% -n 0 -m "not slow" -v
) else if "%MODE%"=="full" (
    echo Mode: FULL TEST SUITE
    echo - Iterations: 5
    echo - All tests
    echo - Single threaded
    echo.
    set PERF_ITERATIONS=5
    pytest %TEST_FILES% -n 0 -m performance -v
) else if "%MODE%"=="ci" (
    echo Mode: CI/PRODUCTION
    echo - Iterations: 10
    echo - All tests with coverage
    echo - Single threaded
    echo.
    set PERF_ITERATIONS=10
    pytest %TEST_FILES% -n 0 -m performance --cov=modules.adaptive_trend_enhance --cov-report=html -v
) else (
    echo Unknown mode: %MODE%
    echo.
    echo Available modes:
    echo   fast - Fast development testing (3 iterations, skip slow)
    echo   full - Full test suite (5 iterations, all tests)
    echo   ci   - CI/Production (10 iterations, with coverage)
    echo.
    echo Example: run_perf_tests.bat fast
    exit /b 1
)

echo.
echo ============================================
echo Tests completed!
echo ============================================

endlocal
