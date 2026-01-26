@echo off
REM Rebuild Rust extensions after CUDA kernel fixes

echo ========================================
echo Rebuilding Rust Extensions with CUDA
echo ========================================
echo.

cd /d "%~dp0"
cd modules\adaptive_trend_LTS\rust_extensions

echo Current directory: %CD%
echo.

echo Cleaning previous build...
cargo clean
if errorlevel 1 (
    echo ERROR: cargo clean failed
    pause
    exit /b 1
)

echo.
echo Building Rust extensions in release mode...
maturin develop --release
if errorlevel 1 (
    echo ERROR: maturin build failed
    echo.
    echo Make sure you have:
    echo 1. Rust toolchain installed (rustup)
    echo 2. CUDA Toolkit installed (if using CUDA features)
    echo 3. Python development headers
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo The atc_rust module has been rebuilt with:
echo - NVRTC-compatible CUDA kernels
echo - Index preservation fixes
echo.
echo You can now run the benchmark:
echo   cd modules\adaptive_trend_LTS\benchmarks\benchmark_comparison
echo   python main.py --symbols 20 --bars 500 --timeframe 1h
echo.
pause
