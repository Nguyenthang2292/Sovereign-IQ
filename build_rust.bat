@echo off
REM Build script for Rust extensions on Windows

echo [BUILD] Navigating to rust_extensions directory...
cd modules\adaptive_trend_enhance_v2\rust_extensions

echo [BUILD] Building and installing Rust extensions (maturin develop --release)...
maturin develop --release

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] maturin build failed!
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Rust extensions installed successfully.
cd ..\..\..
