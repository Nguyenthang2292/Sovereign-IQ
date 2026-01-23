@echo off
REM Build script for Rust extensions on Windows

REM Check if Rust is installed
set "CARGO_BIN=%USERPROFILE%\.cargo\bin"
if not exist "%CARGO_BIN%\rustc.exe" (
    echo [ERROR] Rust is not installed at %CARGO_BIN%
    echo Please install Rust from: https://rustup.rs/
    exit /b 1
)

REM Add Rust to PATH for this session
echo [SETUP] Adding Rust to PATH...
set "PATH=%CARGO_BIN%;%PATH%"

REM Verify Rust is available
"%CARGO_BIN%\rustc.exe" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Rust compiler (rustc) is not accessible!
    echo Please restart your terminal after installing Rust.
    exit /b 1
)

echo [BUILD] Navigating to rust_extensions directory...
cd modules\adaptive_trend_enhance_v2\rust_extensions

echo [BUILD] Building and installing Rust extensions (maturin develop --release)...
maturin develop --release
if errorlevel 1 (
    echo [ERROR] maturin build failed!
    exit /b 1
)

echo [SUCCESS] Rust extensions installed successfully.
cd ..\..\..
