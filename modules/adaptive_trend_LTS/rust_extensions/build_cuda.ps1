# Build script for Rust extensions with CUDA support
# Handles CUDA library path with spaces on Windows

$ErrorActionPreference = "Stop"

Write-Host "Building Rust extensions with CUDA support..." -ForegroundColor Cyan

# Set CUDA library path (adjust version if needed)
$cudaLibPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64"

if (Test-Path $cudaLibPath) {
    Write-Host "Found CUDA libraries at: $cudaLibPath" -ForegroundColor Green
    
    # Use short path to avoid spaces issue
    $shortPath = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($cudaLibPath).ShortPath
    $env:RUSTFLAGS = "-L $shortPath"
    
    Write-Host "Set RUSTFLAGS=-L $shortPath" -ForegroundColor Yellow
}
else {
    Write-Host "WARNING: CUDA libraries not found at $cudaLibPath" -ForegroundColor Red
    Write-Host "CUDA kernels may not build correctly." -ForegroundColor Red
}

# Build with maturin
Write-Host "`nBuilding with maturin..." -ForegroundColor Cyan
maturin develop --release

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "You can now use CUDA-accelerated functions from atc_rust module." -ForegroundColor Green
}
else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
    exit 1
}
