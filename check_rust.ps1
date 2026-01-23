# Script to check Rust installation and setup PATH if needed

Write-Host "[CHECK] Checking Rust installation..." -ForegroundColor Cyan

# Check if Rust is in PATH
$rustcPath = Get-Command rustc -ErrorAction SilentlyContinue
$cargoPath = Get-Command cargo -ErrorAction SilentlyContinue

if ($rustcPath -and $cargoPath) {
    Write-Host "[SUCCESS] Rust is installed and in PATH:" -ForegroundColor Green
    rustc --version
    cargo --version
    Write-Host ""
    
    # Check Maturin
    Write-Host "[CHECK] Checking Maturin..." -ForegroundColor Cyan
    $maturinPath = Get-Command maturin -ErrorAction SilentlyContinue
    if ($maturinPath) {
        Write-Host "[SUCCESS] Maturin is installed:" -ForegroundColor Green
        maturin --version
    } else {
        Write-Host "[WARNING] Maturin is not installed." -ForegroundColor Yellow
        Write-Host "Installing Maturin..." -ForegroundColor Yellow
        pip install maturin
    }
    exit 0
}

# Rust not in PATH, check if installed
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
$rustcExe = Join-Path $cargoBin "rustc.exe"

if (Test-Path $rustcExe) {
    Write-Host "[INFO] Rust is installed but not in PATH" -ForegroundColor Yellow
    Write-Host "Adding Rust to PATH for this session..." -ForegroundColor Yellow
    $env:PATH = "$cargoBin;$env:PATH"
    
    # Verify it works now
    rustc --version
    cargo --version
    Write-Host ""
    Write-Host "[SUCCESS] Rust is now accessible in this session!" -ForegroundColor Green
    Write-Host "[NOTE] To make this permanent, restart your terminal or add to PATH:" -ForegroundColor Yellow
    Write-Host "  $cargoBin" -ForegroundColor Gray
    Write-Host ""
    
    # Check Maturin
    Write-Host "[CHECK] Checking Maturin..." -ForegroundColor Cyan
    $maturinPath = Get-Command maturin -ErrorAction SilentlyContinue
    if ($maturinPath) {
        Write-Host "[SUCCESS] Maturin is installed:" -ForegroundColor Green
        maturin --version
    } else {
        Write-Host "[WARNING] Maturin is not installed." -ForegroundColor Yellow
        Write-Host "Installing Maturin..." -ForegroundColor Yellow
        pip install maturin
    }
    exit 0
} else {
    Write-Host "[ERROR] Rust is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Rust from: https://rustup.rs/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Cyan
    Write-Host "1. Download rustup-init.exe from: https://win.rustup.rs/x86_64" -ForegroundColor White
    Write-Host "2. Run the installer and follow the instructions" -ForegroundColor White
    Write-Host "3. Restart your terminal after installation" -ForegroundColor White
    Write-Host "4. Run this script again to verify" -ForegroundColor White
    exit 1
}
