# PowerShell script to build Rust extensions
# This script handles PATH issues better than batch files

Write-Host "[BUILD] Building Rust extensions..." -ForegroundColor Cyan

# Check if Rust is installed
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
$rustcExe = Join-Path $cargoBin "rustc.exe"

if (-not (Test-Path $rustcExe)) {
    Write-Host "[ERROR] Rust is not installed at $cargoBin" -ForegroundColor Red
    Write-Host "Please install Rust from: https://rustup.rs/" -ForegroundColor Yellow
    exit 1
}

# Add Rust to PATH for this session
Write-Host "[SETUP] Adding Rust to PATH..." -ForegroundColor Yellow
$env:PATH = "$cargoBin;$env:PATH"

# Verify Rust is available
try {
    $rustVersion = & "$rustcExe" --version 2>&1
    Write-Host "[SUCCESS] Rust found: $rustVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Rust compiler (rustc) is not accessible!" -ForegroundColor Red
    Write-Host "Please restart your terminal after installing Rust." -ForegroundColor Yellow
    exit 1
}

# List of Rust extension directories to build
$rustProjectDirs = @(
    "modules\adaptive_trend_LTS\rust_extensions",
    "modules\xgboost_LTS\rust_extensions"
)

foreach ($relativeDir in $rustProjectDirs) {
    $rustExtDir = Join-Path $PSScriptRoot $relativeDir
    if (-not (Test-Path $rustExtDir)) {
        Write-Host "[WARNING] Rust extensions directory not found: $rustExtDir" -ForegroundColor Yellow
        continue
    }

    Write-Host "[BUILD] Building extensions in $relativeDir..." -ForegroundColor Cyan
    Push-Location $rustExtDir

    try {
        Write-Host "[BUILD] Running maturin develop --release..." -ForegroundColor Cyan
        
        # Check if maturin is available
        $maturinPath = Get-Command maturin -ErrorAction SilentlyContinue
        if (-not $maturinPath) {
            Write-Host "[WARNING] Maturin not found. Installing..." -ForegroundColor Yellow
            pip install maturin
        }
        
        # Build with maturin
        & maturin develop --release
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] maturin build failed for $relativeDir!" -ForegroundColor Red
            # Don't exit immediately, try other projects but record failure
            $hasError = $true
        } else {
            Write-Host "[SUCCESS] Extensions in $relativeDir installed successfully!" -ForegroundColor Green
        }
    } finally {
        Pop-Location
    }
}

if ($hasError) {
    exit 1
}
