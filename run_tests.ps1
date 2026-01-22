# PowerShell script to run tests with venv activated
# Usage: .\run_tests.ps1 [pytest arguments]

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "Activating venv and running pytest..."
Write-ColorOutput Yellow "================================================"

# Check if venv exists
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-ColorOutput Red "ERROR: venv not found at .\.venv"
    Write-ColorOutput Yellow "Please create venv first: python -m venv .venv"
    exit 1
}

# Activate venv
Write-ColorOutput Cyan "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

# Verify Python from venv
$pythonPath = (Get-Command python).Source
Write-ColorOutput Cyan "Using Python: $pythonPath"

# Check if pytest is installed
try {
    $pytestVersion = & python -m pytest --version 2>&1
    Write-ColorOutput Cyan "Pytest: $pytestVersion"
}
catch {
    Write-ColorOutput Red "ERROR: pytest not installed in venv"
    Write-ColorOutput Yellow "Installing pytest..."
    & python -m pip install pytest pytest-xdist pytest-cov pytest-timeout
}

Write-ColorOutput Yellow "================================================"
Write-ColorOutput Green "Running tests..."
Write-ColorOutput Yellow "================================================"
Write-Output ""

# Run pytest with all arguments passed to this script
if ($args.Count -eq 0) {
    # Default: run all tests with coverage
    & python -m pytest tests -v --tb=short
}
else {
    # Run with custom arguments
    & python -m pytest @args
}

$exitCode = $LASTEXITCODE

Write-Output ""
Write-ColorOutput Yellow "================================================"
if ($exitCode -eq 0) {
    Write-ColorOutput Green "Tests completed successfully!"
}
else {
    Write-ColorOutput Red "Tests failed with exit code: $exitCode"
}
Write-ColorOutput Yellow "================================================"

exit $exitCode
