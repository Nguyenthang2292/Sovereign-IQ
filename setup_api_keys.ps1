# PowerShell Script to configure API Keys using environment variables
# Run this script in PowerShell with Administrator privileges (if you want to set system-wide)
# Or run normally to set for current session

Write-Host "=== Configure API Keys for Crypto Probability ===" -ForegroundColor Cyan
Write-Host ""

# Check if running with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($isAdmin) {
    Write-Host "Running with Administrator privileges - will set System-wide environment variables" -ForegroundColor Yellow
    $scope = "Machine"
} else {
    Write-Host "Running with User privileges - will set environment variables for current User" -ForegroundColor Yellow
    $scope = "User"
}

Write-Host ""
Write-Host "Enter your API Keys (or press Enter to skip):" -ForegroundColor Green
Write-Host ""

# Binance API Key
$binanceKey = Read-Host "Binance API Key"
if ($binanceKey) {
    [Environment]::SetEnvironmentVariable("BINANCE_API_KEY", $binanceKey, $scope)
    Write-Host "[OK] BINANCE_API_KEY has been set" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Skipped BINANCE_API_KEY" -ForegroundColor Gray
}

# Binance API Secret
$binanceSecret = Read-Host "Binance API Secret"
if ($binanceSecret) {
    [Environment]::SetEnvironmentVariable("BINANCE_API_SECRET", $binanceSecret, $scope)
    Write-Host "[OK] BINANCE_API_SECRET has been set" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Skipped BINANCE_API_SECRET" -ForegroundColor Gray
}

# Gemini API Key
$geminiKey = Read-Host "Google Gemini API Key"
if ($geminiKey) {
    [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $geminiKey, $scope)
    Write-Host "[OK] GEMINI_API_KEY has been set" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Skipped GEMINI_API_KEY" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Complete ===" -ForegroundColor Cyan
Write-Host ""

# Display configured variables (hide actual values)
Write-Host "Environment variables that have been configured:" -ForegroundColor Yellow
if ($binanceKey) {
    Write-Host "  BINANCE_API_KEY = [Set]" -ForegroundColor Green
}
if ($binanceSecret) {
    Write-Host "  BINANCE_API_SECRET = [Set]" -ForegroundColor Green
}
if ($geminiKey) {
    Write-Host "  GEMINI_API_KEY = [Set]" -ForegroundColor Green
}

Write-Host ""
Write-Host "Note:" -ForegroundColor Yellow
Write-Host "- If set for User, you need to restart terminal/PowerShell to apply" -ForegroundColor Gray
Write-Host "- If set for System, you need to restart the machine or log in again" -ForegroundColor Gray
Write-Host '- To check, run: $env:BINANCE_API_KEY' -ForegroundColor Gray
