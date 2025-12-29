# PowerShell Script to configure API Keys using environment variables
# Run this script in PowerShell with Administrator privileges to set system-wide (Machine scope)
# Or run normally to set for current user (User scope)

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

# Helper function to set API key environment variable
function Set-ApiKey {
    param(
        [string]$KeyName,
        [string]$PromptText,
        [string]$Scope
    )
    
    $existingValue = [Environment]::GetEnvironmentVariable($KeyName, $Scope)
    if ($existingValue) {
        $overwrite = Read-Host "$KeyName already exists. Overwrite? (Y/N)"
        if ($overwrite -ne "Y" -and $overwrite -ne "y") {
            Write-Host "[SKIP] $KeyName kept existing value" -ForegroundColor Gray
            return
        }
    }
    
    $secureString = $null
    $bstr = [IntPtr]::Zero
    $plainValue = $null
    
    try {
        $secureString = Read-Host $PromptText -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureString)
        $plainValue = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
        
        if ($plainValue) {
            try {
                [Environment]::SetEnvironmentVariable($KeyName, $plainValue, $Scope)
                Write-Host "[OK] $KeyName has been set" -ForegroundColor Green
            } catch {
                Write-Host "[ERROR] Failed to set $KeyName : $($_.Exception.Message)" -ForegroundColor Red
            }
        } else {
            if ($existingValue) {
                Write-Host "[SKIP] $KeyName kept existing value" -ForegroundColor Gray
            } else {
                Write-Host "[SKIP] Skipped $KeyName" -ForegroundColor Gray
            }
        }
    } catch {
        Write-Host "[ERROR] Failed to convert $KeyName : $($_.Exception.Message)" -ForegroundColor Red
    } finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
        if ($plainValue) {
            $plainValue = $null
        }
        if ($secureString) {
            $secureString.Dispose()
        }
    }
}

# Set API Keys using the helper function
# Prompt for Binance API Key (minimum 16 chars suggested)
do {
    Set-ApiKey -KeyName "BINANCE_API_KEY" -PromptText "Binance API Key" -Scope $scope
    $binanceApiKey = [Environment]::GetEnvironmentVariable("BINANCE_API_KEY", $scope)
    if ($binanceApiKey -and $binanceApiKey.Length -gt 0 -and $binanceApiKey.Length -ge 16) { break }
    Write-Host "[WARN] Binance API Key seems too short or empty. Please try again." -ForegroundColor Yellow
} while ($true)

# Prompt for Binance API Secret (minimum 32 chars suggested)
do {
    Set-ApiKey -KeyName "BINANCE_API_SECRET" -PromptText "Binance API Secret" -Scope $scope
    $binanceApiSecret = [Environment]::GetEnvironmentVariable("BINANCE_API_SECRET", $scope)
    if ($binanceApiSecret -and $binanceApiSecret.Length -gt 0 -and $binanceApiSecret.Length -ge 32) { break }
    Write-Host "[WARN] Binance API Secret seems too short or empty. Please try again." -ForegroundColor Yellow
} while ($true)

# Prompt for Gemini API Key (minimum 20 chars suggested)
do {
    Set-ApiKey -KeyName "GEMINI_API_KEY" -PromptText "Google Gemini API Key" -Scope $scope
    $geminiApiKey = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", $scope)
    if ($geminiApiKey -and $geminiApiKey.Length -gt 0 -and $geminiApiKey.Length -ge 20) { break }
    Write-Host "[WARN] Gemini API Key seems too short or empty. Please try again." -ForegroundColor Yellow
} while ($true)

# Display configured variables (hide actual values)
Write-Host "Environment variables that have been configured:" -ForegroundColor Yellow
if ([Environment]::GetEnvironmentVariable("BINANCE_API_KEY", $scope)) {
    Write-Host "  BINANCE_API_KEY = [Set]" -ForegroundColor Green
}
if ([Environment]::GetEnvironmentVariable("BINANCE_API_SECRET", $scope)) {
    Write-Host "  BINANCE_API_SECRET = [Set]" -ForegroundColor Green
}
if ([Environment]::GetEnvironmentVariable("GEMINI_API_KEY", $scope)) {
    Write-Host "  GEMINI_API_KEY = [Set]" -ForegroundColor Green
}

Write-Host ""
Write-Host "Note:" -ForegroundColor Yellow
Write-Host "- If set for User, you need to restart terminal/PowerShell to apply" -ForegroundColor Gray
Write-Host "- If set for System, you need to restart the machine or log in again" -ForegroundColor Gray
Write-Host '- To check, run: $env:BINANCE_API_KEY' -ForegroundColor Gray
