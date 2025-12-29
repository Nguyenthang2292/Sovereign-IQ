@echo off
REM Batch Script to configure API Keys using environment variables (User scope)
REM Run this script to set environment variables for the current User

echo === Configure API Keys for Crypto Probability ===
echo.

echo Enter your API Keys (or press Enter to skip):
echo.

REM Binance API Key
echo Enter your Binance API Key (note: The string you enter will be displayed publicly on screen).
echo For security, make sure no one is observing the screen or recording when you enter this information.
set /p BINANCE_API_KEY_INPUT="Binance API Key: "
if not "%BINANCE_API_KEY_INPUT%"=="" (
    echo [OK] BINANCE_API_KEY will be set
) else (
    echo [SKIP] Skipping BINANCE_API_KEY
)
REM (Environment variable will be set in the status-check section below)
REM If you need more secure input, please run the PowerShell version with hidden input:

REM Binance API Secret - DO NOT enter here to avoid public display.
REM [SECURITY WARNING] Do not enter Binance API Secret directly here as it will be displayed publicly (weak security).
REM To enter Binance API Secret securely, run the following PowerShell script with hidden input:

echo [WARNING] Do not enter Binance API Secret here to avoid information leakage!
echo To enter Binance API Secret more securely, run PowerShell below:
echo.
echo     powershell -ExecutionPolicy Bypass -NoProfile -Command ^
    "$secret = Read-Host -AsSecureString 'Binance API Secret'; ^
    [System.Environment]::SetEnvironmentVariable('BINANCE_API_SECRET', (New-Object PSCredential 'x', $secret).GetNetworkCredential().Password, 'User'); ^
    Write-Host '[OK] BINANCE_API_SECRET has been set (PowerShell)'"
echo.

REM Gemini API Key
set /p GEMINI_API_KEY_INPUT="Google Gemini API Key: "
if not "%GEMINI_API_KEY_INPUT%"=="" (
    echo [OK] GEMINI_API_KEY will be set
) else (
    echo [SKIP] Skipping GEMINI_API_KEY
)

REM Check setx result for BINANCE_API_KEY
if defined BINANCE_API_KEY_INPUT (
    setlocal EnableDelayedExpansion
    setx BINANCE_API_KEY "!BINANCE_API_KEY_INPUT!" >nul
    if errorlevel 1 (
        set "BINANCE_STATUS_TMP=FAILED"
    ) else (
        set "BINANCE_STATUS_TMP=SUCCESS"
    )
    for /f "delims=" %%X in ('echo !BINANCE_STATUS_TMP!') do (
        endlocal
        set "BINANCE_STATUS=%%X"
    )
) else (
    set "BINANCE_STATUS=SKIPPED"
)

REM Display BINANCE_API_KEY save status to the user
if "%BINANCE_STATUS%"=="SUCCESS" (
    echo [OK] BINANCE_API_KEY was saved successfully.
) else if "%BINANCE_STATUS%"=="FAILED" (
    echo [ERROR] Failed to save BINANCE_API_KEY!
) else if "%BINANCE_STATUS%"=="SKIPPED" (
    echo [SKIP] BINANCE_API_KEY was not set (skipped).
)

REM Check setx result for GEMINI_API_KEY
if defined GEMINI_API_KEY_INPUT (
    setlocal EnableDelayedExpansion
    setx GEMINI_API_KEY "!GEMINI_API_KEY_INPUT!" >nul
    if errorlevel 1 (
        set "GEMINI_STATUS_TMP=FAILED"
    ) else (
        set "GEMINI_STATUS_TMP=SUCCESS"
    )
    for /f "delims=" %%X in ('echo !GEMINI_STATUS_TMP!') do (
        endlocal
        set "GEMINI_STATUS=%%X"
    )
) else (
    set "GEMINI_STATUS=SKIPPED"
)

echo ==========================================
echo.

echo Important security note:
echo   Environment variables provide only limited security.
echo   Security Notice:
echo   Storing API keys in environment variables is convenient and allows easy integration with scripts,
echo   but be aware that these values may be visible to other users on this system, and can persist longer than intended.
echo   For stronger protection, consider using solutions such as Windows Credential Manager or a dedicated secrets management service.
echo   Choose the approach that best fits your security requirements.

echo.
echo IMPORTANT: Environment variables set by 'setx' are NOT available in the current session.
echo You need to:
echo   1. Close this Command Prompt and open a new one
echo   2. Restart any applications that need to access these environment variables
echo.
echo To verify in a NEW Command Prompt window, run: echo %%BINANCE_API_KEY%%
pause

