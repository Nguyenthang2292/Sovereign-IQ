@echo off
REM Batch Script để cấu hình API Keys bằng biến môi trường (User scope)
REM Chạy script này để set biến môi trường cho User hiện tại

echo === Cấu hình API Keys cho Crypto Probability ===
echo.

echo Nhập API Keys của bạn (hoặc nhấn Enter để bỏ qua):
echo.

REM Binance API Key
set /p BINANCE_API_KEY_INPUT="Binance API Key: "
if not "%BINANCE_API_KEY_INPUT%"=="" (
    setx BINANCE_API_KEY "%BINANCE_API_KEY_INPUT%"
    echo [OK] Đã set BINANCE_API_KEY
) else (
    echo [SKIP] Bỏ qua BINANCE_API_KEY
)

REM Binance API Secret
set /p BINANCE_API_SECRET_INPUT="Binance API Secret: "
if not "%BINANCE_API_SECRET_INPUT%"=="" (
    setx BINANCE_API_SECRET "%BINANCE_API_SECRET_INPUT%"
    echo [OK] Đã set BINANCE_API_SECRET
) else (
    echo [SKIP] Bỏ qua BINANCE_API_SECRET
)

REM Gemini API Key
set /p GEMINI_API_KEY_INPUT="Google Gemini API Key: "
if not "%GEMINI_API_KEY_INPUT%"=="" (
    setx GEMINI_API_KEY "%GEMINI_API_KEY_INPUT%"
    echo [OK] Đã set GEMINI_API_KEY
) else (
    echo [SKIP] Bỏ qua GEMINI_API_KEY
)

echo.
echo === Hoàn tất ===
echo.
echo Lưu ý: Bạn cần khởi động lại Command Prompt để biến môi trường có hiệu lực
echo Để kiểm tra, chạy: echo %%BINANCE_API_KEY%%
pause

