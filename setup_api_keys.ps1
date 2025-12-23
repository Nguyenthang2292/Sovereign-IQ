# PowerShell Script để cấu hình API Keys bằng biến môi trường
# Chạy script này trong PowerShell với quyền Administrator (nếu muốn set system-wide)
# Hoặc chạy bình thường để set cho session hiện tại

Write-Host "=== Cấu hình API Keys cho Crypto Probability ===" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra xem đang chạy với quyền Administrator không
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($isAdmin) {
    Write-Host "Đang chạy với quyền Administrator - sẽ set biến môi trường System-wide" -ForegroundColor Yellow
    $scope = "Machine"
} else {
    Write-Host "Đang chạy với quyền User - sẽ set biến môi trường cho User hiện tại" -ForegroundColor Yellow
    $scope = "User"
}

Write-Host ""
Write-Host "Nhập API Keys của bạn (hoặc nhấn Enter để bỏ qua):" -ForegroundColor Green
Write-Host ""

# Binance API Key
$binanceKey = Read-Host "Binance API Key"
if ($binanceKey) {
    [Environment]::SetEnvironmentVariable("BINANCE_API_KEY", $binanceKey, $scope)
    Write-Host "✓ Đã set BINANCE_API_KEY" -ForegroundColor Green
} else {
    Write-Host "⊘ Bỏ qua BINANCE_API_KEY" -ForegroundColor Gray
}

# Binance API Secret
$binanceSecret = Read-Host "Binance API Secret"
if ($binanceSecret) {
    [Environment]::SetEnvironmentVariable("BINANCE_API_SECRET", $binanceSecret, $scope)
    Write-Host "✓ Đã set BINANCE_API_SECRET" -ForegroundColor Green
} else {
    Write-Host "⊘ Bỏ qua BINANCE_API_SECRET" -ForegroundColor Gray
}

# Gemini API Key
$geminiKey = Read-Host "Google Gemini API Key"
if ($geminiKey) {
    [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $geminiKey, $scope)
    Write-Host "✓ Đã set GEMINI_API_KEY" -ForegroundColor Green
} else {
    Write-Host "⊘ Bỏ qua GEMINI_API_KEY" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Hoàn tất ===" -ForegroundColor Cyan
Write-Host ""

# Hiển thị các biến đã set (ẩn giá trị thực)
Write-Host "Các biến môi trường đã được cấu hình:" -ForegroundColor Yellow
if ($binanceKey) {
    Write-Host "  BINANCE_API_KEY = [Đã set]" -ForegroundColor Green
}
if ($binanceSecret) {
    Write-Host "  BINANCE_API_SECRET = [Đã set]" -ForegroundColor Green
}
if ($geminiKey) {
    Write-Host "  GEMINI_API_KEY = [Đã set]" -ForegroundColor Green
}

Write-Host ""
Write-Host "Lưu ý:" -ForegroundColor Yellow
Write-Host "- Nếu set cho User, bạn cần khởi động lại terminal/PowerShell để áp dụng" -ForegroundColor Gray
Write-Host "- Nếu set cho System, bạn cần khởi động lại máy hoặc đăng nhập lại" -ForegroundColor Gray
Write-Host "- Để kiểm tra, chạy: `$env:BINANCE_API_KEY" -ForegroundColor Gray

