# =====================================================================
# Đăng ký SSH Tunnel vào Windows Task Scheduler
# Chạy file này 1 lần với quyền Administrator
#
# Usage: Right-click → "Run as Administrator"
#        hoặc: Start-Process powershell -Verb RunAs -ArgumentList "-File register_task.ps1"
# =====================================================================

$TaskName = "BinanceProxyTunnel"
$ScriptPath = "C:\tools\binance_proxy\start_tunnel.ps1"
$ToolsDir = "C:\tools\binance_proxy"

# Tạo thư mục nếu chưa có
if (-not (Test-Path $ToolsDir)) {
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    Write-Host "Created directory: $ToolsDir"
}

# Copy script vào C:\tools\binance_proxy\
$sourceScript = Join-Path $PSScriptRoot "start_tunnel.ps1"
$sourceKey = Join-Path $PSScriptRoot "oracle-binance-key.pem"

if (Test-Path $sourceScript) {
    Copy-Item $sourceScript $ScriptPath -Force
    Write-Host "Copied start_tunnel.ps1 to $ToolsDir"
}

if (Test-Path $sourceKey) {
    Copy-Item $sourceKey "$ToolsDir\oracle-binance-key.pem" -Force
    Write-Host "Copied oracle-binance-key.pem to $ToolsDir"
}
else {
    Write-Warning "oracle-binance-key.pem not found in $PSScriptRoot"
    Write-Warning "Please manually copy your .pem key to: $ToolsDir"
}

# Xóa task cũ nếu tồn tại
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

# Tạo task mới
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$ScriptPath`""

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `   # Không timeout
-RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable $true

$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Binance Proxy: SSH SOCKS5 tunnel qua Oracle Cloud VM (fixed IP)" `
    -Force

Write-Host ""
Write-Host "✅ Task registered: $TaskName"
Write-Host ""

# Hỏi có muốn start ngay không
$start = Read-Host "Start tunnel now? (Y/n)"
if ($start -ne "n" -and $start -ne "N") {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "✅ Tunnel started. Check task status:"
    Write-Host "   Get-ScheduledTask -TaskName '$TaskName' | Select-Object State"
    Start-Sleep -Seconds 3
    
    # Verify tunnel is listening
    $listening = netstat -an | Select-String "127.0.0.1:1080"
    if ($listening) {
        Write-Host "✅ SOCKS5 proxy listening on 127.0.0.1:1080"
    }
    else {
        Write-Host "⚠️  Port 1080 not yet listening — tunnel may still be connecting..."
        Write-Host "   Wait 10s then run: netstat -an | Select-String 1080"
    }
}

Write-Host ""
Write-Host "=== Next Steps ==="
Write-Host "1. Verify IP: python tools\oracle_proxy\check_proxy.py"
Write-Host "2. Whitelist the shown IP on Binance API Management"
Write-Host "3. Start app: python run_auto_trade_gui.py"
