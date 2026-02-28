# =====================================================================
# Binance Proxy — SSH SOCKS5 Tunnel (Windows Auto-start)
# File: C:\tools\binance_proxy\start_tunnel.ps1
#
# Cách dùng:
#   1. Copy file này và oracle-binance-key.pem vào C:\tools\binance_proxy\
#   2. Sửa ORACLE_IP bên dưới thành Reserved IP của bạn
#   3. Đăng ký Task Scheduler bằng register_task.ps1
# =====================================================================

# ===== CONFIG — SỬA Ở ĐÂY =====
$ORACLE_IP   = "REPLACE_WITH_YOUR_ORACLE_RESERVED_IP"   # VD: 129.146.xxx.xxx
$SSH_KEY     = "$PSScriptRoot\oracle-binance-key.pem"
$SSH_USER    = "ubuntu"
$LOCAL_PORT  = 1080
$RETRY_DELAY = 5   # giây chờ trước khi reconnect
# ================================

$LogFile = "$PSScriptRoot\tunnel.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

# Fix SSH key permissions (Windows yêu cầu chỉ owner mới đọc được)
function Set-KeyPermissions {
    param([string]$KeyPath)
    if (Test-Path $KeyPath) {
        icacls $KeyPath /inheritance:r | Out-Null
        icacls $KeyPath /grant:r "$($env:USERNAME):(R)" | Out-Null
        Write-Log "Key permissions set for: $KeyPath"
    } else {
        Write-Log "ERROR: SSH key not found at: $KeyPath"
        Write-Log "Please place oracle-binance-key.pem in: $PSScriptRoot"
        exit 1
    }
}

Write-Log "=== Binance Proxy Tunnel Starting ==="
Write-Log "Oracle IP  : $ORACLE_IP"
Write-Log "Local SOCKS: 127.0.0.1:$LOCAL_PORT"
Write-Log "SSH Key    : $SSH_KEY"

Set-KeyPermissions -KeyPath $SSH_KEY

# Auto-reconnect loop
$attempt = 0
while ($true) {
    $attempt++
    Write-Log "Connecting (attempt #$attempt)..."

    # Start SSH tunnel — -D = SOCKS5 dynamic forwarding, -N = no remote command
    $proc = Start-Process -FilePath "ssh" -ArgumentList @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",      # ping mỗi 30s để giữ kết nối
        "-o", "ServerAliveCountMax=3",       # ngắt nếu 3 ping liên tiếp fail
        "-o", "ExitOnForwardFailure=yes",    # thoát nếu port forward thất bại
        "-o", "ConnectTimeout=15",
        "-o", "TCPKeepAlive=yes",
        "-i", $SSH_KEY,
        "-D", $LOCAL_PORT,
        "-N",                                 # không mở shell
        "$SSH_USER@$ORACLE_IP"
    ) -NoNewWindow -PassThru -Wait

    $exitCode = $proc.ExitCode
    Write-Log "Tunnel exited (code: $exitCode). Reconnecting in ${RETRY_DELAY}s..."
    Start-Sleep -Seconds $RETRY_DELAY
}
