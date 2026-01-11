# Hướng dẫn Set TradingView Credentials

Hướng dẫn thiết lập environment variables cho TradingView username và password để sử dụng tvDatafeed với login.

## Tại sao cần credentials?

- Giảm lỗi 403 (Forbidden) từ TradingView
- Tăng rate limit và ổn định hơn
- Truy cập được nhiều dữ liệu hơn

## Các cách thiết lập

### Cách 1: Set cho Session hiện tại (PowerShell) - **Tạm thời**

Mở PowerShell và chạy:

```powershell
$env:TRADINGVIEW_USERNAME = "YourTradingViewUsername"
$env:TRADINGVIEW_PASSWORD = "YourTradingViewPassword"
```

**Lưu ý**: Các biến này chỉ tồn tại trong session PowerShell hiện tại. Khi đóng PowerShell, các biến sẽ mất.

### Cách 2: Set cho Session hiện tại (CMD) - **Tạm thời**

Mở Command Prompt (CMD) và chạy:

```cmd
set TRADINGVIEW_USERNAME=YourTradingViewUsername
set TRADINGVIEW_PASSWORD=YourTradingViewPassword
```

**Lưu ý**: Các biến này chỉ tồn tại trong session CMD hiện tại. Khi đóng CMD, các biến sẽ mất.

### Cách 3: Set vĩnh viễn cho User (PowerShell) - **Khuyên dùng**

Set vĩnh viễn cho user hiện tại (chỉ user này mới thấy):

```powershell
[System.Environment]::SetEnvironmentVariable('TRADINGVIEW_USERNAME', 'YourTradingViewUsername', 'User')
[System.Environment]::SetEnvironmentVariable('TRADINGVIEW_PASSWORD', 'YourTradingViewPassword', 'User')
```

**Sau khi set, cần:**
1. Đóng và mở lại terminal/PowerShell/CMD
2. Hoặc restart IDE/application để nhận biến mới

### Cách 4: Set vĩnh viễn qua GUI - **Dễ nhất**

1. **Mở System Properties:**
   - Nhấn `Win + R`
   - Gõ: `sysdm.cpl`
   - Nhấn Enter

2. **Hoặc qua Settings:**
   - Nhấn `Win + X`
   - Chọn "System"
   - Click "Advanced system settings"
   - Click "Environment Variables..."

3. **Thêm biến mới:**
   - Ở phần "User variables" (phía trên), click "New..."
   - Variable name: `TRADINGVIEW_USERNAME`
   - Variable value: `YourTradingViewUsername`
   - Click "OK"
   - Lặp lại với `TRADINGVIEW_PASSWORD`

4. **Áp dụng:**
   - Click "OK" để đóng tất cả dialog
   - **Quan trọng**: Đóng và mở lại terminal/IDE để nhận biến mới

## Kiểm tra xem đã set thành công chưa

### PowerShell:
```powershell
echo $env:TRADINGVIEW_USERNAME
echo $env:TRADINGVIEW_PASSWORD
```

### CMD:
```cmd
echo %TRADINGVIEW_USERNAME%
echo %TRADINGVIEW_PASSWORD%
```

Nếu hiển thị username và password của bạn (không phải trống), thì đã set thành công!

## Sử dụng trong Python script

Sau khi set environment variables, bạn có thể chạy script bình thường:

```bash
python main_gemini_chart_batch_scanner_forex.py
```

Hệ thống sẽ tự động lấy credentials từ environment variables.

## Xóa environment variables (nếu cần)

### PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable('TRADINGVIEW_USERNAME', $null, 'User')
[System.Environment]::SetEnvironmentVariable('TRADINGVIEW_PASSWORD', $null, 'User')
```

### Hoặc qua GUI:
1. Mở Environment Variables (như bước 4 ở trên)
2. Chọn biến cần xóa
3. Click "Delete"
4. Click "OK"

## Lưu ý bảo mật

- ⚠️ **KHÔNG** commit credentials vào git
- ⚠️ **KHÔNG** chia sẻ credentials
- ✅ Sử dụng environment variables (an toàn hơn hardcode)
- ✅ File `config/config_api.py` đã có trong `.gitignore`

## Troubleshooting

### Vấn đề: Script vẫn không nhận credentials

**Giải pháp:**
1. Đảm bảo đã set đúng tên biến: `TRADINGVIEW_USERNAME` và `TRADINGVIEW_PASSWORD`
2. Đóng và mở lại terminal/IDE
3. Kiểm tra lại bằng lệnh `echo` (xem phần "Kiểm tra" ở trên)
4. Nếu set trong một terminal, chạy script trong cùng terminal đó

### Vấn đề: Vẫn gặp lỗi 403

**Giải pháp:**
1. Kiểm tra username và password có đúng không
2. Thử login vào TradingView website bằng credentials đó
3. Nếu vẫn lỗi, có thể TradingView đang block request (thử lại sau)

### Vấn đề: Quên mất đã set credentials ở đâu

**Giải pháp:**
- Kiểm tra User Environment Variables qua GUI (bước 4)
- Hoặc chạy lệnh PowerShell:
  ```powershell
  [System.Environment]::GetEnvironmentVariable('TRADINGVIEW_USERNAME', 'User')
  ```
