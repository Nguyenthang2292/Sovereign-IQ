# 🚀 Hướng dẫn nhanh: Cấu hình API Keys

## Cách nhanh nhất (Windows PowerShell):

1. **Mở PowerShell** trong thư mục project
2. **Chạy script:**
   ```powershell
   .\setup_api_keys.ps1
   ```
3. **Nhập API keys** khi được yêu cầu
4. **Khởi động lại PowerShell** để áp dụng

## Các cách khác:

### Windows (Command Prompt):
```cmd
setup_api_keys.bat
```

### Linux/Mac:
```bash
chmod +x setup_api_keys.sh
./setup_api_keys.sh
source ~/.bashrc
```

### Set thủ công (PowerShell):
```powershell
# Set vĩnh viễn cho User
[Environment]::SetEnvironmentVariable("BINANCE_API_KEY", "your-key", "User")
[Environment]::SetEnvironmentVariable("BINANCE_API_SECRET", "your-secret", "User")
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-gemini-key", "User")
```

### Set thủ công (Linux/Mac):
```bash
# Thêm vào ~/.bashrc hoặc ~/.zshrc
echo 'export BINANCE_API_KEY="your-key"' >> ~/.bashrc
echo 'export BINANCE_API_SECRET="your-secret"' >> ~/.bashrc
echo 'export GEMINI_API_KEY="your-gemini-key"' >> ~/.bashrc
source ~/.bashrc
```

## ✅ Kiểm tra:

### Windows (PowerShell):
```powershell
$env:BINANCE_API_KEY
$env:BINANCE_API_SECRET
$env:GEMINI_API_KEY
```

### Linux/Mac:
```bash
echo $BINANCE_API_KEY
echo $BINANCE_API_SECRET
echo $GEMINI_API_KEY
```

## 📝 Lưu ý:

- Sau khi set, **khởi động lại terminal** để áp dụng
- Nếu keys đã bị exposed trong git, **hãy rotate (thay đổi) keys ngay lập tức**
- Xem `SECURITY.md` để biết thêm chi tiết

## 🔗 Lấy API Keys:

- **Binance**: https://www.binance.com/en/my/settings/api-management
- **Google Gemini**: https://aistudio.google.com/app/apikey

