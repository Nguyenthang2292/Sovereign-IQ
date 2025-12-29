# Setup và Cấu hình API Keys

Folder này chứa các file hướng dẫn và scripts để cấu hình API keys cho project.

## 📁 Nội dung

- **QUICK_START_API_KEYS.md**: Hướng dẫn nhanh để cấu hình API keys
- **SECURITY.md**: Thông tin bảo mật và best practices cho API keys
- **setup_api_keys.ps1**: PowerShell script cho Windows
- **setup_api_keys.bat**: Batch script cho Windows Command Prompt
- **setup_api_keys.sh**: Bash script cho Linux/Mac

## 🚀 Bắt đầu nhanh

### Windows (PowerShell):
```powershell
.\setup\setup_api_keys.ps1
```

### Windows (Command Prompt):
```cmd
setup\setup_api_keys.bat
```

### Linux/Mac:
```bash
chmod +x setup/setup_api_keys.sh
./setup/setup_api_keys.sh
source ~/.bashrc
```

## 📖 Tài liệu

- Xem [QUICK_START_API_KEYS.md](./QUICK_START_API_KEYS.md) để biết hướng dẫn chi tiết
- Xem [SECURITY.md](./SECURITY.md) để biết về bảo mật và best practices

## ⚠️ Lưu ý quan trọng

- **KHÔNG BAO GIỜ** commit API keys vào git repository
- Nếu keys đã bị exposed, hãy **rotate (thay đổi) keys ngay lập tức**
- Sử dụng biến môi trường thay vì hardcode keys trong code

