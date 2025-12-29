# 🚀 Hướng dẫn nhanh: Cấu hình API Keys

## Cách nhanh nhất (Windows PowerShell):

1. **Mở PowerShell** trong thư mục project
2. **Chạy script:**
   ```powershell
   .\setup\setup_api_keys.ps1
   ```
3. **Nhập API keys** khi được yêu cầu
4. **Khởi động lại PowerShell** để áp dụng

## Các cách khác:

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

### Phương pháp khuyến nghị: Sử dụng file .env (Phát triển local)

**⚠️ QUAN TRỌNG: Không bao giờ commit file `.env` hoặc các secrets vào source control!**

1. **Tạo file `.env`** trong thư mục gốc của project:
   ```bash
   # .env (file này đã được gitignore)
   BINANCE_API_KEY=your-key
   BINANCE_API_SECRET=your-secret
   GEMINI_API_KEY=your-gemini-key
   ```

2. **Cài đặt python-dotenv** (nếu chưa có):
   ```bash
   pip install python-dotenv
   ```

3. **Load biến môi trường trong code Python**:
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()  # Load từ file .env
   
   api_key = os.getenv("BINANCE_API_KEY")
   ```

### Phương pháp thay thế: Biến môi trường hệ thống (Production/CI)

**Lưu ý:** Chỉ sử dụng phương pháp này cho production hoặc CI/CD pipelines. Đối với phát triển local, nên dùng file `.env`.

#### Windows (PowerShell):
```powershell
# Set vĩnh viễn cho User
[Environment]::SetEnvironmentVariable("BINANCE_API_KEY", "your-key", "User")
[Environment]::SetEnvironmentVariable("BINANCE_API_SECRET", "your-secret", "User")
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-gemini-key", "User")
```

#### Linux/Mac:
```bash
# Thêm vào ~/.bashrc hoặc ~/.zshrc
echo 'export BINANCE_API_KEY="your-key"' >> ~/.bashrc
echo 'export BINANCE_API_SECRET="your-secret"' >> ~/.bashrc
echo 'export GEMINI_API_KEY="your-gemini-key"' >> ~/.bashrc
source ~/.bashrc
```

### 🔐 Quản lý secrets cho Production

Đối với môi trường production, nên sử dụng các công cụ quản lý secrets chuyên nghiệp:

- **OS Keychain**: Windows Credential Manager, macOS Keychain, Linux Secret Service
- **Cloud Secret Managers**: 
  - AWS Secrets Manager / Parameter Store
  - Azure Key Vault
  - Google Cloud Secret Manager
  - HashiCorp Vault

**Lưu ý bảo mật:**
- ⚠️ **KHÔNG BAO GIỜ** commit API keys, secrets, hoặc file `.env` vào Git
- Luôn kiểm tra `.gitignore` đã bao gồm `.env` và các file chứa secrets
- Rotate (thay đổi) keys ngay lập tức nếu chúng bị exposed

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

## 🛡️ CẢNH BÁO BẢO MẬT:

- Sau khi set, **hãy khởi động lại terminal** để áp dụng các biến môi trường.
- **TUYỆT ĐỐI KHÔNG** commit các file chứa credentials như `.env`, `.bashrc`, `.zshrc` hoặc nội dung script xuất các biến môi trường có chứa API keys vào bất kỳ hệ thống quản lý phiên bản nào (VD: git).
    - Đảm bảo rằng các file này đã được thêm vào `.gitignore` (hoặc tương đương) trước khi thực hiện commit.
    - Kiểm tra kỹ lịch sử git để đảm bảo không có credential bị commit nhầm.
- Nếu credentials bị lộ trên repository hoặc đã bị commit (dù chỉ một lần), **hãy xoay vòng (rotate) hoặc thay đổi các keys đó NGAY LẬP TỨC** để đảm bảo an toàn.
- Tham khảo thêm chi tiết và hướng dẫn xử lý sự cố bảo mật trong `setup/SECURITY.md`.

## 🔗 Lấy API Keys:

- **Binance**: https://www.binance.com/en/my/settings/api-management
- **Google Gemini**: https://aistudio.google.com/app/apikey

