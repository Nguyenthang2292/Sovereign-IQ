# Security Notice - API Keys

## ⚠️ QUAN TRỌNG: API Keys đã bị exposed trong Git History

Nếu bạn đã commit file `config/config_api.py` với API keys hardcoded vào git repository, các keys đó đã bị exposed trong git history và có thể bị lộ.

## 🔒 Hành động cần thiết ngay lập tức:

1. **ROTATE (Thay đổi) tất cả API keys đã bị exposed:**
   - Binance: Vào [Binance API Management](https://www.binance.com/en/my/settings/api-management) và tạo keys mới, sau đó xóa keys cũ
   - Google Gemini: Vào [Google AI Studio](https://aistudio.google.com/app/apikey) và tạo key mới, sau đó xóa key cũ

2. **Xóa keys cũ khỏi git history (nếu repository là private và bạn có quyền):**
   ```bash
   # Sử dụng git filter-branch hoặc BFG Repo-Cleaner để xóa file khỏi history
   # Lưu ý: Chỉ làm điều này nếu repository là private
   ```

3. **Cấu hình lại API keys bằng biến môi trường (khuyến nghị):**
   ```bash
   # Windows (PowerShell)
   $env:BINANCE_API_KEY='your-new-key-here'
   $env:BINANCE_API_SECRET='your-new-secret-here'
   $env:GEMINI_API_KEY='your-new-gemini-key-here'
   
   # Linux/Mac
   export BINANCE_API_KEY='your-new-key-here'
   export BINANCE_API_SECRET='your-new-secret-here'
   export GEMINI_API_KEY='your-new-gemini-key-here'
   ```

## ✅ Đã được sửa:

- ✅ File `config/config_api.py` đã được thêm vào `.gitignore`
- ✅ File `config/config_api.py` giờ đọc từ biến môi trường thay vì hardcode
- ✅ Đã tạo file template `config/config_api.py.example` để hướng dẫn

## 📝 Cách sử dụng an toàn:

### Cách 1: Sử dụng Script tự động (Khuyến nghị - Dễ nhất)

#### Windows (PowerShell):
```powershell
# Chạy script với quyền User (không cần Admin)
.\setup_api_keys.ps1

# Hoặc chạy với quyền Administrator để set System-wide
# Right-click PowerShell > Run as Administrator, sau đó:
.\setup_api_keys.ps1
```

#### Windows (Command Prompt):
```cmd
setup_api_keys.bat
```

#### Linux/Mac:
```bash
chmod +x setup_api_keys.sh
./setup_api_keys.sh
source ~/.bashrc  # Hoặc ~/.zshrc tùy shell của bạn
```

### Cách 2: Set thủ công bằng biến môi trường

#### Windows (PowerShell):
```powershell
# Set cho session hiện tại
$env:BINANCE_API_KEY='your-key-here'
$env:BINANCE_API_SECRET='your-secret-here'
$env:GEMINI_API_KEY='your-gemini-key-here'

# Set vĩnh viễn cho User (không cần Admin)
[Environment]::SetEnvironmentVariable("BINANCE_API_KEY", "your-key-here", "User")
[Environment]::SetEnvironmentVariable("BINANCE_API_SECRET", "your-secret-here", "User")
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-gemini-key-here", "User")

# Set vĩnh viễn cho System (cần Admin)
[Environment]::SetEnvironmentVariable("BINANCE_API_KEY", "your-key-here", "Machine")
```

#### Windows (Command Prompt):
```cmd
# Set vĩnh viễn cho User
setx BINANCE_API_KEY "your-key-here"
setx BINANCE_API_SECRET "your-secret-here"
setx GEMINI_API_KEY "your-gemini-key-here"
```

#### Linux/Mac:
```bash
# Thêm vào ~/.bashrc hoặc ~/.zshrc
export BINANCE_API_KEY='your-key-here'
export BINANCE_API_SECRET='your-secret-here'
export GEMINI_API_KEY='your-gemini-key-here'

# Áp dụng ngay
source ~/.bashrc  # hoặc source ~/.zshrc
```

### Cách 3: Tạo file local (chỉ cho development - KHÔNG khuyến nghị)
```bash
# Copy template
cp config/config_api.py.example config/config_api.py
# Điền API keys vào file (file này đã được .gitignore)
```

## ✅ Kiểm tra cấu hình:

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

**LƯU Ý:** 
- File `config/config_api.py` đã được thêm vào `.gitignore`, nhưng nếu bạn đã commit nó trước đó, nó vẫn tồn tại trong git history. Hãy rotate keys ngay lập tức!
- Sau khi set biến môi trường, bạn cần khởi động lại terminal/PowerShell để áp dụng (hoặc reload shell config)

