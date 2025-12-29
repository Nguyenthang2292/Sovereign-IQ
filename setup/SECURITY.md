# Security Notice - API Keys

## ⚠️ QUAN TRỌNG: API Keys đã bị exposed trong Git History

Nếu bạn đã commit file `config/config_api.py` với API keys hardcoded vào git repository, các keys đó đã bị exposed trong git history và có thể bị lộ.

## 🔒 Hành động cần thiết ngay lập tức:

1. **ROTATE (Thay đổi) tất cả API keys đã bị exposed:**
   - Binance: Vào [Binance API Management](https://www.binance.com/en/my/settings/api-management) và tạo keys mới, sau đó xóa keys cũ
   - Google Gemini: Vào [Google AI Studio](https://aistudio.google.com/app/apikey) và tạo key mới, sau đó xóa key cũ

2. **Xóa keys cũ khỏi git history (BẮT BUỘC cho mọi repository có exposed credentials):**

   ⚠️ **CẢNH BÁO QUAN TRỌNG:**
   - Việc xóa lịch sử Git sẽ **rewrite toàn bộ history** và yêu cầu force-push
   - **PHẢI phối hợp với tất cả collaborators** trước khi thực hiện
   - Tất cả collaborators cần re-clone repository sau khi history được cleanup
   - Áp dụng cho **CẢ public VÀ private repositories** - credentials exposed trong history đều nguy hiểm

   **BƯỚC 0: Sửa file config/config_api.py để đọc từ biến môi trường (BẮT BUỘC trước khi cleanup)**

   ⚠️ **QUAN TRỌNG:** 
   - **Nếu repository của bạn đã chứa các thay đổi này (xem phần "Đã được sửa" ở dưới, dòng 174-178)**, bạn chỉ cần **xác minh** rằng file `config/config_api.py` đã đọc từ biến môi trường, sau đó **bỏ qua** bước này.
   - **Nếu repository chưa có các thay đổi này**, bạn PHẢI sửa file `config/config_api.py` để đọc API keys từ biến môi trường và commit thay đổi này TRƯỚC KHI thực hiện bất kỳ bước cleanup git history nào. Điều này đảm bảo rằng sau khi cleanup, repository sẽ không còn chứa hardcoded keys.

   **Để xác minh hoặc áp dụng**, kiểm tra và đảm bảo file `config/config_api.py` có nội dung như sau:

   ```python
   import os

   # Binance API Configuration
   BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
   BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')

   # Google Gemini API Configuration
   GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
   ```

   **Nếu bạn vừa áp dụng các thay đổi** (chưa có trong repo), hãy commit thay đổi này:

   ```bash
   git add config/config_api.py
   git commit -m "Security: Update config_api.py to read from environment variables"
   ```

   **Nếu file đã có sẵn các thay đổi này**, bạn có thể bỏ qua bước commit và tiếp tục với BƯỚC 1.

   **BƯỚC 1: Tạo backup đầy đủ trước khi bắt đầu (BẮT BUỘC)**

   ⚠️ **QUAN TRỌNG:** Luôn tạo một bản backup hoàn chỉnh trước khi chạy bất kỳ lệnh rewrite history nào. Nếu có lỗi xảy ra, bạn có thể khôi phục từ backup.

   ```bash
   # Tạo một clone backup hoàn chỉnh của repository (Khuyến nghị)
   cd ..
   git clone --mirror <repository-url> backup-repo.git
   
   # Hoặc tạo backup local
   cp -r <current-repo> <current-repo>-backup
   ```

   **BƯỚC 2: Chọn phương pháp cleanup**

   **Phương án A: Sử dụng git filter-branch (Built-in, không cần cài thêm)**

   ```bash
   # Xóa file config/config_api.py khỏi toàn bộ history
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch config/config_api.py" \
     --prune-empty --tag-name-filter cat -- --all

   # Hoặc nếu muốn xóa theo pattern (ví dụ: tất cả file chứa "api_key")
   git filter-branch --force --tree-filter \
     "find . -name '*api_key*' -type f -delete" \
     --prune-empty --tag-name-filter cat -- --all
   ```

   ⚠️ **LƯU Ý QUAN TRỌNG về git filter-branch:**
   - `git filter-branch` có thể **rất chậm** trên các repository lớn với nhiều commits (có thể mất hàng giờ hoặc thậm chí nhiều ngày)
   - Đối với repository có lịch sử lớn hoặc nhiều commits, nên **ưu tiên sử dụng các công cụ nhanh hơn** như:
     - **BFG Repo-Cleaner** (xem Phương án B bên dưới) - nhanh hơn 10-50 lần
     - **git-filter-repo** (công cụ được Git khuyến nghị thay thế cho filter-branch) - nhanh và mạnh mẽ hơn
   - Nhớ **tạo backup đầy đủ** trước khi chạy bất kỳ lệnh rewrite history nào

   **Phương án B: Sử dụng BFG Repo-Cleaner hoặc git-filter-repo (Nhanh hơn, khuyến nghị cho repo lớn)**

   **BFG Repo-Cleaner:**

   ```bash
   # Cài đặt BFG (cần Java)
   # Windows: choco install bfg hoặc download từ https://rtyley.github.io/bfg-repo-cleaner/
   # Linux/Mac: brew install bfg hoặc download JAR file

   # Xóa file cụ thể
   bfg --delete-files config/config_api.py

   # Hoặc xóa theo pattern
   bfg --delete-files '*api_key*'

   # Sau khi chạy BFG, cần cleanup
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

   ⚠️ **LƯU Ý:** Flag `--aggressive` trong `git gc` có thể rất chậm trên các repository lớn và có thể tăng đáng kể thời gian thực thi. Nếu quan ngại về thời gian chạy, bạn có thể bỏ qua flag `--aggressive` (chỉ dùng `git gc --prune=now`), hoặc chuẩn bị sẵn sàng cho thời gian chạy dài.

   **git-filter-repo (Công cụ được Git khuyến nghị):**

   ```bash
   # Cài đặt git-filter-repo
   # Windows: pip install git-filter-repo
   # Linux/Mac: pip install git-filter-repo hoặc brew install git-filter-repo

   # Xóa file cụ thể
   git filter-repo --path config/config_api.py --invert-paths

   # Hoặc xóa theo pattern
   git filter-repo --path-glob '*api_key*' --invert-paths
   ```

   **BƯỚC 3: Cleanup và force-push (Áp dụng cho CẢ hai phương án)**

   ```bash
   # Expire tất cả reflogs để xóa references đến old commits
   git reflog expire --expire=now --all

   # Garbage collection với aggressive pruning để xóa hoàn toàn old objects
   git gc --prune=now --aggressive

   # ⚠️ CẢNH BÁO VỀ BRANCH PROTECTION:
   # Force-push sẽ THẤT BẠI trên các branches được bảo vệ (protected branches).
   # Trước khi chạy các lệnh dưới, bạn PHẢI:
   # 1. Kiểm tra và xác minh branch protection rules trên remote repository
   # 2. Tạm thời vô hiệu hóa branch protection, HOẶC
   # 3. Phối hợp với repository administrators để họ thực hiện force-push, HOẶC
   # 4. Làm việc trên branch không được bảo vệ, HOẶC
   # 5. Tạo repository mới nếu không thể thay đổi protection rules
   # PHẢI có sự chấp thuận từ quản trị viên trước khi tiếp tục.

   # Force-push tất cả branches (CẢNH BÁO: Sẽ overwrite remote history)
   git push origin --force --all

   # Force-push tất cả tags
   git push origin --force --tags
   ```

   ⚠️ **CẢNH BÁO VỀ BRANCH PROTECTION:**

   Các lệnh force-push ở trên **sẽ thất bại** nếu repository có branch protection policies (thường gặp trong các tổ chức). Trước khi thực hiện force-push, bạn cần:

   - **Tạm thời vô hiệu hóa branch protection** trên remote repository trước khi force-push, hoặc
   - **Phối hợp với repository administrators** để họ thực hiện force-push thay cho bạn

   Điều này giúp tránh gặp lỗi không mong muốn giữa chừng quá trình cleanup. Nếu bạn không có quyền quản trị repository, hãy liên hệ với team lead hoặc repository owner để được hỗ trợ.

   **BƯỚC 4: Thông báo collaborators**
   - Tất cả collaborators PHẢI re-clone repository:

     ```bash
     # Collaborators cần xóa local repo và clone lại
     rm -rf <local-repo>
     git clone <repository-url>
     ```

   - Hoặc nếu muốn giữ local changes, reset hard:

     ```bash
     git fetch origin
     git reset --hard origin/main  # hoặc origin/master
     git clean -fd
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

4. **Xác minh ứng dụng đã đọc đúng từ biến môi trường:**
   Sau khi đặt biến môi trường, hãy kiểm tra rằng ứng dụng hoặc file `config/config_api.py` thực sự đọc giá trị từ biến môi trường (không phải hardcode hay file khác).

   Chạy đoạn lệnh dưới để xác thực:
   ```bash
   # Chạy Python shell hoặc file test
   python -c "import os; print('BINANCE_API_KEY:', os.environ.get('BINANCE_API_KEY')); print('BINANCE_API_SECRET:', os.environ.get('BINANCE_API_SECRET')); print('GEMINI_API_KEY:', os.environ.get('GEMINI_API_KEY'))"
   ```

   Nếu kết quả trả về đúng giá trị bạn đã đặt, quá trình cấu hình đã thành công.

## ✅ Đã được sửa:

- ✅ File `config/config_api.py` đã được thêm vào `.gitignore`
- ✅ File `config/config_api.py` giờ đọc từ biến môi trường thay vì hardcode
- ✅ Đã tạo file template `config/config_api.py.example` để hướng dẫn

## 📝 Cách sử dụng an toàn:

### Cách 1: Sử dụng Script tự động (Khuyến nghị - Dễ nhất)

#### Windows (PowerShell):
```powershell
# Chạy script với quyền User (không cần Admin)
.\setup\setup_api_keys.ps1

# Hoặc chạy với quyền Administrator để set System-wide
# Right-click PowerShell > Run as Administrator, sau đó:
.\setup\setup_api_keys.ps1
```

#### Windows (Command Prompt):
```cmd
setup\setup_api_keys.bat
```

#### Linux/Mac:
```bash
chmod +x setup/setup_api_keys.sh
./setup/setup_api_keys.sh
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

