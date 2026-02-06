# Cập nhật skills cho .agent, .claude, .opencode

Chạy các lệnh dưới khi cần kéo bản mới nhất của skills. Trên Windows cần có Git và (nếu dùng HTTPS) credentials đã cấu hình.

---

## 1. @.agent (Antigravity Awesome Skills)

**Cài đặt trong project (repo hiện tại):**
```powershell
git -C ".agent" pull
```
*(Từ thư mục gốc của project crypto-probability.)*

**Cài đặt global (trong home):**
```powershell
git -C "$env:USERPROFILE\.agent\skills" pull
```
*(Chỉ chạy nếu bạn đã clone skills vào `~/.agent/skills`.)*

---

## 2. @.claude (Claude Code)

- **Personal (toàn máy):** `~/.claude/skills/` — nếu thư mục này là clone git:
  ```powershell
  git -C "$env:USERPROFILE\.claude\skills" pull
  ```
- **Project:** `.claude/skills/` trong từng repo — nếu là submodule hoặc clone:
  ```powershell
  git -C ".claude\skills" pull
  ```

*(Tạo `~/.claude/skills` hoặc `.claude/skills` và clone repo skills nếu chưa có.)*

---

## 3. @.opencode (OpenCode)

- **Global:** `~/.config/opencode/skills/` (Linux/macOS) hoặc `%USERPROFILE%\.config\opencode\skills\` (Windows):
  ```powershell
  git -C "$env:USERPROFILE\.config\opencode\skills" pull
  ```
- **Project:** `.opencode/skills/` trong repo:
  ```powershell
  git -C ".opencode\skills" pull
  ```

*(Chỉ chạy nếu thư mục tương ứng đã tồn tại và là git repo.)*

---

## Lỗi thường gặp

- **SEC_E_NO_CREDENTIALS / unable to access HTTPS:** Cấu hình Git credentials (HTTPS) hoặc dùng SSH:
  ```powershell
  git -C ".agent" remote set-url origin git@github.com:sickn33/antigravity-awesome-skills.git
  git -C ".agent" pull
  ```
- **Path not found:** Thư mục skills chưa tồn tại — clone repo tương ứng vào đúng đường dẫn trước khi chạy `pull`.
