# Cấu hình Git credentials (HTTPS) trên Windows

Đã cấu hình Git dùng **credential helper** để lưu thông tin đăng nhập khi kéo/push qua HTTPS.

---

## Đã thiết lập

```bash
git config --global credential.helper manager
```

- **manager** = Git Credential Manager (GCM), lưu credentials trong Windows Credential Manager.
- Nếu máy dùng Git for Windows 2.29+ thì GCM thường đã có sẵn.

---

## Lần đầu dùng HTTPS (GitHub)

GitHub **không còn dùng mật khẩu** cho HTTPS. Bạn cần dùng **Personal Access Token (PAT)**.

### 1. Tạo PAT trên GitHub

1. Vào **GitHub.com** → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**.
2. **Generate new token (classic)**.
3. Đặt tên (ví dụ: `git-https`), chọn quyền **repo** (đủ cho clone/push/pull).
4. Generate → **copy token** (chỉ hiện một lần).

### 2. Dùng token khi Git hỏi

Khi chạy lệnh cần HTTPS (ví dụ `git pull`):

- **Username:** GitHub username của bạn (ví dụ: `Nguyenthang2292`).
- **Password:** dán **PAT** (token vừa tạo), **không** dùng mật khẩu GitHub.

Sau lần nhập, GCM lưu vào Windows Credential Manager; các lần sau không cần nhập lại (trừ khi token hết hạn hoặc bị xóa).

### 3. (Tùy chọn) Chỉ dùng PAT cho GitHub

Để tránh nhầm với host khác:

```bash
git config --global credential.https://github.com.helper manager
```

---

## Kiểm tra

```powershell
git config --global --get credential.helper
```

Kết quả mong đợi: `manager`.

Sau đó chạy pull trong repo (ví dụ `.agent`):

```powershell
git -C ".agent" pull
```

Nếu lần đầu, cửa sổ đăng nhập hoặc trình duyệt sẽ mở; dùng **username** + **PAT** như trên.

---

## Nếu không có Git Credential Manager

- Cài **Git for Windows** mới (2.29+) từ https://git-scm.com/download/win (đã kèm GCM), hoặc
- Cài **Git Credential Manager** từ https://github.com/git-ecosystem/git-credential-manager/releases.

Sau khi cài, chạy lại:

```bash
git config --global credential.helper manager
```

---

## Bảo mật

- **Không** commit PAT hoặc mật khẩu vào repo.
- PAT nên có quyền tối thiểu (ví dụ chỉ **repo**) và đặt expiry nếu có thể.
- Xem/sửa credentials đã lưu: **Windows** → **Control Panel** → **Credential Manager** → **Windows Credentials** → mục `git:https://github.com`.
