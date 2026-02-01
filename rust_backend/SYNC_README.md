# Rust Backend Synchronization

## Quy trình Đồng bộ

### Source of Truth

`modules/adaptive_trend_LTS_mini/rust_extensions/` là **source chính**.

### Khi Sửa Rust Code

1. **Sửa code** trong `modules/adaptive_trend_LTS_mini/rust_extensions/src/`
2. **Chạy script sync**:

   ```powershell
   .\sync_rust.ps1
   ```

3. **Build lại rust_backend**:

   ```powershell
   cd rust_backend
   python -m maturin build --release
   ```

4. **Cài đặt wheel mới**:

   ```powershell
   pip install target/wheels/sovereign_prime-0.1.0-cp312-cp312-win_amd64.whl --force-reinstall
   ```

### Lưu ý

- Script `sync_rust.ps1` sẽ **copy** tất cả `.rs` files (trừ `lib.rs`) từ source sang `rust_backend/src/`
- File `lib.rs` trong `rust_backend` được giữ riêng để expose functions dưới module name `sovereign_prime`

## Alternative: Symbolic Links (Nâng cao)

Nếu muốn sync tự động hoàn toàn, có thể dùng symlinks:

```powershell
# Xóa thư mục src hiện tại
Remove-Item rust_backend\src -Recurse -Force

# Tạo symlink (cần Run as Administrator)
New-Item -ItemType SymbolicLink -Path rust_backend\src -Target modules\adaptive_trend_LTS_mini\rust_extensions\src
```

**Lưu ý**: Cách này cần quyền Administrator và có thể gây conflict với `lib.rs`.
