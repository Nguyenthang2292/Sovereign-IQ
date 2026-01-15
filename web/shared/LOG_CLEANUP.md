# Log Cleanup Configuration

Tài liệu này mô tả cách cấu hình tự động dọn dẹp logs cũ trong hệ thống.

## Tổng quan

Hệ thống tự động dọn dẹp logs cũ trước khi tạo log file mới để:
- Giảm dung lượng disk
- Tránh tích lũy quá nhiều log files
- Cải thiện hiệu suất

## Cấu hình

### Environment Variables

Có thể cấu hình qua biến môi trường:

#### `LOG_AUTO_CLEANUP`
- **Mặc định**: `true`
- **Giá trị**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Mô tả**: Bật/tắt tự động cleanup logs cũ trước khi tạo log mới

**Ví dụ:**
```bash
# Windows PowerShell
$env:LOG_AUTO_CLEANUP = "true"

# Windows Command Prompt
set LOG_AUTO_CLEANUP=true

# Linux/Mac
export LOG_AUTO_CLEANUP=true
```

#### `LOG_MAX_AGE_HOURS`
- **Mặc định**: `24`
- **Giá trị**: Số giờ (integer)
- **Mô tả**: Tuổi tối đa của log files trước khi bị xóa (tính bằng giờ)

**Ví dụ:**
```bash
# Windows PowerShell
$env:LOG_MAX_AGE_HOURS = "48"  # Giữ logs trong 48 giờ

# Windows Command Prompt
set LOG_MAX_AGE_HOURS=48

# Linux/Mac
export LOG_MAX_AGE_HOURS=48
```

### Programmatic Configuration

Có thể cấu hình khi khởi tạo `LogFileManager`:

```python
from web.utils.log_manager import LogFileManager

# Tắt auto-cleanup
log_manager = LogFileManager(
    auto_cleanup_before_new=False,
    max_log_age_hours=48
)

# Bật auto-cleanup với custom max age
log_manager = LogFileManager(
    auto_cleanup_before_new=True,
    max_log_age_hours=12  # Giữ logs trong 12 giờ
)
```

## Hoạt động

### Auto-Cleanup Behavior

1. **Khi nào cleanup chạy:**
   - Tự động chạy trước khi tạo log file mới (khi front-end gọi request mới)
   - Chỉ chạy nếu đã đủ thời gian kể từ lần cleanup cuối (mặc định: 5 phút)
   - Điều này tránh overhead khi có nhiều requests liên tiếp

2. **Logs nào bị xóa:**
   - Tất cả log files (`.log`) trong folder `logs/`
   - Chỉ xóa logs cũ hơn `LOG_MAX_AGE_HOURS` giờ
   - Dựa trên thời gian modification time (`mtime`) của file

3. **Thread Safety:**
   - Cleanup được bảo vệ bởi lock để tránh race conditions
   - An toàn khi có nhiều requests đồng thời

### Manual Cleanup

Có thể cleanup thủ công bất cứ lúc nào:

```python
from web.utils.log_manager import get_log_manager

log_manager = get_log_manager()

# Cleanup logs cũ hơn 24 giờ (mặc định)
deleted_count = log_manager.cleanup_old_logs()

# Cleanup logs cũ hơn 12 giờ
deleted_count = log_manager.cleanup_old_logs(max_age_hours=12)

print(f"Đã xóa {deleted_count} log files")
```

## Best Practices

1. **Production:**
   - Bật auto-cleanup (`LOG_AUTO_CLEANUP=true`)
   - Đặt `LOG_MAX_AGE_HOURS` phù hợp với nhu cầu (ví dụ: 24-48 giờ)
   - Monitor disk usage để điều chỉnh nếu cần

2. **Development:**
   - Có thể tắt auto-cleanup để giữ logs lâu hơn cho debugging
   - Hoặc đặt `LOG_MAX_AGE_HOURS` cao hơn (ví dụ: 168 giờ = 7 ngày)

3. **Debugging:**
   - Nếu cần giữ logs lâu hơn, tăng `LOG_MAX_AGE_HOURS`
   - Hoặc tắt auto-cleanup tạm thời: `LOG_AUTO_CLEANUP=false`

## Monitoring

Logs về cleanup được ghi vào application logs:

```
INFO: Cleaned up 5 old log file(s) (older than 24 hours)
DEBUG: Auto-cleaned 3 old log file(s) before creating new log
```

## Troubleshooting

### Logs không bị xóa
- Kiểm tra `LOG_AUTO_CLEANUP` có được set đúng không
- Kiểm tra `LOG_MAX_AGE_HOURS` có quá cao không
- Kiểm tra file permissions

### Cleanup chạy quá thường xuyên
- Đây là bình thường, cleanup chỉ chạy khi tạo log mới
- Có cơ chế throttle (5 phút) để tránh overhead

### Muốn giữ logs lâu hơn
- Tăng `LOG_MAX_AGE_HOURS`
- Hoặc tắt auto-cleanup và cleanup thủ công khi cần

