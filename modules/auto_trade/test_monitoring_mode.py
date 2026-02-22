"""
Chạy hệ thống auto_trade ở chế độ chỉ giám sát (monitoring only).

- Không đặt lệnh thật (dry_run=True).
- Vẫn chạy vòng lặp: kiểm tra vị thế, quét tín hiệu, backup DB.
- Dùng để test pipeline và monitoring mà không giao dịch.

Cách chạy (từ project root):
    python modules/auto_trade/test_monitoring_mode.py

Hoặc từ thư mục modules/auto_trade:
    python test_monitoring_mode.py
"""

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
import sys
from pathlib import Path


# Thêm project root vào path
_current_file = Path(__file__).resolve()
_auto_trade_dir = _current_file.parent
_modules_dir = _auto_trade_dir.parent
_project_root = _modules_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def run_monitoring() -> None:
    """Deprecated entrypoint retained for backward compatibility."""
    log_warn(
        "modules.auto_trade.main has been removed. "
        "Use the GUI dashboard entrypoint instead (main.py at repository root)."
    )


if __name__ == "__main__":
    run_monitoring()
