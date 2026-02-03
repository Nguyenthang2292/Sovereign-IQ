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

import asyncio
import logging
import signal as signal_module
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Thêm project root vào path
_current_file = Path(__file__).resolve()
_auto_trade_dir = _current_file.parent
_modules_dir = _auto_trade_dir.parent
_project_root = _modules_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from modules.auto_trade.auto_trade_config import AutoTradeConfig, load_config
from modules.auto_trade.main import AutoTradeSystem


def get_monitoring_config() -> AutoTradeConfig:
    """
    Load config cho chế độ chỉ giám sát.
    Ưu tiên: config.json trong auto_trade → mặc định từ env, luôn bật dry_run.
    """
    config_path = _auto_trade_dir / "config.json"
    if config_path.exists():
        config = AutoTradeConfig.from_json(str(config_path))
    else:
        # Tạo config mặc định với dry_run=True để không bắt buộc API key khi chỉ chạy monitoring
        config = AutoTradeConfig(dry_run=True)
    config.dry_run = True  # Luôn chỉ giám sát, không đặt lệnh
    return config


async def run_monitoring():
    """Khởi tạo hệ thống và chạy vòng lặp giám sát (main_loop)."""
    config = get_monitoring_config()
    system = AutoTradeSystem(config=config)

    # Xử lý tắt gọn (Ctrl+C, SIGTERM)
    signal_module.signal(signal_module.SIGINT, system.signal_handler)
    signal_module.signal(signal_module.SIGTERM, system.signal_handler)

    try:
        await system.initialize()
        # Chạy vòng lặp chính (scan, monitor positions, backup; không execute order vì dry_run)
        await system.main_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await system.shutdown()


if __name__ == "__main__":
    print("=" * 60)
    print("Chế độ chỉ giám sát (Monitoring only) - dry_run=True")
    print("Không đặt lệnh thật. Dừng: Ctrl+C")
    print("=" * 60)
    asyncio.run(run_monitoring())
