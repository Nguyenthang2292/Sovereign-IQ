"""
API Configuration for external services.
File này chứa các API keys và secrets - KHÔNG commit lên git!

Để sử dụng:
1. Đặt API keys của bạn vào biến môi trường (khuyến nghị) hoặc
2. Tạo file config/config_api.py từ config/config_api.py.example và điền API keys
3. File này đã được thêm vào .gitignore

LƯU Ý BẢO MẬT:
- KHÔNG hardcode API keys trực tiếp trong file này
- Luôn sử dụng biến môi trường hoặc file local không được commit
- Nếu file này đã bị commit với keys, hãy rotate (thay đổi) keys ngay lập tức
"""

import os
from contextlib import contextmanager

# Binance API Configuration
# Ưu tiên đọc từ biến môi trường, nếu không có thì để None
# Để bảo mật, luôn sử dụng biến môi trường:
#   export BINANCE_API_KEY='your-key-here'
#   export BINANCE_API_SECRET='your-secret-here'
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or None
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET") or None

# Google Gemini API Configuration
# Lấy API key từ: https://makersuite.google.com/app/apikey
# Hoặc: https://aistudio.google.com/app/apikey
# Ưu tiên đọc từ biến môi trường, nếu không có thì để None
# Để bảo mật, luôn sử dụng biến môi trường:
#   export GEMINI_API_KEY='your-api-key-here'
#   hoặc trong PowerShell: $env:GEMINI_API_KEY='your-api-key-here'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None

# Helper function to read from Windows Registry (for cases where env var was set via [Environment]::SetEnvironmentVariable)
# This handles the case where Python process doesn't see newly set environment variables
def _read_from_registry(env_var_name):
    """
    Try to read environment variable from Windows Registry.
    Tries HKEY_CURRENT_USER first, then HKEY_LOCAL_MACHINE if not found.
    Returns the value if found, None otherwise.
    """
    try:
        import winreg
        
        # Context manager for winreg keys (requires Python 3.2+)
        @contextmanager
        def open_registry_key(hkey, subkey):
            """Context manager for Windows registry keys that automatically closes them."""
            key = None
            try:
                key = winreg.OpenKey(hkey, subkey)
                yield key
            finally:
                if key is not None:
                    winreg.CloseKey(key)
        
        # Try User environment first (HKEY_CURRENT_USER\Environment)
        try:
            with open_registry_key(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                value = winreg.QueryValueEx(key, env_var_name)[0] or None
                if value is not None:
                    return value
        except (OSError, FileNotFoundError):
            # Registry key doesn't exist or access denied
            pass
        
        # If still None, try Machine environment (HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment)
        try:
            with open_registry_key(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                value = winreg.QueryValueEx(key, env_var_name)[0] or None
                if value is not None:
                    return value
        except (OSError, FileNotFoundError):
            # Registry key doesn't exist or access denied
            pass
        
        return None
    except ImportError:
        # winreg not available (non-Windows system or Python < 3.2)
        return None

# If still None, try reading from Windows Registry
if BINANCE_API_KEY is None:
    BINANCE_API_KEY = _read_from_registry("BINANCE_API_KEY")

if BINANCE_API_SECRET is None:
    BINANCE_API_SECRET = _read_from_registry("BINANCE_API_SECRET")

if GEMINI_API_KEY is None:
    GEMINI_API_KEY = _read_from_registry("GEMINI_API_KEY")

