
from contextlib import contextmanager
import os
import threading

"""
API Configuration for external services.
This file contains API keys and secrets - DO NOT commit to git!

To use:
1. Put your API keys in environment variables (recommended), or
2. Create the file config/config_api.py from config/config_api.py.example and fill in the API keys
3. This file is already added to .gitignore

SECURITY NOTES:
- DO NOT hardcode API keys directly in this file
- Always use environment variables or a local file that is not committed
- If this file has been committed with keys, rotate (change) the keys immediately
"""


# Try to import winreg (Windows only)
try:
    import winreg
except ImportError:
    winreg = None  # winreg not available (non-Windows system or Python < 3.2)

# Lock for serializing writes/updates to module-level API key constants.
# Note: This lock only protects writes, not reads. For thread-safe reads,
# use get_api_keys() or the individual getter functions (get_binance_api_key(), etc.).
_api_keys_lock = threading.Lock()

# Context manager for Windows registry keys that automatically closes them
@contextmanager
def open_registry_key(hkey, subkey):
    """Context manager for Windows registry keys that automatically closes them."""
    if winreg is None:
        raise ImportError("winreg module is not available (non-Windows system)")
    key = None
    try:
        key = winreg.OpenKey(hkey, subkey)
        yield key
    finally:
        if key is not None:
            winreg.CloseKey(key)

# Helper function to read from Windows Registry (for cases where env var was set via [Environment]::SetEnvironmentVariable)
# This handles the case where Python process doesn't see newly set environment variables
def _read_from_registry(env_var_name):
    """
    Try to read environment variable from Windows Registry.
    Tries HKEY_CURRENT_USER first, then HKEY_LOCAL_MACHINE if not found.
    Returns the value if found, None otherwise.
    """
    if winreg is None:
        # winreg not available (non-Windows system or Python < 3.2)
        return None
    
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


def _get_key(env_var_name):
    """
    Read API key from environment variable, falling back to Windows Registry if not found.
    
    Empty string values are normalized to None (empty API keys are invalid and treated
    as missing). Downstream code should check for None and handle it appropriately
    (typically raising ValueError or using fallback authentication methods).
    
    Args:
        env_var_name: Name of the environment variable to read
        
    Returns:
        str or None: The API key value, or None if not found or empty
    """
    # Try environment variable first
    value = os.getenv(env_var_name) or None
    # If not found, try Windows Registry
    if value is None:
        value = _read_from_registry(env_var_name)
    return value


# Binance API Configuration
# Prefer to read from environment variables; if not present, set to None
# For security, always use environment variables:
#   export BINANCE_API_KEY='your-key-here'
#   export BINANCE_API_SECRET='your-secret-here'
BINANCE_API_KEY = _get_key("BINANCE_API_KEY")
BINANCE_API_SECRET = _get_key("BINANCE_API_SECRET")

# Google Gemini API Configuration
# Get API key from: https://makersuite.google.com/app/apikey
# Or: https://aistudio.google.com/app/apikey
# Prefer to read from environment variables; if not present, set to None
# For security, always use environment variables:
#   export GEMINI_API_KEY='your-api-key-here'
#   or in PowerShell: $env:GEMINI_API_KEY='your-api-key-here'
GEMINI_API_KEY = _get_key("GEMINI_API_KEY")


def load_api_keys():
    """
    Reload API keys from environment variables and Windows Registry.
    
    This function reads API keys from environment variables first,
    then falls back to Windows Registry if not found. It updates
    the module-level constants and returns a dictionary of the loaded keys.
    
    Thread-safety: The lock (_api_keys_lock) only serializes writes/updates
    to module-level constants. It does NOT protect concurrent reads of the
    module-level constants. For thread-safe reads, use get_api_keys() or
    the individual getter functions (get_binance_api_key(), etc.).
    
    Returns:
        dict: Dictionary with keys 'BINANCE_API_KEY', 'BINANCE_API_SECRET', 'GEMINI_API_KEY'
    """
    global BINANCE_API_KEY, BINANCE_API_SECRET, GEMINI_API_KEY
    
    # Use helper to read keys (env -> registry fallback)
    binance_key = _get_key("BINANCE_API_KEY")
    binance_secret = _get_key("BINANCE_API_SECRET")
    gemini_key = _get_key("GEMINI_API_KEY")
    
    # Thread-safe update of module-level constants
    with _api_keys_lock:
        BINANCE_API_KEY = binance_key
        BINANCE_API_SECRET = binance_secret
        GEMINI_API_KEY = gemini_key
    
    return {
        'BINANCE_API_KEY': binance_key,
        'BINANCE_API_SECRET': binance_secret,
        'GEMINI_API_KEY': gemini_key
    }


def get_api_keys():
    """
    Get all API keys in a thread-safe manner.
    
    This function acquires the lock before reading the module-level constants,
    ensuring thread-safe access when multiple threads may be reading or updating
    the keys concurrently.
    
    Returns:
        dict: Dictionary with keys 'BINANCE_API_KEY', 'BINANCE_API_SECRET', 'GEMINI_API_KEY'
    """
    with _api_keys_lock:
        return {
            'BINANCE_API_KEY': BINANCE_API_KEY,
            'BINANCE_API_SECRET': BINANCE_API_SECRET,
            'GEMINI_API_KEY': GEMINI_API_KEY
        }


def get_binance_api_key():
    """
    Get Binance API key in a thread-safe manner.
    
    Returns:
        str or None: The Binance API key, or None if not set
    """
    with _api_keys_lock:
        return BINANCE_API_KEY


def get_binance_api_secret():
    """
    Get Binance API secret in a thread-safe manner.
    
    Returns:
        str or None: The Binance API secret, or None if not set
    """
    with _api_keys_lock:
        return BINANCE_API_SECRET


def get_gemini_api_key():
    """
    Get Google Gemini API key in a thread-safe manner.
    
    Returns:
        str or None: The Gemini API key, or None if not set
    """
    with _api_keys_lock:
        return GEMINI_API_KEY
