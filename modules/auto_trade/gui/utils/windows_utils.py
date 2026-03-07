import sys
import ctypes
import platform

def apply_dark_titlebar(window):
    """
    Apply dark title bar to a Tkinter/CustomTkinter window on Windows 10/11.
    """
    if sys.platform != "win32":
        return

    try:
        window.update() # ensure hwnd is available
        
        # Win 11 needs 20, Win 10 (later builds) needs 19
        build = int(platform.version().split('.')[2])
        if build >= 22000:
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        else:
            DWMWA_USE_IMMERSIVE_DARK_MODE = 19
            
        set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        get_parent = ctypes.windll.user32.GetParent
        hwnd = get_parent(window.winfo_id())
        
        # If GetParent returns 0, use winfo_id directly
        if hwnd == 0:
            hwnd = window.winfo_id()
            
        value = ctypes.c_int(2)  # 2 = dark mode, 1 = light mode, 0 = default
        set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
    except Exception as e:
        print(f"Could not set dark titlebar: {e}")
