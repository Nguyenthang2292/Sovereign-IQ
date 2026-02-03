import customtkinter as ctk
import time


class ToastNotification(ctk.CTkToplevel):
    def __init__(self, parent, message, duration=3000, fg_color="#333333", text_color="white"):
        super().__init__(parent)
        self.overrideredirect(True)

        # Calculate position (center bottom of parent)
        try:
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()

            width = 300
            height = 50
            x = parent_x + (parent_width - width) // 2
            y = parent_y + parent_height - height - 50

            self.geometry(f"{width}x{height}+{x}+{y}")
        except:
            # Fallback if parent geometry fails
            self.geometry("300x50")

        # Style
        self.configure(fg_color=fg_color)

        # Content
        self.label = ctk.CTkLabel(
            self, text=message, font=("Arial", 14, "bold"), text_color=text_color, fg_color="transparent"
        )
        self.label.pack(expand=True, fill="both", padx=20, pady=10)

        # Settings
        self.attributes("-alpha", 0.9)
        self.attributes("-topmost", True)

        # Auto close
        self.after(duration, self._fade_out)

        # Click to dismiss
        self.bind("<Button-1>", lambda e: self.destroy())
        self.label.bind("<Button-1>", lambda e: self.destroy())

    def _fade_out(self):
        """Simple fade out effect"""
        alpha = self.attributes("-alpha")
        if alpha > 0:
            alpha -= 0.1
            self.attributes("-alpha", alpha)
            self.after(50, self._fade_out)
        else:
            self.destroy()


def show_toast(parent, message, type="info", duration=3000):
    """
    Helper to show toast
    type: "info", "success", "error", "warning"
    """
    colors = {"info": "#333333", "success": "#228822", "error": "#aa2222", "warning": "#aa8822"}
    color = colors.get(type, "#333333")
    ToastNotification(parent, message, duration, fg_color=color)
