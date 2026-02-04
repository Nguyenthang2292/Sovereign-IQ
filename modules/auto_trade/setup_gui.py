"""
Quick Setup Script for Auto Trade GUI
Tạo cấu trúc folder và file templates cơ bản

Run: python setup_gui.py
"""

from pathlib import Path


def create_directories():
    """Tạo cấu trúc thư mục"""
    base_path = Path(__file__).parent / "gui"

    dirs = [
        base_path,
        base_path / "components",
        base_path / "utils",
        base_path / "assets",
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")


def create_init_files():
    """Tạo __init__.py files"""
    base_path = Path(__file__).parent / "gui"

    init_files = [
        base_path / "__init__.py",
        base_path / "components" / "__init__.py",
        base_path / "utils" / "__init__.py",
    ]

    for init_file in init_files:
        init_file.touch(exist_ok=True)
        print(f"✅ Created: {init_file}")


def create_main_window():
    """Tạo main window template"""
    content = '''"""
Auto Trade Dashboard - Main Window
"""
import customtkinter as ctk
from typing import Optional


class AutoTradeDashboard(ctk.CTk):
    """Main GUI application window"""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("🚀 Auto Trade Dashboard")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create layout
        self._create_layout()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_layout(self):
        """Create main UI layout"""
        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        self._create_header()

        # Content area
        self._create_content()

        # Status bar
        self._create_statusbar()

    def _create_header(self):
        """Create header with title and mode indicator"""
        header_frame = ctk.CTkFrame(self, height=60, fg_color="#1e1e1e")
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)

        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🚀 Auto Trade Dashboard",
            font=("Arial", 20, "bold")
        )
        title_label.pack(side="left", padx=20)

        # Mode indicator
        mode_label = ctk.CTkLabel(
            header_frame,
            text="🔴 PRODUCTION",
            font=("Arial", 12, "bold"),
            text_color="red"
        )
        mode_label.pack(side="right", padx=20)

    def _create_content(self):
        """Create main content area"""
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
                # Configure grid for 2 columns
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)

        # Left panel (Account + Stats)
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        # Right panel (Signals + Positions)
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        # Placeholder labels
        ctk.CTkLabel(
            left_panel,
            text="💰 Account Overview\\n\\n(Coming soon)",
            font=("Arial", 14)
        ).pack(pady=50)

        ctk.CTkLabel(
            right_panel,
            text="🎯 Live Signals\\n\\n(Coming soon)",
            font=("Arial", 14)
        ).pack(pady=50)

    def _create_statusbar(self):
        """Create status bar at bottom"""
        statusbar = ctk.CTkFrame(self, height=30, fg_color="#1e1e1e")
        statusbar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        statusbar.grid_propagate(False)

        status_label = ctk.CTkLabel(
            statusbar,
            text="Ready",
            font=("Arial", 10),
            text_color="gray"
        )
        status_label.pack(side="left", padx=10)

    def on_closing(self):
        """Handle window closing"""
        print("Closing application...")
        self.destroy()


def main():
    """Run the application"""
    app = AutoTradeDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
'''

    file_path = Path(__file__).parent / "gui" / "main_window.py"
    file_path.write_text(content, encoding="utf-8")
    print(f"✅ Created: {file_path}")


def create_run_script():
    """Tạo run_gui.py entry point"""
    content = '''"""
Auto Trade GUI Dashboard Entry Point

Run with: python run_gui.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from modules.auto_trade.gui.main_window import AutoTradeDashboard
except ImportError as e:
    print(f"❌ Error importing: {e}")
    print("\\n📦 Did you install customtkinter?")
    print("   Run: pip install customtkinter")
    sys.exit(1)


def main():
    """Launch GUI application"""
    print("🚀 Starting Auto Trade Dashboard...")
    
    try:
        app = AutoTradeDashboard()
        app.mainloop()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
'''

    file_path = Path(__file__).parent / "run_gui.py"
    file_path.write_text(content, encoding="utf-8")
    print(f"✅ Created: {file_path}")


def create_requirements():
    """Tạo requirements_gui.txt"""
    content = """# Auto Trade GUI Requirements
# Install with: pip install -r requirements_gui.txt

customtkinter>=5.0.0
pillow>=10.0.0
matplotlib>=3.7.0
pandas>=2.0.0
plyer>=2.1.0
"""

    file_path = Path(__file__).parent / "requirements_gui.txt"
    file_path.write_text(content, encoding="utf-8")
    print(f"✅ Created: {file_path}")


def main():
    """Run all setup steps"""
    print("\n🎨 Setting up Auto Trade GUI...\n")

    create_directories()
    create_init_files()
    create_main_window()
    create_run_script()
    create_requirements()

    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("   1. pip install -r requirements_gui.txt")
    print("   2. python run_gui.py")
    print("\n📖 See phase1_python_gui_tasks.md for detailed implementation plan")


if __name__ == "__main__":
    main()
