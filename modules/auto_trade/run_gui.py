import sys
import warnings
from pathlib import Path

# Suppress pkg_resources deprecation warning from lightning_fabric
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning_fabric")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.main_window import AutoTradeDashboard


def main():
    # Step 1: Build Rust extensions before launching GUI
    print("\nStarting Auto-Trade System")
    print("=" * 60)

    try:
        from utils.rust_builder import auto_build_with_checks

        print("Checking and building Rust extensions...\n")
        build_success = auto_build_with_checks(verbose=False)

        if build_success:
            print("All Rust extensions ready!\n")
        else:
            print("Rust build incomplete. Using Python fallback.\n")

    except ImportError as e:
        print(f"Could not import rust_builder: {e}")
        print("   Skipping Rust builds, using Python fallback.\n")
    except Exception as e:
        print(f"Rust build failed: {e}")
        print("   Using Python fallback.\n")

    print("=" * 60)
    print("Launching GUI...\n")

    # Step 2: Launch the GUI
    app = AutoTradeDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
