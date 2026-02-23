import sys
import warnings
import os
from pathlib import Path


def _patch_windows_platform_for_customtkinter() -> None:
    """Avoid darkdetect WMI hangs on some Windows setups.

    darkdetect imports platform.release()/version() during customtkinter import.
    On certain systems this path can block in WMI queries. Provide fast values
    from sys.getwindowsversion() before importing GUI modules.
    """
    if sys.platform != "win32":
        return

    try:
        import platform

        win = sys.getwindowsversion()
        release_value = "10" if int(getattr(win, "major", 10)) >= 10 else str(int(getattr(win, "major", 10)))
        version_value = f"{int(getattr(win, 'major', 10))}.{int(getattr(win, 'minor', 0))}.{int(getattr(win, 'build', 0))}"

        # Prevent slow/hanging WMI calls inside platform.* on some Windows systems.
        # Several dependencies (darkdetect, aiohttp, boto3, numba) trigger these during import.
        if hasattr(platform, "_wmi_query"):
            platform._wmi_query = lambda *args, **kwargs: (  # type: ignore[attr-defined,assignment]
                version_value,
                "1",
                "Multiprocessor Free",
                "0",
                "0",
            )

        platform.release = lambda: release_value  # type: ignore[assignment]
        platform.version = lambda: version_value  # type: ignore[assignment]
        platform.system = lambda: "Windows"  # type: ignore[assignment]
        platform.machine = lambda: os.getenv("PROCESSOR_ARCHITECTURE", "AMD64")  # type: ignore[assignment]
    except Exception:
        pass

# Suppress pkg_resources deprecation warning from lightning_fabric
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning_fabric")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    module_env = project_root / ".env"
    if module_env.exists():
        load_dotenv(module_env)
    else:
        load_dotenv()
except ImportError:
    pass


def main() -> None:
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Trade System GUI")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Python bytecode cache before starting GUI",
    )
    parser.add_argument(
        "--no-rust-build",
        action="store_true",
        help="Skip Rust extension build check",
    )
    args = parser.parse_args()

    # Step 1: Clear cache if requested
    if args.clear_cache:
        print("\nClearing Python bytecode cache...")
        try:
            from utils.cache_cleaner import clear_module_cache

            clear_module_cache(verbose=False)
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")
            print("Continuing with GUI startup...\n")

    # Step 2: Build Rust extensions before launching GUI (unless skipped)
    print("\nStarting Auto-Trade System")
    print("=" * 60)

    if not args.no_rust_build:
        try:
            from utils.rust_builder import auto_build_with_checks

            print("Checking and building Rust extensions...\n")
            build_success = auto_build_with_checks(verbose=False, clear_cache=True)

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
    else:
        print("Skipping Rust build check (--no-rust-build)\n")

    print("=" * 60)
    print("Launching GUI...\n")

    # Step 3: Launch the GUI
    _patch_windows_platform_for_customtkinter()
    from modules.auto_trade.gui.main_window import AutoTradeDashboard

    app = AutoTradeDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
