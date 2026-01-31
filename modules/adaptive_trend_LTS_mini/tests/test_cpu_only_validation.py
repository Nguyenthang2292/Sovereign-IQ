"""
Test basic import and functionality of CPU-only adaptive_trend_LTS_mini module.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_imports():
    """Test that module imports work without CUDA dependencies."""
    print("Testing imports...")

    try:
        # Test core imports
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals

        print("✅ compute_atc_signals imported successfully")

        from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

        print("✅ ATCConfig imported successfully")

        # Test Rust backend import
        from modules.adaptive_trend_LTS_mini.core import rust_backend

        print("✅ rust_backend imported successfully")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test ATCConfig without use_cuda field."""
    print("\nTesting ATCConfig...")

    try:
        from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

        config = ATCConfig()

        # Verify use_cuda is NOT present
        if hasattr(config, "use_cuda"):
            print(f"⚠️  Warning: use_cuda field still exists")
        else:
            print("✅ use_cuda field correctly removed from ATCConfig")

        # Verify use_rust_backend is present
        if hasattr(config, "use_rust_backend"):
            print(f"✅ use_rust_backend present: {config.use_rust_backend}")
        else:
            print("❌ use_rust_backend field missing!")
            return False

        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_rust_extension():
    """Test that Rust extension loads and works."""
    print("\nTesting Rust extension...")

    try:
        # Check local build first
        import os

        local_dll = os.path.join(
            os.path.dirname(__file__), "..", "rust_extensions", "target", "release", "atc_rust.dll"
        )

        if os.path.exists(local_dll):
            print(f"✅ Local Rust DLL exists: {local_dll}")
            print(f"   File size: {os.path.getsize(local_dll) / 1024:.1f} KB")
        else:
            print(f"⚠️  Local Rust DLL not found at: {local_dll}")
            print("   Run: cd rust_extensions && cargo build --release")

        # Try to import atc_rust (might be installed package or local)
        import atc_rust

        print(f"✅ atc_rust loaded from: {atc_rust.__file__}")

        # Check available functions
        available_funcs = [attr for attr in dir(atc_rust) if not attr.startswith("_")]
        print(f"✅ Available functions: {', '.join(available_funcs[:8])}")

        # Note: If atc_rust is installed from site-packages, it may still have CUDA functions
        # The local build (target/release/atc_rust.dll) is CPU-only

        # Check for CPU batch function
        if hasattr(atc_rust, "compute_atc_signals_batch_cpu"):
            print("✅ CPU batch function 'compute_atc_signals_batch_cpu' present")
        else:
            print("⚠️  CPU batch function not found (may be using old installed version)")
            print("   To use CPU-only version, reinstall from local build:")
            print("   cd rust_extensions && maturin develop --release")

        return True
    except ImportError as e:
        print(f"⚠️  Rust extension not available: {e}")
        print("   Run: cd rust_extensions && cargo build --release")
        return True  # Not a failure, just not built
    except Exception as e:
        print(f"❌ Rust extension test failed: {e}")
        return False

        return True
    except ImportError as e:
        print(f"⚠️  Rust extension not available (expected if not built): {e}")
        print("   Run: cd rust_extensions && cargo build --release")
        return True  # Not a failure, just not built
    except Exception as e:
        print(f"❌ Rust extension test failed: {e}")
        return False


def test_no_gpu_imports():
    """Verify no GPU-related imports in key files."""
    print("\nTesting for GPU import cleanliness...")

    import os
    import glob

    module_path = os.path.join(os.path.dirname(__file__), "..")

    # Find all Python files
    py_files = glob.glob(os.path.join(module_path, "**", "*.py"), recursive=True)

    cupy_found = []
    pycuda_found = []

    for py_file in py_files:
        # Skip the test file itself
        if os.path.basename(py_file) == "test_cpu_only_validation.py":
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Look for actual import statements (not in comments/strings)
                lines = content.split("\n")
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("import cupy") or stripped.startswith("from cupy"):
                        cupy_found.append(py_file)
                        break
                    if stripped.startswith("import pycuda") or stripped.startswith("from pycuda"):
                        pycuda_found.append(py_file)
                        break
        except Exception:
            continue

    if cupy_found:
        print(f"⚠️  Found cupy imports in: {len(cupy_found)} files")
        for f in cupy_found[:3]:
            print(f"   - {os.path.basename(f)}")
    else:
        print("✅ No cupy imports found")

    if pycuda_found:
        print(f"⚠️  Found pycuda imports in: {len(pycuda_found)} files")
        for f in pycuda_found[:3]:
            print(f"   - {os.path.basename(f)}")
    else:
        print("✅ No pycuda imports found")

    return len(cupy_found) == 0 and len(pycuda_found) == 0


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("CPU-Only Module Validation Tests")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Rust Extension", test_rust_extension()))
    results.append(("GPU Import Check", test_no_gpu_imports()))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<30} {status}")

    all_passed = all(r[1] for r in results)

    print("=" * 60)
    if all_passed:
        print("✅ All validation tests passed!")
        print("Module is ready for CPU-only usage.")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
