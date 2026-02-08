#!/usr/bin/env python3
"""
Test verification script for new error recovery and CLI integration tests.

This script validates that all new tests are properly installed and can run successfully.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"\n✅ {description} - PASSED")
            return True
        else:
            print(f"\n❌ {description} - FAILED (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n⏱️  {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {e}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False


def main():
    """Main verification routine."""
    print("="*70)
    print("TEST COVERAGE ENHANCEMENT VERIFICATION")
    print("="*70)

    # Get the test directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent.parent

    print(f"\nTest Directory: {test_dir}")
    print(f"Project Root: {project_root}")

    # Step 1: Verify files exist
    print("\n" + "="*70)
    print("STEP 1: Verify New Test Files")
    print("="*70)

    files_ok = True
    files_ok &= check_file_exists(
        test_dir / "test_error_recovery.py",
        "Error Recovery Tests"
    )
    files_ok &= check_file_exists(
        test_dir / "test_cli_integration.py",
        "CLI Integration Tests"
    )
    files_ok &= check_file_exists(
        test_dir / "TEST_COVERAGE.md",
        "Coverage Documentation"
    )
    files_ok &= check_file_exists(
        test_dir / "TEST_ENHANCEMENT_SUMMARY.md",
        "Enhancement Summary"
    )

    if not files_ok:
        print("\n❌ Some files are missing. Cannot proceed.")
        return 1

    print("\n✅ All required files present")

    # Step 2: Check pytest installation
    print("\n" + "="*70)
    print("STEP 2: Verify pytest Installation")
    print("="*70)

    pytest_check = run_command(
        ["pytest", "--version"],
        "pytest version check"
    )

    if not pytest_check:
        print("\n❌ pytest not installed. Install with: pip install pytest")
        return 1

    # Step 3: Run error recovery tests
    print("\n" + "="*70)
    print("STEP 3: Run Error Recovery Tests")
    print("="*70)

    error_recovery_ok = run_command(
        ["pytest", str(test_dir / "test_error_recovery.py"), "-v", "--tb=short"],
        "Error Recovery Tests"
    )

    # Step 4: Run CLI integration tests
    print("\n" + "="*70)
    print("STEP 4: Run CLI Integration Tests")
    print("="*70)

    cli_integration_ok = run_command(
        ["pytest", str(test_dir / "test_cli_integration.py"), "-v", "--tb=short"],
        "CLI Integration Tests"
    )

    # Step 5: Run all new tests together
    print("\n" + "="*70)
    print("STEP 5: Run All New Tests Together")
    print("="*70)

    all_new_tests_ok = run_command(
        [
            "pytest",
            str(test_dir / "test_error_recovery.py"),
            str(test_dir / "test_cli_integration.py"),
            "-v",
            "--tb=line"
        ],
        "All New Tests"
    )

    # Step 6: Count test functions
    print("\n" + "="*70)
    print("STEP 6: Count Test Functions")
    print("="*70)

    count_ok = run_command(
        [
            "pytest",
            str(test_dir / "test_error_recovery.py"),
            str(test_dir / "test_cli_integration.py"),
            "--collect-only",
            "-q"
        ],
        "Test Collection Count"
    )

    # Step 7: Run with coverage (optional)
    print("\n" + "="*70)
    print("STEP 7: Generate Coverage Report (Optional)")
    print("="*70)

    try:
        import pytest_cov
        coverage_ok = run_command(
            [
                "pytest",
                str(test_dir / "test_error_recovery.py"),
                str(test_dir / "test_cli_integration.py"),
                "--cov=modules.adaptive_trend_LTS_mini",
                "--cov-report=term-missing",
                "--cov-report=html"
            ],
            "Coverage Report Generation"
        )
    except ImportError:
        print("ℹ️  pytest-cov not installed. Skipping coverage report.")
        print("   Install with: pip install pytest-cov")
        coverage_ok = True  # Don't fail if coverage not available

    # Final Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    results = {
        "Files Present": files_ok,
        "pytest Installed": pytest_check,
        "Error Recovery Tests": error_recovery_ok,
        "CLI Integration Tests": cli_integration_ok,
        "All New Tests": all_new_tests_ok,
        "Test Collection": count_ok,
        "Coverage Report": coverage_ok,
    }

    for check, status in results.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("="*70)
        print("\nThe new test coverage is successfully installed and working.")
        print("\nNext steps:")
        print("  1. Review test output above for any warnings")
        print("  2. Check coverage report in htmlcov/index.html (if generated)")
        print("  3. Add tests to CI/CD pipeline")
        print("  4. See TEST_COVERAGE.md for detailed documentation")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("="*70)
        print("\nPlease review the failed checks above and resolve issues.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements-dev.txt")
        print("  - Import errors: Check PYTHONPATH or install package")
        print("  - Mock failures: Verify unittest.mock usage")
        return 1


if __name__ == "__main__":
    sys.exit(main())
