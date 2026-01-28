import subprocess
import sys

# Run the specific tests with venv Python
result = subprocess.run(
    [
        "C:\\Users\\Admin\\Desktop\\i-ching\\crypto-probability\\venv\\Scripts\\python.exe",
        "-m",
        "pytest",
        "tests/adaptive_trend_LTS/test_rust_dask_bridge.py::test_process_partition_python_with_none",
        "tests/adaptive_trend_LTS/test_rust_dask_bridge.py::test_process_partition_with_rust_cuda_empty",
        "tests/adaptive_trend_LTS/test_rust_dask_bridge.py::test_process_symbols_rust_dask_error_handling",
        "-v",
    ],
    cwd="C:\\Users\\Admin\\Desktop\\i-ching\\crypto-probability",
    capture_output=True,
    text=True,
    timeout=180,
    shell=True,
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")
