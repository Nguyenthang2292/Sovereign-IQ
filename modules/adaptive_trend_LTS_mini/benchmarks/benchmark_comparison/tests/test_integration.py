"""Integration tests for benchmark comparison pipeline."""

import subprocess
import sys
from pathlib import Path

import pytest

# Constants for project structure
MODULE_BASE = "modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison"


@pytest.fixture
def mock_price_data():
    """Generate synthetic price data for testing."""
    import numpy as np
    import pandas as pd

    dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
    data = {}

    # Generate 5 fake symbols
    for i in range(5):
        symbol = f"BTC_USDT_{i}"
        # Random walk
        returns = np.random.normal(0, 0.01, 100)
        price_paths = 100 * np.exp(np.cumsum(returns))
        data[symbol] = pd.Series(price_paths, index=dates, name="close")

    return data


@pytest.fixture
def mock_runners(mock_price_data, monkeypatch):
    """Fixture to mock all benchmark runners and data fetching using monkeypatch."""
    import pandas as pd

    # Create mock results
    mock_signals = {
        k: {
            "Average_Signal": pd.Series(0.0, index=v.index),
            "Trend_Pulse": pd.Series(0.0, index=v.index),
        }
        for k, v in mock_price_data.items()
    }
    mock_result_tuple = (mock_signals, 0.1, 100.0)

    # Helper for mocking return values
    def mock_return_data(*args, **kwargs):
        return mock_price_data

    def mock_return_result(*args, **kwargs):
        return mock_result_tuple

    def mock_do_nothing(*args, **kwargs):
        pass

    # Patch SOURCE modules
    # 1. Patch data fetching
    monkeypatch.setattr(f"{MODULE_BASE}.data.fetch_symbols_data", mock_return_data)

    # 2. Patch build tools
    monkeypatch.setattr(f"{MODULE_BASE}.build.ensure_rust_extensions_built", mock_do_nothing)

    # 3. Patch runner functions in runners.py
    runner_targets = [
        "run_original_module",
        "run_rust_module",
        "run_rust_batch_module",
        "run_approximate_module",
        "run_adaptive_approximate_module",
        "run_dask_module",
        "run_rust_dask_module",
    ]

    for target in runner_targets:
        monkeypatch.setattr(f"{MODULE_BASE}.runners.{target}", mock_return_result)

    # Force reload of main module to pick up mocked functions
    sys.modules.pop(f"{MODULE_BASE}.main", None)


def test_benchmark_cli_help():
    """Test benchmark CLI help command."""
    # Get path to main.py
    main_path = Path(__file__).parent.parent / "main.py"

    result = subprocess.run(
        [sys.executable, str(main_path), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0
    assert "Benchmark" in result.stdout or "benchmark" in result.stdout
    assert "--symbols" in result.stdout
    assert "--bars" in result.stdout


def test_benchmark_cli_version_info():
    """Test that main.py can show help without errors."""
    main_path = Path(__file__).parent.parent / "main.py"

    # Check file exists
    assert main_path.exists(), f"main.py not found at {main_path}"

    # Check it's readable
    content = main_path.read_text(encoding="utf-8")
    assert "argparse" in content
    assert "main" in content


def test_full_benchmark_pipeline_mocked(mock_price_data, mock_runners, monkeypatch, tmp_path):
    """Integration test for full benchmark pipeline using mocked data.

    This uses the 'mock_runners' fixture to handle all dependency mocking.
    """
    # Import main AFTER mocks are set up and module is popped from sys.modules
    from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.main import main

    # Mock command line arguments
    test_args = [
        "main.py",
        "--symbols",
        "5",
        "--bars",
        "100",
        "--force",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the main function
    try:
        main()
    except SystemExit as e:
        assert e.code == 0, f"Benchmark pipeline failed with exit code {e.code}"
    except Exception as e:
        pytest.fail(f"Benchmark pipeline failed with unexpected error: {e}")

    # Verify output files were created
    results_dir = Path(__file__).parent.parent / "results"
    assert (results_dir / "benchmark_results.txt").exists()
    assert (results_dir / "benchmark_results.html").exists()


def test_module_imports():
    """Test that all benchmark modules can be imported."""
    try:
        from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison import (
            build,
            comparison,
            data,
            html_formatter,
            runners,
        )

        assert build is not None
        assert comparison is not None
        assert data is not None
        assert html_formatter is not None
        assert runners is not None
    except ImportError as e:
        pytest.fail(f"Failed to import benchmark modules: {e}")


def test_constants_defined():
    """Test that module constants are defined."""
    from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.comparison import (
        SIGNAL_MATCH_TOLERANCE,
    )
    from modules.adaptive_trend_LTS_mini.benchmarks.benchmark_comparison.main import (
        MAX_LOGS_TO_KEEP,
    )

    assert SIGNAL_MATCH_TOLERANCE == 1e-6
    assert MAX_LOGS_TO_KEEP == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
