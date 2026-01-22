import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# Mock cupy before importing gpu_scan if possible, or patch it after
# Since gpu_scan imports cupy at top level try-except, we can simulate success/failure via patching modules.


@pytest.fixture
def mock_cupy():
    with patch.dict(sys.modules, {"cupy": MagicMock()}):
        yield


def test_scan_gpu_batch_pipeline_logic():
    """Verify that _scan_gpu_batch correctly iterates batches and aggregates results."""

    # We need to import the module under test
    # But it might fail to import cupy.
    # We will patch _HAS_CUPY and the GPU functions regardless of actual import success

    with (
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan._HAS_CUPY", True),
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan.cp", create=True) as mock_cp,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan.rate_of_change_gpu", create=True) as mock_roc,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan.calculate_batch_ema_gpu", create=True) as mock_ema,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan._calculate_hma_gpu", create=True) as mock_hma,
        patch(
            "modules.adaptive_trend_enhance.core.scanner.gpu_scan._calculate_wma_gpu_optimized", create=True
        ) as mock_wma,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan._calculate_dema_gpu", create=True) as mock_dema,
        patch(
            "modules.adaptive_trend_enhance.core.scanner.gpu_scan._calculate_lsma_gpu_optimized", create=True
        ) as mock_lsma,
        patch(
            "modules.adaptive_trend_enhance.core.scanner.gpu_scan.generate_signal_from_ma_gpu", create=True
        ) as mock_gen_sig,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan.calculate_equity_gpu", create=True) as mock_eq,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan.trend_sign_gpu", create=True) as mock_trend,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan._fetch_batch_data", create=True) as mock_fetch,
        patch("modules.adaptive_trend_enhance.core.scanner.gpu_scan.diflen", create=True) as mock_diflen,
    ):
        from modules.adaptive_trend_enhance.core.scanner.gpu_scan import _scan_gpu_batch
        from modules.adaptive_trend_enhance.utils.config import ATCConfig

        # Setup Inputs
        symbols = [f"SYM{i}" for i in range(10)]
        data_fetcher = MagicMock()
        atc_config = ATCConfig(limit=100)

        # Mock fetch returns: (batch_data, meta, skip, err, skip_syms)
        # We will split into 2 batches of 5
        batch1_data = [np.zeros(100) for _ in range(5)]
        batch1_meta = [(f"SYM{i}", 100.0, "binance") for i in range(5)]

        batch2_data = [np.zeros(100) for _ in range(5)]
        batch2_meta = [(f"SYM{i}", 200.0, "binance") for i in range(5, 10)]

        # Side effect for fetch: return batch1 then batch2
        mock_fetch.side_effect = [(batch1_data, batch1_meta, 0, 0, []), (batch2_data, batch2_meta, 0, 0, [])]

        # Mock GPU processing to return usable shapes
        # For simplicity, we just need mocks to not crash and return mocks that support indexing/slicing

        # mock_cp.asarray returns a MagicMock
        # mock_trend returns a mock. We need `[:, -1].get()` to work.
        # So trend_gpu return value must be subscriptable.

        mock_gpu_array = MagicMock()
        mock_gpu_array.__getitem__.return_value = mock_gpu_array  # Slicing returns self
        mock_gpu_array.get.return_value = np.ones(5)  # .get() returns numpy array of ones (signal/trend)

        mock_trend.return_value = mock_gpu_array
        mock_gen_sig.return_value = mock_gpu_array
        mock_eq.return_value = mock_gpu_array

        # Also final_signal_gpu assignment
        # cp.zeros_like -> mock
        mock_cp.zeros_like.return_value = mock_gpu_array

        # Run
        results, skipped, errors, skipped_syms = _scan_gpu_batch(
            symbols, data_fetcher, atc_config, min_signal=0.5, batch_size=5
        )

        # Verification
        assert len(results) == 10  # All 10 symbols should have results (since we mocked sig=1.0 > 0.5)
        assert mock_fetch.call_count == 2

        # Verify batching calls
        # We can't strict verify concurrency order easily here without sleeps,
        # but we verify that all batches were fetched and processed.


if __name__ == "__main__":
    test_scan_gpu_batch_pipeline_logic()
