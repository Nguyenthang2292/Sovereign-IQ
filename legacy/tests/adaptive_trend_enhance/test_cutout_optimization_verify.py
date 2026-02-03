import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_enhance.core.compute_atc_signals.compute_atc_signals import compute_atc_signals


def test_cutout_slicing_consistency():
    """Verify that cutout parameter correctly slices the data and results are consistent."""
    # Create dummy price data (reproducible)
    n_bars = 500
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(n_bars)), name="Close")

    cutout_val = 50

    # 1. Run with cutout=0
    res_full = compute_atc_signals(prices, cutout=0)

    # 2. Run with cutout=cutout_val
    res_cutout = compute_atc_signals(prices, cutout=cutout_val)

    # 3. Verify lengths
    expected_len = n_bars - cutout_val
    for key, series in res_cutout.items():
        assert len(series) == expected_len, f"Length mismatch for {key}: {len(series)} != {expected_len}"

    # 4. Verify Content Consistency
    # Note: Layer 1 signals should match after a few bars due to crossover logic needing 1-bar history.
    # We skip the first 2 bars of the cutout result to compare with the full run.
    for key in res_cutout.keys():
        if "_Signal" in key or key == "Average_Signal":
            # Signals should be very similar after a brief stabilization
            # Skip first 2 bars of cutout result (indices 0, 1) and compare with corresponding part of full run
            full_slice = res_full[key].iloc[cutout_val + 2 :].reset_index(drop=True)
            cutout_slice = res_cutout[key].iloc[2:].reset_index(drop=True)

            # Use a slightly more relaxed tolerance for Average_Signal as it's weighted by Equity
            # which starts at 1.0 in both cases but evolved differently in res_full.
            atol = 0.05 if key == "Average_Signal" else 1e-7
            np.testing.assert_allclose(
                full_slice.values, cutout_slice.values, atol=atol, err_msg=f"Signal mismatch for {key}"
            )

    # 5. Verify NaN elimination
    # The output should have ABSOLUTELY NO NaNs if cutout is large enough
    # (MA length is 28, cutout is 50, so stabilization happened before slice)
    for key, series in res_cutout.items():
        nan_count = series.isna().sum()
        assert nan_count == 0, f"{key} contains {nan_count} NaN values after cutout optimization"

    print("Verification successful: No NaNs, correct lengths, and signals mostly consistent.")


if __name__ == "__main__":
    pytest.main([__file__])
