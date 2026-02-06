from typing import Optional

import numpy as np
import pandas as pd

from modules.common.utils import log_error, log_warn

"""Calculate exponential growth factor over time."""


def exp_growth(
    lambda_val: float,
    index: Optional[pd.Index] = None,
    *,
    cutout: int = 0,
) -> pd.Series:
    """Calculate exponential growth factor over time.

    Port of Pine Script function:
        e(lambda_val) =>
            bars = bar_index == 0 ? 1 : bar_index
            x = 1.0
            if time >= cuttime
                x := math.pow(math.e, lambda_val * (bar_index - cutout))
            x

    In TradingView, `time` and `bar_index` are global environment variables.
    Here we approximate using positional indices (0, 1, 2, ...) of the Series.

    Args:
        lambda_val: Lambda (growth rate parameter, must be finite).
        index: Time/bar index of the data. If None, creates empty RangeIndex.
        cutout: Number of bars to skip at the beginning (bars before cutout
            will have value 1.0, must be >= 0).

    Returns:
        Series containing exponential growth factors e^(lambda_val * (bar_index - cutout))
        for bars >= cutout, and 1.0 for bars < cutout.

    Raises:
        ValueError: If lambda_val is invalid, cutout is invalid, or overflow occurs.
        TypeError: If lambda_val is not a number or cutout is not an integer.
    """
    if not isinstance(lambda_val, (int, float)) or np.isnan(lambda_val) or np.isinf(lambda_val):
        raise ValueError(f"lambda_val must be a finite number, got {lambda_val}")

    # Validate lambda_val is within safe range to prevent overflow
    # For typical use cases with bar counts up to 10000, lambda_val should be in [-1.0, 1.0]
    # to avoid exp(lambda_val * bars) overflow (exp(700) is max for float64)
    SAFE_LAMBDA_RANGE = 1.0
    if abs(lambda_val) > SAFE_LAMBDA_RANGE:
        log_warn(
            f"lambda_val parameter ({lambda_val}) is outside safe range [-{SAFE_LAMBDA_RANGE}, {SAFE_LAMBDA_RANGE}]. "
            f"This may cause overflow in exponential calculations. Proceeding with caution."
        )

    if not isinstance(cutout, int) or cutout < 0:
        raise ValueError(f"cutout must be a non-negative integer, got {cutout}")

    if index is None:
        index = pd.RangeIndex(0, 0)

    try:
        # Use position 0..n-1 as equivalent to `bar_index`
        bars = pd.Series(range(len(index)), index=index, dtype="float64")
        # In Pine: if bar_index == 0 then bars = 1, else = bar_index
        bars = bars.where(bars != 0, 1.0)

        # Condition "has passed cutout"
        active = bars >= cutout
        x = pd.Series(1.0, index=index, dtype="float64")

        # Calculate exponential growth for active bars
        if active.any():
            # Calculate exponent to check for overflow
            exponents = lambda_val * (bars[active] - cutout)

            # Check for potential overflow (exp > 700 will overflow float64)
            max_exponent = exponents.max() if len(exponents) > 0 else 0
            if max_exponent > 700:
                log_warn(
                    f"Potential overflow in exp_growth: max exponent = {max_exponent:.2f}. "
                    f"Values > 700 may result in inf. lambda_val={lambda_val}, max_bar={bars[active].max()}, cutout={cutout}"
                )

            # Calculate exponential growth (use np.exp to avoid type-checker confusion with exception var 'e')
            growth_values = np.exp(exponents)

            # Check for overflow/inf values
            inf_count = np.isinf(growth_values).sum()
            if inf_count > 0:
                log_warn(
                    f"exp_growth produced {inf_count} inf values. "
                    f"This may indicate overflow. Consider reducing lambda_val or cutout."
                )
                # Replace inf with a large but finite value
                growth_values = np.where(np.isinf(growth_values), np.finfo(np.float64).max, growth_values)

            x.loc[active] = growth_values.astype("float64")  # type: ignore[assignment]

        return x

    except OverflowError as e:
        log_error(f"Overflow error in exp_growth: {e}. lambda_val={lambda_val}, cutout={cutout}")
        raise ValueError(f"Overflow in exponential calculation. lambda_val={lambda_val} may be too large.") from e
    except Exception as e:
        log_error(f"Error calculating exp_growth: {e}")
        raise
