"""Momentum indicator block."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from modules.common.utils import validate_ohlcv_input

from .base import IndicatorResult, collect_metadata

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """Momentum indicators: RSI, MACD, BBands, StochRSI, KAMA."""

    CATEGORY = "momentum"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        """
        Apply all momentum indicators to the input DataFrame.

        Args:
            df: DataFrame with OHLCV data, must contain 'close' column

        Returns:
            Tuple of (result DataFrame with indicators, metadata)
        """
        # Validate input
        validate_ohlcv_input(df, required_columns=["close"])

        result = df.copy()
        before = result.columns.tolist()

        # RSI indicators
        def _fill_rsi(length: int) -> pd.Series:
            rsi = ta.rsi(result["close"], length=length)
            return rsi.fillna(50.0) if rsi is not None else pd.Series(50.0, index=result.index)

        result["RSI_9"] = _fill_rsi(9)
        result["RSI_14"] = _fill_rsi(14)
        result["RSI_25"] = _fill_rsi(25)

        # MACD indicator
        macd = ta.macd(result["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd = macd.copy()
            for col in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
                if col in macd.columns:
                    macd[col] = macd[col].fillna(0.0)
            result[macd.columns] = macd
        else:
            logger.warning("MACD calculation failed, using neutral values (0).")
            result["MACD_12_26_9"] = 0.0
            result["MACDh_12_26_9"] = 0.0
            result["MACDs_12_26_9"] = 0.0

        # Bollinger Bands (BBP - Bollinger Band Percentage)
        bbands = ta.bbands(result["close"], length=20, std=2.0)
        if bbands is not None and not bbands.empty:
            bbp_cols = [col for col in bbands.columns if col.startswith("BBP")]
            if bbp_cols:
                # Sử dụng giá trị từ cột thực tế của pandas_ta nhưng giữ tên BBP_5_2.0 để tương thích ngược
                result["BBP_5_2.0"] = bbands[bbp_cols[0]].fillna(0.5)
            else:
                logger.warning("BBP column missing in Bollinger Bands output, falling back to 0.5.")
                result["BBP_5_2.0"] = 0.5
        else:
            logger.warning("Bollinger Bands calculation failed, defaulting BBP to 0.5.")
            result["BBP_5_2.0"] = 0.5

        # Stochastic RSI
        stochrsi_df = calculate_stochrsi_series(result["close"], length=14, rsi_length=14, k=3, d=3)

        if stochrsi_df is None or stochrsi_df.empty:
            logger.warning("Stochastic RSI failed, using neutral values (50).")
            result["STOCHRSIk_14_14_3_3"] = 50.0
            result["STOCHRSId_14_14_3_3"] = 50.0
        else:
            result[stochrsi_df.columns] = stochrsi_df

        # KAMA indicator
        kama_series = calculate_kama_series(result["close"], period=10)

        if kama_series is not None:
            result["KAMA_10"] = kama_series

        # Collect metadata and return
        metadata = collect_metadata(
            before,
            result.columns,
            MomentumIndicators.CATEGORY,
        )
        return result, metadata


# ============================================================================
# Helper Functions: RSI
# ============================================================================


def calculate_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) with custom period.

    Args:
        close: Close price series
        period: Period for RSI calculation (default: 14)

    Returns:
        Series with RSI values (0-100). NaN values are filled with 50.0 (neutral).
    """
    rsi = ta.rsi(close, length=period)
    if rsi is None:
        return pd.Series(50.0, index=close.index)
    # Normalize undefined RSI values to 50.0 (neutral)
    rsi = rsi.fillna(50.0)
    return rsi


# ============================================================================
# Helper Functions: MACD
# ============================================================================


def calculate_macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) with custom parameters.

    Args:
        close: Close price series
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        DataFrame with columns: MACD, MACD_signal, MACD_hist.
        Returns default values (0.0) if calculation fails.
    """
    macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
    if macd_df is None or macd_df.empty:
        # Return default values
        return pd.DataFrame({"MACD": 0.0, "MACD_signal": 0.0, "MACD_hist": 0.0}, index=close.index)

    # Map pandas_ta column names to our expected names
    # pandas_ta returns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    result = pd.DataFrame(index=close.index)

    # Find the MACD line column (format: MACD_fast_slow_signal)
    # More explicit: match exact pattern MACD_fast_slow_signal
    macd_col = [
        col
        for col in macd_df.columns
        if col.startswith("MACD_") and "h" not in col.split("_")[-1] and "s" not in col.split("_")[-1][0].lower()
    ]

    # Signal column (format: MACDs_fast_slow_signal)
    signal_col = [col for col in macd_df.columns if col.startswith("MACDs_")]

    # Histogram column (format: MACDh_fast_slow_signal)
    hist_col = [col for col in macd_df.columns if col.startswith("MACDh_")]

    if macd_col:
        result["MACD"] = macd_df[macd_col[0]].fillna(0.0)
    else:
        result["MACD"] = 0.0

    if signal_col:
        result["MACD_signal"] = macd_df[signal_col[0]].fillna(0.0)
    else:
        result["MACD_signal"] = 0.0

    if hist_col:
        result["MACD_hist"] = macd_df[hist_col[0]].fillna(0.0)
    else:
        # Calculate histogram as difference if not available
        result["MACD_hist"] = result["MACD"] - result["MACD_signal"]

    return result


# ============================================================================
# Helper Functions: Bollinger Bands
# ============================================================================


def calculate_bollinger_bands_series(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands with custom parameters.

    Args:
        close: Close price series
        period: Period for Bollinger Bands calculation (default: 20)
        std: Standard deviation multiplier (default: 2.0)

    Returns:
        DataFrame with columns: BB_upper, BB_middle, BB_lower.
        Returns default values based on SMA if calculation fails.
    """
    bbands = ta.bbands(close, length=period, std=std)
    if bbands is None or bbands.empty:
        # Return default values based on SMA
        sma = ta.sma(close, length=period)
        if sma is None:
            sma = close.rolling(window=period).mean()
        return pd.DataFrame({"BB_upper": sma, "BB_middle": sma, "BB_lower": sma}, index=close.index)

    # Map pandas_ta column names to our expected names
    # pandas_ta returns: BBU_length_std, BBM_length_std, BBL_length_std, BBP_length_std
    result = pd.DataFrame(index=close.index)

    # Find columns (format: BBU_period_std, BBM_period_std, BBL_period_std)
    upper_col = [col for col in bbands.columns if col.startswith("BBU_")]
    middle_col = [col for col in bbands.columns if col.startswith("BBM_")]
    lower_col = [col for col in bbands.columns if col.startswith("BBL_")]

    # Fallback: calculate manually if columns not found
    sma = close.rolling(window=period).mean()
    std_val = close.rolling(window=period).std()

    if upper_col:
        result["BB_upper"] = bbands[upper_col[0]].fillna(sma + (std_val * std))
    else:
        result["BB_upper"] = sma + (std_val * std)

    if middle_col:
        result["BB_middle"] = bbands[middle_col[0]].fillna(sma)
    else:
        result["BB_middle"] = sma

    if lower_col:
        result["BB_lower"] = bbands[lower_col[0]].fillna(sma - (std_val * std))
    else:
        result["BB_lower"] = sma - (std_val * std)

    return result


# ============================================================================
# Helper Functions: Stochastic RSI
# ============================================================================


def calculate_stochrsi_series(
    close: pd.Series, length: int = 14, rsi_length: int = 14, k: int = 3, d: int = 3
) -> pd.DataFrame:
    """
    Calculate Stochastic RSI series.

    Args:
        close: Close price series
        length: Length for Stochastic RSI calculation (default: 14)
        rsi_length: RSI length (default: 14)
        k: K period for smoothing (default: 3)
        d: D period for smoothing (default: 3)

    Returns:
        DataFrame with columns: STOCHRSIk_length_rsi_length_k_d, STOCHRSId_length_rsi_length_k_d.
        Returns default values (50.0) if calculation fails.
    """
    stochrsi = ta.stochrsi(close, length=length, rsi_length=rsi_length, k=k, d=d)
    if stochrsi is not None and not stochrsi.empty:
        # Fill NaN values with 50.0 (neutral)
        result = stochrsi.fillna(50.0)
        return result

    else:
        # Return default DataFrame with neutral values
        default_cols = [f"STOCHRSIk_{length}_{rsi_length}_{k}_{d}", f"STOCHRSId_{length}_{rsi_length}_{k}_{d}"]
        return pd.DataFrame({col: 50.0 for col in default_cols}, index=close.index)


# ============================================================================
# Helper Functions: KAMA (Kaufman Adaptive Moving Average)
# ============================================================================


def calculate_kama(
    prices: Union[np.ndarray, pd.Series, list], window: int = 10, fast: int = 2, slow: int = 30
) -> np.ndarray:
    """
    Calculate KAMA (Kaufman Adaptive Moving Average) using numpy.

    Args:
        prices: Price data (array-like or Series)
        window: Period for efficiency ratio calculation (default: 10)
        fast: Fast smoothing constant period (default: 2)
        slow: Slow smoothing constant period (default: 30)

    Returns:
        NumPy array with KAMA values
    """
    prices_array = np.asarray(prices, dtype=np.float64)

    if len(prices_array) < window:
        return np.full_like(prices_array, float(prices_array.flat[0])) if len(prices_array) > 0 else np.array([0.0])

    kama = np.zeros_like(prices_array, dtype=np.float64)
    first_valid_idx = next((i for i, price in enumerate(prices_array) if np.isfinite(price)), 0)
    initial_value = (
        float(prices_array[first_valid_idx])
        if first_valid_idx < len(prices_array)
        else float(np.nanmean(prices_array[:window]))
    )
    kama[:window] = initial_value

    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)

    try:
        price_series = prices if isinstance(prices, pd.Series) else pd.Series(prices)
        changes = price_series.diff(window).abs()
        volatility = (
            price_series.rolling(window)
            .apply(
                lambda values: (np.sum(np.abs(np.diff(values))) if len(values) > 1 else 1e-10),
                raw=False,
            )
            .fillna(1e-10)
        )

        volatility = np.where(np.logical_or(volatility == 0, np.isinf(volatility)), 1e-10, volatility)

        efficiency_ratio = np.clip((changes / volatility).fillna(0).replace([np.inf, -np.inf], 0), 0, 1)

        for idx in range(window, len(prices_array)):
            if not np.isfinite(prices_array[idx]):
                kama[idx] = kama[idx - 1]
                continue

            ratio_value = float(
                efficiency_ratio.iloc[idx] if isinstance(efficiency_ratio, pd.Series) else efficiency_ratio[idx]
            )
            if not np.isfinite(ratio_value):
                kama[idx] = kama[idx - 1]
                continue

            smoothing_constant = np.clip((ratio_value * (fast_sc - slow_sc) + slow_sc) ** 2, 1e-10, 1.0)
            price_diff = prices_array[idx] - kama[idx - 1]
            kama[idx] = kama[idx - 1] + smoothing_constant * price_diff

            if not np.isfinite(kama[idx]):
                kama[idx] = kama[idx - 1]

    except (ValueError, TypeError, IndexError):
        kama = pd.Series(prices).rolling(window=window, min_periods=1).mean().ffill().values

    kama_array = np.asarray(kama, dtype=np.float64)
    return np.where(~np.isfinite(kama_array), initial_value, kama_array).astype(np.float64)


def calculate_kama_series(prices: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> Optional[pd.Series]:
    """
    Calculate KAMA series from price data.

    Args:
        prices: Price series (typically close prices)
        period: Period for efficiency ratio calculation (default: 10)
        fast: Fast smoothing constant period (default: 2)
        slow: Slow smoothing constant period (default: 30)

    Returns:
        Series with KAMA values, or None if calculation fails
    """
    kama_values = calculate_kama(prices, window=period, fast=fast, slow=slow)
    if kama_values is None or len(kama_values) == 0:
        return None
    return pd.Series(kama_values, index=prices.index)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "MomentumIndicators",
    "calculate_kama",
    "calculate_kama_series",
    "calculate_rsi_series",
    "calculate_macd_series",
    "calculate_bollinger_bands_series",
    "calculate_stochrsi_series",
]
