"""Momentum indicator block."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata
from modules.common.utils import validate_ohlcv_input

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """Momentum indicators: RSI, MACD, BBands, StochRSI, KAMA."""

    CATEGORY = "momentum"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        # Validate input
        validate_ohlcv_input(df, required_columns=["close"])
        
        result = df.copy()
        before = result.columns.tolist()

        def _fill_rsi(length: int) -> pd.Series:
            rsi = ta.rsi(result["close"], length=length)
            return (
                rsi.fillna(50.0)
                if rsi is not None
                else pd.Series(50.0, index=result.index)
            )

        result["RSI_9"] = _fill_rsi(9)
        result["RSI_14"] = _fill_rsi(14)
        result["RSI_25"] = _fill_rsi(25)

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

        bbands = ta.bbands(result["close"], length=20, std=2.0)
        if bbands is not None and not bbands.empty:
            bbp_cols = [col for col in bbands.columns if col.startswith("BBP")]
            if bbp_cols:
                # Sử dụng giá trị từ cột thực tế của pandas_ta nhưng giữ tên BBP_5_2.0 để tương thích ngược
                result["BBP_5_2.0"] = bbands[bbp_cols[0]].fillna(0.5)
            else:
                logger.warning(
                    "BBP column missing in Bollinger Bands output, falling back to 0.5."
                )
                result["BBP_5_2.0"] = 0.5
        else:
            logger.warning("Bollinger Bands calculation failed, defaulting BBP to 0.5.")
            result["BBP_5_2.0"] = 0.5

        stochrsi = ta.stochrsi(result["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi is not None and not stochrsi.empty:
            stochrsi = stochrsi.copy()
            for col in ["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"]:
                if col in stochrsi.columns:
                    stochrsi[col] = stochrsi[col].fillna(50.0)
            result[stochrsi.columns] = stochrsi
        else:
            logger.warning("Stochastic RSI failed, using neutral values (50).")
            result["STOCHRSIk_14_14_3_3"] = 50.0
            result["STOCHRSId_14_14_3_3"] = 50.0

        kama_series = calculate_kama_series(result["close"], period=10)
        if kama_series is not None:
            result["KAMA_10"] = kama_series

        metadata = collect_metadata(
            before,
            result.columns,
            MomentumIndicators.CATEGORY,
        )
        return result, metadata

def calculate_kama(prices, window: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    prices_array = np.asarray(prices, dtype=np.float64)

    if len(prices_array) < window:
        return (
            np.full_like(prices_array, float(prices_array.flat[0]))
            if len(prices_array) > 0
            else np.array([0.0])
        )

    kama = np.zeros_like(prices_array, dtype=np.float64)
    first_valid_idx = next(
        (i for i, price in enumerate(prices_array) if np.isfinite(price)), 0
    )
    initial_value = (
        float(prices_array[first_valid_idx])
        if first_valid_idx < len(prices_array)
        else float(np.nanmean(prices_array[:window]))
    )
    kama[:window] = initial_value

    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)

    try:
        price_series = pd.Series(prices)
        changes = price_series.diff(window).abs()
        volatility = (
            price_series.rolling(window)
            .apply(
                lambda values: (
                    np.sum(np.abs(np.diff(values))) if len(values) > 1 else 1e-10
                ),
                raw=False,
            )
            .fillna(1e-10)
        )

        volatility = np.where(
            np.logical_or(volatility == 0, np.isinf(volatility)), 1e-10, volatility
        )

        efficiency_ratio = np.clip(
            (changes / volatility).fillna(0).replace([np.inf, -np.inf], 0), 0, 1
        )

        for idx in range(window, len(prices_array)):
            if not np.isfinite(prices_array[idx]):
                kama[idx] = kama[idx - 1]
                continue

            ratio_value = float(
                efficiency_ratio.iloc[idx]
                if isinstance(efficiency_ratio, pd.Series)
                else efficiency_ratio[idx]
            )
            if not np.isfinite(ratio_value):
                kama[idx] = kama[idx - 1]
                continue

            smoothing_constant = np.clip(
                (ratio_value * (fast_sc - slow_sc) + slow_sc) ** 2, 1e-10, 1.0
            )
            price_diff = prices_array[idx] - kama[idx - 1]
            kama[idx] = kama[idx - 1] + smoothing_constant * price_diff

            if not np.isfinite(kama[idx]):
                kama[idx] = kama[idx - 1]

    except Exception as err:  # pragma: no cover
        from modules.common.utils import log_warn
        log_warn(f"Error in KAMA calculation: {err}. Using simple moving average fallback.")
        kama = (
            pd.Series(prices)
            .rolling(window=window, min_periods=1)
            .mean()
            .ffill()
            .values
        )

    kama_array = np.asarray(kama, dtype=np.float64)
    return np.where(~np.isfinite(kama_array), initial_value, kama_array).astype(
        np.float64
    )


def calculate_kama_series(
    prices: pd.Series, period: int = 10, fast: int = 2, slow: int = 30
) -> Optional[pd.Series]:
    kama_values = calculate_kama(prices, window=period, fast=fast, slow=slow)
    if kama_values is None or len(kama_values) == 0:
        return None
    return pd.Series(kama_values, index=prices.index)


def calculate_rsi_series(
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Tính toán RSI với period tùy chỉnh.
    
    Args:
        close: Close price series
        period: Period cho RSI (default: 14)
        
    Returns:
        Series với RSI values (0-100)
    """
    rsi = ta.rsi(close, length=period)
    if rsi is None:
        return pd.Series(50.0, index=close.index)
    # Normalize undefined RSI values to 50.0 (neutral)
    rsi = rsi.fillna(50.0)
    return rsi


def calculate_macd_series(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Tính toán MACD với parameters tùy chỉnh.
    
    Args:
        close: Close price series
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
        
    Returns:
        DataFrame với columns: MACD, MACD_signal, MACD_hist
    """
    macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
    if macd_df is None or macd_df.empty:
        # Return default values
        return pd.DataFrame({
            'MACD': 0.0,
            'MACD_signal': 0.0,
            'MACD_hist': 0.0
        }, index=close.index)
    
    # Map pandas_ta column names to our expected names
    # pandas_ta returns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    result = pd.DataFrame(index=close.index)
    
    # Find the MACD line column (format: MACD_fast_slow_signal)
    macd_col = [col for col in macd_df.columns if col.startswith('MACD_') and not col.startswith('MACDh') and not col.startswith('MACDs')]
    # Signal column (format: MACDs_fast_slow_signal)
    signal_col = [col for col in macd_df.columns if col.startswith('MACDs_')]
    # Histogram column (format: MACDh_fast_slow_signal)
    hist_col = [col for col in macd_df.columns if col.startswith('MACDh_')]
    
    if macd_col:
        result['MACD'] = macd_df[macd_col[0]].fillna(0.0)
    else:
        result['MACD'] = 0.0
    
    if signal_col:
        result['MACD_signal'] = macd_df[signal_col[0]].fillna(0.0)
    else:
        result['MACD_signal'] = 0.0
    
    if hist_col:
        result['MACD_hist'] = macd_df[hist_col[0]].fillna(0.0)
    else:
        # Calculate histogram as difference if not available
        result['MACD_hist'] = result['MACD'] - result['MACD_signal']
    
    return result


def calculate_bollinger_bands_series(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0
) -> pd.DataFrame:
    """
    Tính toán Bollinger Bands với parameters tùy chỉnh.
    
    Args:
        close: Close price series
        period: Period cho BB (default: 20)
        std: Standard deviation multiplier (default: 2.0)
        
    Returns:
        DataFrame với columns: BB_upper, BB_middle, BB_lower
    """
    bbands = ta.bbands(close, length=period, std=std)
    if bbands is None or bbands.empty:
        # Return default values based on SMA
        sma = ta.sma(close, length=period)
        if sma is None:
            sma = close.rolling(window=period).mean()
        return pd.DataFrame({
            'BB_upper': sma,
            'BB_middle': sma,
            'BB_lower': sma
        }, index=close.index)
    
    # Map pandas_ta column names to our expected names
    # pandas_ta returns: BBU_length_std, BBM_length_std, BBL_length_std, BBP_length_std
    result = pd.DataFrame(index=close.index)
    
    # Find columns (format: BBU_period_std, BBM_period_std, BBL_period_std)
    upper_col = [col for col in bbands.columns if col.startswith('BBU_')]
    middle_col = [col for col in bbands.columns if col.startswith('BBM_')]
    lower_col = [col for col in bbands.columns if col.startswith('BBL_')]
    
    # Fallback: calculate manually if columns not found
    sma = close.rolling(window=period).mean()
    std_val = close.rolling(window=period).std()
    
    if upper_col:
        result['BB_upper'] = bbands[upper_col[0]].fillna(sma + (std_val * std))
    else:
        result['BB_upper'] = sma + (std_val * std)
    
    if middle_col:
        result['BB_middle'] = bbands[middle_col[0]].fillna(sma)
    else:
        result['BB_middle'] = sma
    
    if lower_col:
        result['BB_lower'] = bbands[lower_col[0]].fillna(sma - (std_val * std))
    else:
        result['BB_lower'] = sma - (std_val * std)
    
    return result


__all__ = [
    "MomentumIndicators",
    "calculate_kama",
    "calculate_kama_series",
    "calculate_rsi_series",
    "calculate_macd_series",
    "calculate_bollinger_bands_series",
]
