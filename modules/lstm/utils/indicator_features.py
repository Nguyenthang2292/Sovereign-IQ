
import pandas as pd

from config.model_features import CANDLESTICK_PATTERN_NAMES, MODEL_FEATURES
from modules.common.indicators.momentum import (
import pandas_ta as ta
import pandas_ta as ta

"""
Indicator feature generation utilities for CNN-LSTM models.
Generates only features defined in config.model_features.MODEL_FEATURES.
Uses indicator calculation functions from modules.common.indicators for consistency.
"""


    calculate_bollinger_bands_series,
    calculate_macd_series,
    calculate_rsi_series,
    calculate_stochrsi_series,
)
from modules.common.indicators.trend import calculate_ma_series
from modules.common.indicators.volatility import calculate_atr_series
from modules.common.indicators.volume import calculate_obv_series
from modules.common.ui.logging import log_error, log_model, log_warn


def generate_indicator_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for market data analysis.
    Only calculates features that are defined in MODEL_FEATURES.
    Uses indicator calculation functions from modules.common.indicators for consistency.

    Args:
        df_input: DataFrame with OHLCV price data (lowercase column names)

    Returns:
        DataFrame with added technical indicators or empty DataFrame on error
    """
    try:
        # Validate input data - normalize column names first for case-insensitive check
        if df_input.empty:
            log_warn("Input DataFrame is empty")
            return pd.DataFrame()

        df = df_input.copy()

        # Normalize column names to lowercase for processing
        df.columns = df.columns.str.lower()

        # Check for required columns after normalization
        if "close" not in df.columns:
            log_warn("Input DataFrame is missing close column")
            return pd.DataFrame()

        close_col = "close"

        # Calculate RSI indicators (if in MODEL_FEATURES)
        rsi_features = [f for f in MODEL_FEATURES if f.startswith("RSI_")]
        if rsi_features:
            for rsi_feature in rsi_features:
                try:
                    # Extract period from feature name (e.g., RSI_9 -> 9, RSI_14 -> 14)
                    period = int(rsi_feature.split("_")[1])
                    rsi = calculate_rsi_series(df[close_col], period=period)
                    df[rsi_feature] = rsi.fillna(50.0)
                except Exception as e:
                    log_warn(f"RSI {rsi_feature} calculation failed: {e}. Using default value 50.0.")
                    df[rsi_feature] = 50.0

        # Calculate SMA indicators (if in MODEL_FEATURES)
        sma_features = [f for f in MODEL_FEATURES if f.startswith("SMA_")]
        if sma_features:
            for sma_feature in sma_features:
                # Parse and validate period first
                try:
                    period = int(sma_feature.split("_")[1])
                except (ValueError, IndexError) as e:
                    log_warn(f"SMA {sma_feature} period extraction failed: {e}. Skipping feature.")
                    df[sma_feature] = df[close_col]  # Use close price as fallback
                    continue

                # Only proceed if period is a valid positive integer
                if not isinstance(period, int) or period <= 0:
                    log_warn(f"SMA {sma_feature} has invalid period {period}. Using fallback value 0.0.")
                    df[sma_feature] = 0.0
                    continue

                try:
                    sma = calculate_ma_series(df[close_col], period=period, ma_type="SMA")
                    if sma is not None:
                        df[sma_feature] = sma.fillna(df[close_col].rolling(window=period).mean())
                    else:
                        df[sma_feature] = df[close_col].rolling(window=period).mean()
                except Exception as e:
                    log_warn(f"SMA {sma_feature} calculation failed: {e}. Using fallback value 0.0.")
                    df[sma_feature] = 0.0

        # Calculate MACD indicators (if in MODEL_FEATURES)
        macd_features = [f for f in MODEL_FEATURES if f.startswith("MACD")]
        if macd_features:
            try:
                macd_df = calculate_macd_series(df[close_col], fast=12, slow=26, signal=9)
                if macd_df is not None and not macd_df.empty:
                    # Map common/indicators column names to MODEL_FEATURES column names
                    # calculate_macd_series returns: 'MACD', 'MACD_signal', 'MACD_hist'
                    # MODEL_FEATURES expects: 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
                    column_mapping = {
                        "MACD": "MACD_12_26_9",
                        "MACD_signal": "MACDs_12_26_9",
                        "MACD_hist": "MACDh_12_26_9",
                    }

                    for common_col, feature_col in column_mapping.items():
                        if feature_col in macd_features and common_col in macd_df.columns:
                            df[feature_col] = macd_df[common_col].fillna(0.0)
                        elif feature_col in macd_features:
                            log_warn(f"MACD column {common_col} not found, using default 0.0 for {feature_col}.")
                            df[feature_col] = 0.0
                else:
                    # Fallback to default values
                    for macd_feature in macd_features:
                        df[macd_feature] = 0.0
            except Exception as e:
                log_warn(f"MACD calculation failed: {e}. Using default values 0.0.")
                for macd_feature in macd_features:
                    df[macd_feature] = 0.0

        # Calculate Bollinger Bands Percent (BBP) if in MODEL_FEATURES
        if "BBP_5_2.0" in MODEL_FEATURES:
            try:
                # calculate_bollinger_bands_series returns BB_upper, BB_middle, BB_lower
                # We need to calculate BBP = (close - lower) / (upper - lower)
                bbands_df = calculate_bollinger_bands_series(df[close_col], period=5, std=2.0)
                if bbands_df is not None and not bbands_df.empty:
                    if "BB_lower" in bbands_df.columns and "BB_upper" in bbands_df.columns:
                        bb_lower = bbands_df["BB_lower"]
                        bb_upper = bbands_df["BB_upper"]
                        bb_range = bb_upper - bb_lower
                        # Avoid division by zero
                        df["BBP_5_2.0"] = ((df[close_col] - bb_lower) / bb_range.replace(0, 1.0)).fillna(0.5)
                    else:
                        # Fallback: try to get BBP directly from pandas_ta
                        bbands_ta = ta.bbands(df[close_col], length=5, std=2.0)
                        if bbands_ta is not None and not bbands_ta.empty:
                            bbp_cols = [col for col in bbands_ta.columns if col.startswith("BBP")]
                            if bbp_cols:
                                df["BBP_5_2.0"] = bbands_ta[bbp_cols[0]].fillna(0.5)
                            else:
                                df["BBP_5_2.0"] = 0.5
                        else:
                            df["BBP_5_2.0"] = 0.5
                else:
                    df["BBP_5_2.0"] = 0.5
            except Exception as e:
                log_warn(f"Bollinger Bands calculation failed: {e}. Using default value 0.5.")
                df["BBP_5_2.0"] = 0.5

        # Calculate Stochastic RSI if in MODEL_FEATURES
        stochrsi_features = [f for f in MODEL_FEATURES if f.startswith("STOCHRSI")]
        if stochrsi_features:
            try:
                stochrsi_df = calculate_stochrsi_series(df[close_col], length=14, rsi_length=14, k=3, d=3)
                if stochrsi_df is not None and not stochrsi_df.empty:
                    for stochrsi_feature in stochrsi_features:
                        if stochrsi_feature in stochrsi_df.columns:
                            df[stochrsi_feature] = stochrsi_df[stochrsi_feature].fillna(50.0)
                        else:
                            df[stochrsi_feature] = 50.0
                else:
                    for stochrsi_feature in stochrsi_features:
                        df[stochrsi_feature] = 50.0
            except Exception as e:
                log_warn(f"Stochastic RSI calculation failed: {e}. Using default value 50.0.")
                for stochrsi_feature in stochrsi_features:
                    df[stochrsi_feature] = 50.0

        # Calculate ATR if in MODEL_FEATURES
        if "ATR_14" in MODEL_FEATURES:
            try:
                if "high" in df.columns and "low" in df.columns:
                    atr = calculate_atr_series(df["high"], df["low"], df[close_col], length=14)
                    df["ATR_14"] = atr.fillna(0.0)
                else:
                    df["ATR_14"] = 0.0
            except Exception as e:
                log_warn(f"ATR calculation failed: {e}. Using default value 0.0.")
                df["ATR_14"] = 0.0

        # Calculate OBV if in MODEL_FEATURES
        if "OBV" in MODEL_FEATURES:
            try:
                if "volume" in df.columns:
                    obv = calculate_obv_series(df[close_col], df["volume"])
                    df["OBV"] = obv.fillna(0.0)
                else:
                    df["OBV"] = 0.0
            except Exception as e:
                log_warn(f"OBV calculation failed: {e}. Using default value 0.0.")
                df["OBV"] = 0.0

        # Note: Candlestick patterns are typically calculated elsewhere
        # They are included in MODEL_FEATURES but not calculated here
        candlestick_features = [f for f in MODEL_FEATURES if f in CANDLESTICK_PATTERN_NAMES]
        if candlestick_features:
            log_warn(
                "Candlestick patterns are not calculated in this version. Setting all candlestick pattern features to 0.0."
            )
            for candlestick_pattern in candlestick_features:
                df[candlestick_pattern] = 0.0

        # Fill NaNs from indicator calculations
        df = df.ffill().bfill()

        # Remove rows with remaining NaN values
        rows_before = len(df)
        result_df = df.dropna()
        rows_after = len(result_df)

        if rows_after > 0 and rows_after < rows_before * 0.5:
            log_warn(
                f"Significant data loss: {rows_before - rows_after} of {rows_before} rows dropped due to NaN values ({(1 - rows_after / rows_before) * 100:.1f}%)"
            )

        if result_df.empty:
            log_warn("All features resulted in NaN - insufficient data for technical indicators")
        else:
            # Normalize all column names to lowercase for consistency
            result_df.columns = result_df.columns.str.lower()
            calculated_features = [f.lower() for f in MODEL_FEATURES if f.lower() in result_df.columns]
            log_model(
                f"Technical indicators calculated successfully: {len(calculated_features)} features for {len(result_df)} rows"
            )

        return result_df

    except Exception as e:
        log_error(f"Error in feature calculation: {e}")
        return pd.DataFrame()
