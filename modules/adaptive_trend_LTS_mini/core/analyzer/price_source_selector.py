"""
Price Source Selector for ATC Analysis.

This module handles selection and validation of price sources from OHLCV data.
"""

from typing import Optional

import pandas as pd

from modules.common.utils import log_error, log_warn

__all__ = ["PriceSourceSelector"]


class PriceSourceSelector:
    """
    Handles price source selection and validation.

    Responsible for:
    - Validating calculation source configuration
    - Extracting price series from DataFrame
    - Extracting current price
    """

    # Valid price sources from OHLCV data
    VALID_SOURCES = ["close", "open", "high", "low"]

    @staticmethod
    def select_price_source(
        df: pd.DataFrame,
        calculation_source: str,
        symbol: str,
    ) -> Optional[tuple[pd.Series, float]]:
        """
        Select and validate price source from DataFrame.

        Args:
            df: OHLCV DataFrame containing price data
            calculation_source: Desired price source (close, open, high, low)
            symbol: Symbol name (for error messages)

        Returns:
            Tuple of (price_series, current_price) if successful, None if failed
            price_series: pandas Series containing historical prices
            current_price: Latest price as float
        """
        # Normalize source to lowercase
        source = calculation_source.lower()

        # Validate source
        if source not in PriceSourceSelector.VALID_SOURCES:
            log_warn(
                f"Invalid calculation_source '{calculation_source}', using 'close'"
            )
            source = "close"

        # Check if source exists in DataFrame
        if source not in df.columns:
            log_error(f"No '{source}' column in data for {symbol}")
            return None

        # Extract price series and current price
        price_series: pd.Series = df[source]
        current_price: float = float(price_series.iloc[-1])

        return price_series, current_price
