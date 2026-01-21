"""
Data utilities for DataFrame/Series manipulation and OHLCV data fetching.

This package provides:
- Validation functions for OHLCV DataFrames and price series
- Transformation functions for DataFrames and Series
- Data fetching functions for multiple symbols and timeframes
"""

from .fetchers import fetch_ohlcv_data_dict
from .transformation import dataframe_to_close_series
from .validation import OHLCV_REQUIRED_COLUMNS, validate_ohlcv_input, validate_price_series

__all__ = [
    # Validation
    "validate_ohlcv_input",
    "validate_price_series",
    "OHLCV_REQUIRED_COLUMNS",
    # Transformation
    "dataframe_to_close_series",
    # Data fetching
    "fetch_ohlcv_data_dict",
]
