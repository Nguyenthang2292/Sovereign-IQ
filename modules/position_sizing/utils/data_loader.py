
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging

import pandas as pd

from modules.common.utils import (

from modules.common.utils import (

"""
Data loader utilities for position sizing module.

This module provides functions to load symbols from hybrid/voting analyzer results
or from CSV/JSON files.
"""



    log_error,
    log_progress,
    log_warn,
)

logger = logging.getLogger(__name__)


def load_symbols_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Load symbols from a pandas DataFrame (from hybrid/voting analyzer results).

    Args:
        df: DataFrame with columns: symbol, signal, price, exchange, etc.

    Returns:
        List of symbol dictionaries with required fields

    Raises:
        ValueError: If required columns are missing
    """
    if df is None or df.empty:
        return []

    required_columns = ["symbol"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    symbols = []
    for _, row in df.iterrows():
        symbol_dict = {
            "symbol": str(row["symbol"]).upper(),
            "signal": row.get("signal", 0),  # Default to 0 if not present
            "price": float(row.get("price", 0.0)),
            "exchange": str(row.get("exchange", "binance")).lower(),
        }

        # Add optional fields if present
        optional_fields = [
            "trend",
            "weighted_score",
            "cumulative_vote",
            "voting_breakdown",
            "osc_signal",
            "osc_confidence",
            "spc_signal",
            "spc_confidence",
            "xgboost_signal",
            "xgboost_confidence",
            "hmm_signal",
            "hmm_confidence",
            "random_forest_signal",
            "random_forest_confidence",
            "source",
        ]

        for field in optional_fields:
            if field in row and pd.notna(row[field]):
                symbol_dict[field] = row[field]

        symbols.append(symbol_dict)

    return symbols


def load_symbols_from_file(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load symbols from a CSV or JSON file.

    Args:
        filepath: Path to CSV or JSON file

    Returns:
        List of symbol dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or invalid
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_ext = filepath.suffix.lower()

    try:
        if file_ext == ".csv":
            df = pd.read_csv(filepath)
            return load_symbols_from_dataframe(df)
        elif file_ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                # List of dictionaries
                return data
            elif isinstance(data, dict):
                # Dictionary with 'symbols' key or DataFrame-like structure
                if "symbols" in data:
                    return data["symbols"]
                elif "data" in data:
                    # Try to convert to DataFrame
                    df = pd.DataFrame(data["data"])
                    return load_symbols_from_dataframe(df)
                else:
                    # Try to convert entire dict to DataFrame
                    df = pd.DataFrame([data])
                    return load_symbols_from_dataframe(df)
            else:
                raise ValueError(f"Unsupported JSON structure: {type(data)}")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .csv, .json")
    except Exception as e:
        log_error(f"Error loading symbols from file {filepath}: {e}")
        raise


def load_symbols_from_results(source: str, results_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Load symbols from hybrid/voting analyzer results.

    This function can be called with:
    1. A DataFrame directly (from analyzer results)
    2. A file path to saved results

    Args:
        source: Source identifier ("hybrid", "voting", or file path)
        results_df: Optional DataFrame from analyzer (if None, will try to load from file)

    Returns:
        List of symbol dictionaries
    """
    if results_df is not None:
        log_progress(f"Loading symbols from {source} analyzer results (DataFrame)...")
        return load_symbols_from_dataframe(results_df)

    # Try to load from common result file locations
    if source.lower() in ["hybrid", "voting"]:
        # Look for saved results files
        possible_paths = [
            Path(f"artifacts/results_{source.lower()}.csv"),
            Path(f"artifacts/results_{source.lower()}.json"),
            Path(f"results_{source.lower()}.csv"),
            Path(f"results_{source.lower()}.json"),
        ]

        for path in possible_paths:
            if path.exists():
                log_progress(f"Loading symbols from {path}...")
                return load_symbols_from_file(path)

        log_warn(f"No saved results file found for {source}. Please provide a DataFrame or file path.")
        return []

    # Assume source is a file path
    if Path(source).exists():
        log_progress(f"Loading symbols from file: {source}...")
        return load_symbols_from_file(source)

    raise FileNotFoundError(f"Source not found: {source}")


def validate_symbols(symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and clean symbol data.

    Args:
        symbols: List of symbol dictionaries

    Returns:
        Validated and cleaned list of symbols
    """
    validated = []

    for symbol in symbols:
        # Validate required fields
        if "symbol" not in symbol or not symbol["symbol"]:
            log_warn(f"Skipping symbol with missing 'symbol' field: {symbol}")
            continue

        # Ensure symbol is in correct format (BASE/QUOTE)
        symbol_str = str(symbol["symbol"]).upper().strip()
        if "/" not in symbol_str:
            # Try to add /USDT if not present
            symbol_str = f"{symbol_str}/USDT"

        symbol["symbol"] = symbol_str

        # Set defaults for missing fields
        symbol.setdefault("signal", 0)
        symbol.setdefault("price", 0.0)
        symbol.setdefault("exchange", "binance")

        # Validate signal type
        if "signal" in symbol:
            signal = symbol["signal"]
            if isinstance(signal, str):
                signal = signal.upper()
                if signal in ["LONG", "BUY", "1"]:
                    symbol["signal"] = 1
                elif signal in ["SHORT", "SELL", "-1"]:
                    symbol["signal"] = -1
                else:
                    symbol["signal"] = 0
            elif not isinstance(signal, (int, float)):
                symbol["signal"] = 0

        validated.append(symbol)

    return validated


def export_symbols_to_file(symbols: List[Dict[str, Any]], filepath: Union[str, Path], format: str = "csv") -> None:
    """
    Export symbols to a file.

    Args:
        symbols: List of symbol dictionaries
        filepath: Output file path
        format: Export format ('csv' or 'json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "csv":
        df = pd.DataFrame(symbols)
        df.to_csv(filepath, index=False)
        log_progress(f"Exported {len(symbols)} symbols to {filepath}")
    elif format.lower() == "json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(symbols, f, indent=2, default=str)
        log_progress(f"Exported {len(symbols)} symbols to {filepath}")
    else:
        raise ValueError(f"Unsupported export format: {format}")
