
from pathlib import Path
import json
import tempfile

import pandas as pd

from modules.position_sizing.utils.data_loader import (

from modules.position_sizing.utils.data_loader import (

"""
Tests for Data Loader.
"""



    load_symbols_from_dataframe,
    load_symbols_from_file,
    validate_symbols,
)


def test_load_symbols_from_dataframe():
    """Test loading symbols from DataFrame."""
    df = pd.DataFrame(
        {
            "symbol": ["BTC/USDT", "ETH/USDT"],
            "signal": [1, -1],
            "price": [50000.0, 3000.0],
            "exchange": ["binance", "binance"],
        }
    )

    symbols = load_symbols_from_dataframe(df)

    assert len(symbols) == 2
    assert symbols[0]["symbol"] == "BTC/USDT"
    assert symbols[0]["signal"] == 1
    assert symbols[1]["symbol"] == "ETH/USDT"
    assert symbols[1]["signal"] == -1


def test_load_symbols_from_csv():
    """Test loading symbols from CSV file."""
    df = pd.DataFrame(
        {
            "symbol": ["BTC/USDT", "ETH/USDT"],
            "signal": [1, -1],
            "price": [50000.0, 3000.0],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        symbols = load_symbols_from_file(temp_path)
        assert len(symbols) == 2
    finally:
        Path(temp_path).unlink()


def test_load_symbols_from_json():
    """Test loading symbols from JSON file."""
    data = [
        {"symbol": "BTC/USDT", "signal": 1, "price": 50000.0},
        {"symbol": "ETH/USDT", "signal": -1, "price": 3000.0},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        symbols = load_symbols_from_file(temp_path)
        assert len(symbols) == 2
    finally:
        Path(temp_path).unlink()


def test_validate_symbols():
    """Test symbol validation."""
    symbols = [
        {"symbol": "BTC/USDT", "signal": 1},
        {"symbol": "ETH/USDT", "signal": "LONG"},
        {"symbol": "INVALID", "signal": 0},  # Missing /USDT
    ]

    validated = validate_symbols(symbols)

    assert len(validated) == 3
    assert validated[0]["symbol"] == "BTC/USDT"
    assert validated[1]["symbol"] == "ETH/USDT"
    assert validated[1]["signal"] == 1  # Converted from 'LONG'
    assert validated[2]["symbol"] == "INVALID/USDT"  # Auto-added /USDT
