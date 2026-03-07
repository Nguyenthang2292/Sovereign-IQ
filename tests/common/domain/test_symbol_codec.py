import pytest

from modules.common.domain.symbol_codec import SymbolCodec

codec = SymbolCodec()  # heuristic mode — no exchange


@pytest.mark.parametrize(
    "input_sym, expected_db",
    [
        ("BTCUSDT", "BTCUSDT"),
        ("BTC/USDT", "BTCUSDT"),
        ("BTC/USDT:USDT", "BTCUSDT"),  # the double-USDT bug case
        ("BTC-USDT", "BTCUSDT"),
        ("btc_usdt", "BTCUSDT"),
        ("SKL/USDT:USDT", "SKLUSDT"),  # normalize_symbol_key would give SKLUSDTUSDT
        ("eth/usdt", "ETHUSDT"),  # case-insensitive
    ],
)
def test_to_db(input_sym, expected_db):
    assert codec.to_db(input_sym) == expected_db


@pytest.mark.parametrize(
    "input_sym, expected_ccxt",
    [
        ("BTCUSDT", "BTC/USDT"),
        ("BTC/USDT:USDT", "BTC/USDT"),
        ("BTC-USDT", "BTC/USDT"),
        ("btc_usdt", "BTC/USDT"),
        ("ETHUSDT", "ETH/USDT"),
    ],
)
def test_to_ccxt(input_sym, expected_ccxt):
    assert codec.to_ccxt(input_sym) == expected_ccxt


@pytest.mark.parametrize(
    "input_sym, expected_futures",
    [
        ("BTCUSDT", "BTC/USDT:USDT"),
        ("BTC/USDT", "BTC/USDT:USDT"),
        ("BTC-USDT", "BTC/USDT:USDT"),
    ],
)
def test_to_futures_heuristic(input_sym, expected_futures):
    assert codec.to_futures(input_sym) == expected_futures


@pytest.mark.parametrize(
    "a, b",
    [
        ("BTCUSDT", "BTC/USDT"),
        ("BTC/USDT:USDT", "BTC/USDT"),
        ("SKL/USDT:USDT", "SKLUSDT"),
    ],
)
def test_equal(a, b):
    assert codec.equal(a, b)
