"""
Order Book Module - Manual Test Entry Point
===========================================
Run this file directly to smoke-test all components against real Binance Futures data:
  - market_data_fetcher  : fetch_depth, fetch_agg_trades
  - order_book_imbalance_calculator : calculate_combined_score
  - order_book_imbalance_gate       : OrderBookImbalanceGate.check

Usage (from repo root):
    python -m modules.order_book.main
    python -m modules.order_book.main --symbol BTCUSDT --signal LONG
    python -m modules.order_book.main --testnet
"""

import argparse

# ---------------------------------------------------------------------------
# Allow running from repo root OR from within the module directory
# ---------------------------------------------------------------------------
import os
import sys
import time

# Make sure the repo root is on sys.path when invoked directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.order_book.market_data_fetcher import fetch_agg_trades, fetch_depth
from modules.order_book.order_book_imbalance_calculator import calculate_combined_score
from modules.order_book.order_book_imbalance_gate import OrderBookImbalanceGate

# ---------------------------------------------------------------------------
# ANSI helpers (work on Win10+ / WSL / most terminals)
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔ {msg}{RESET}")


def _fail(msg: str) -> None:
    print(f"  {RED}✘ {msg}{RESET}")


def _header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


# ---------------------------------------------------------------------------
# Individual component tests
# ---------------------------------------------------------------------------


def test_fetch_depth(symbol: str, testnet: bool) -> bool:
    _header(f"[1] fetch_depth  →  {symbol}  (testnet={testnet})")
    t0 = time.perf_counter()
    snapshot = fetch_depth(symbol=symbol, limit=20, testnet=testnet)
    elapsed = time.perf_counter() - t0

    if snapshot is None:
        _fail(f"fetch_depth returned None  ({elapsed:.2f}s)")
        return False

    _ok(f"Symbol  : {snapshot.symbol}")
    _ok(f"Bids    : {len(snapshot.bids)} levels  | top bid = {snapshot.bids[0] if snapshot.bids else 'N/A'}")
    _ok(f"Asks    : {len(snapshot.asks)} levels  | top ask = {snapshot.asks[0] if snapshot.asks else 'N/A'}")
    _ok(f"TS      : {snapshot.timestamp:.3f}  ({elapsed * 1000:.0f} ms)")
    return True


def test_fetch_agg_trades(symbol: str, testnet: bool) -> bool:
    _header(f"[2] fetch_agg_trades  →  {symbol}  (window=5m, testnet={testnet})")
    t0 = time.perf_counter()
    trades = fetch_agg_trades(symbol=symbol, window_minutes=5, testnet=testnet)
    elapsed = time.perf_counter() - t0

    if trades is None:
        _fail(f"fetch_agg_trades returned None  ({elapsed:.2f}s)")
        return False

    _ok(f"Trade count : {len(trades)}  ({elapsed * 1000:.0f} ms)")
    if trades:
        sample = trades[0]
        _ok(
            f"Sample trade: price={sample.price}, qty={sample.quantity}, "
            f"buyer_maker={sample.is_buyer_maker}, ts={sample.timestamp:.3f}"
        )
    return True


def test_calculate_combined_score(symbol: str, testnet: bool) -> bool:
    _header(f"[3] calculate_combined_score  →  {symbol}")

    snapshot = fetch_depth(symbol=symbol, limit=100, testnet=testnet)
    trades = fetch_agg_trades(symbol=symbol, window_minutes=5, testnet=testnet)

    if snapshot is None:
        _fail("Skipping: fetch_depth returned None")
        return False
    if trades is None:
        _fail("Skipping: fetch_agg_trades returned None")
        return False

    t0 = time.perf_counter()
    result = calculate_combined_score(snapshot=snapshot, trades=trades)
    elapsed = time.perf_counter() - t0

    _ok(f"OBI score     : {result.obi_score:+.4f}  (raw={result.obi_raw:+.4f})")
    _ok(f"Delta score   : {result.delta_score:+.4f}  (raw={result.delta_raw:+.4f})")
    _ok(f"Combined score: {result.combined_score:+.4f}")
    _ok(f"Weighted mid  : {result.weighted_mid:.4f}  ({elapsed * 1000:.0f} ms)")
    return True


def test_gate(symbol: str, signal: str, testnet: bool) -> bool:
    _header(f"[4] OrderBookImbalanceGate.check  →  {symbol}  signal={signal}")

    gate = OrderBookImbalanceGate(
        threshold=0.15,
        retry_wait_seconds=0,  # no real wait during smoke-test
        max_retries=0,  # single attempt to keep test fast
        delta_window_minutes=5,
        testnet=testnet,
        enabled=True,
    )

    t0 = time.perf_counter()
    decision, combined_result = gate.check(symbol=symbol, signal_type=signal)
    elapsed = time.perf_counter() - t0

    _ok(f"Decision : {decision.value}  ({elapsed * 1000:.0f} ms)")
    if combined_result is not None:
        _ok(f"Combined score: {combined_result.combined_score:+.4f}")
    else:
        _ok("Combined result: None (fail-open path)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the order_book module against real Binance Futures data.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Futures symbol, e.g. BTCUSDT (default)")
    parser.add_argument("--signal", default="LONG", choices=["LONG", "SHORT"], help="Signal direction for gate test")
    parser.add_argument("--testnet", action="store_true", help="Use Binance Futures testnet endpoints")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    signal = args.signal.upper()
    testnet = args.testnet

    print(f"\n{BOLD}Order Book Module — Smoke Test{RESET}")
    print(f"Symbol : {symbol}")
    print(f"Signal : {signal}")
    print(f"Testnet: {testnet}")

    results: dict[str, bool] = {}

    results["fetch_depth"] = test_fetch_depth(symbol, testnet)
    results["fetch_agg_trades"] = test_fetch_agg_trades(symbol, testnet)
    results["calculate_combined_score"] = test_calculate_combined_score(symbol, testnet)
    results["gate.check"] = test_gate(symbol, signal, testnet)

    # Summary
    _header("Summary")
    all_pass = True
    for name, passed in results.items():
        if passed:
            _ok(name)
        else:
            _fail(name)
            all_pass = False

    print()
    if all_pass:
        print(f"{GREEN}{BOLD}All tests PASSED.{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}Some tests FAILED. Check warnings above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
