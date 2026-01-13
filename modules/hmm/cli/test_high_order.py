"""
Test script for High-Order HMM with real BTC data from exchanges.

This script demonstrates how to:
1. Fetch BTC/USDT OHLCV data from exchanges
2. Run High-Order HMM analysis
3. Display results
"""

import os
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")

# Add project root to path (3 levels up from cli/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import log_error, log_success, log_warn
from modules.hmm.core.high_order import TrueHighOrderHMM, true_high_order_hmm


def main():
    """Main test function."""
    print("=" * 80)
    print("High-Order HMM Test voi du lieu BTC/USDT thuc te")
    print("=" * 80)

    # Initialize ExchangeManager va DataFetcher
    print("\n1. Khoi tao ExchangeManager va DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Fetch BTC/USDT data
    symbol = "BTC/USDT"
    timeframe = "1h"
    limit = 1000  # So luong nen can lay (tang len de co nhieu swing points hon)

    print(f"\n2. Dang lay du lieu {symbol} tu exchange ({timeframe}, {limit} nen)...")
    try:
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol=symbol, timeframe=timeframe, limit=limit, check_freshness=False
        )

        if df is None or df.empty:
            log_error("Khong the lay du lieu OHLCV")
            return

        log_success(f"Da lay {len(df)} nen tu {exchange_id.upper()}")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"   - Thoi gian bat dau: {df.index[0]}")
            print(f"   - Thoi gian ket thuc: {df.index[-1]}")
        print(f"   - Gia dong cua cuoi: ${df['close'].iloc[-1]:.2f}")
        print(f"   - Gia cao nhat: ${df['high'].max():.2f}")
        print(f"   - Gia thap nhat: ${df['low'].min():.2f}")

    except Exception as e:
        log_error(f"Loi khi lay du lieu: {e}")
        return

    # Set index to timestamp if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
        else:
            log_warn("DataFrame khong co timestamp index. Tao index mac dinh...")
            df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="1h")

    # Run High-Order HMM analysis
    print("\n3. Chay High-Order HMM analysis...")
    print("   (Qua trinh nay co the mat vai phut do toi uu order k va cross-validation)")

    try:
        result = true_high_order_hmm(
            df=df,
            train_ratio=0.8,
            eval_mode=False,
            min_order=2,
            max_order=3,  # Giam xuong 3 de nhanh hon
        )

        print("\n" + "=" * 80)
        print("KET QUA HIGH-ORDER HMM")
        print("=" * 80)

        # Map signal to text
        signal_map = {-1: "BEARISH (Giam)", 0: "NEUTRAL (Trung tinh)", 1: "BULLISH (Tang)"}

        signal_text = signal_map.get(result.next_state_with_high_order_hmm, "UNKNOWN")

        print(f"\nDu doan trang thai tiep theo: {signal_text}")
        print(f"Xac suat: {result.next_state_probability:.2%}")
        print(f"Thoi gian du kien: {result.next_state_duration} {timeframe}")

        # Show analyzer details if available
        print("\n" + "-" * 80)
        print("CHI TIET MODEL")
        print("-" * 80)

        # Create analyzer de xem thong tin chi tiet
        analyzer = TrueHighOrderHMM(
            min_order=2,
            max_order=3,
            train_ratio=0.8,
        )
        analyzer.analyze(df, eval_mode=False)

        if analyzer.optimal_order is not None:
            print(f"Order duoc chon: {analyzer.optimal_order}")
            print(f"So trang thai: {analyzer.optimal_n_states}")

            if analyzer.optimal_order > 1:
                from modules.hmm.core.high_order import N_BASE_STATES, get_expanded_state_count

                expanded_states = get_expanded_state_count(N_BASE_STATES, analyzer.optimal_order)
                print(f"   -> So trang thai mo rong: {expanded_states} (3^{analyzer.optimal_order})")
        else:
            print("Khong the xac dinh order toi uu")

        if analyzer.states is not None:
            print(f"Tong so trang thai tu swing points: {len(analyzer.states)}")
            print(f"   - Train: {len(analyzer.train_states) if analyzer.train_states else 0}")
            print(f"   - Test: {len(analyzer.test_states) if analyzer.test_states else 0}")

        print("\n" + "=" * 80)
        print("TEST HOAN TAT!")
        print("=" * 80)

    except Exception as e:
        log_error(f"Loi khi chay High-Order HMM: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
