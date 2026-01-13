"""
Test script for HMM-KAMA with real BTC data from exchanges.

This script demonstrates how to:
1. Fetch BTC/USDT OHLCV data from exchanges
2. Run HMM-KAMA analysis
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
from modules.hmm.core.kama import hmm_kama


def main():
    """Main test function."""
    print("=" * 80)
    print("HMM-KAMA Test voi du lieu BTC/USDT thuc te")
    print("=" * 80)

    # Initialize ExchangeManager va DataFetcher
    print("\n1. Khoi tao ExchangeManager va DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Fetch BTC/USDT data
    symbol = "BTC/USDT"
    timeframe = "1h"
    limit = 1000  # So luong nen can lay

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

    # Run HMM-KAMA analysis
    print("\n3. Chay HMM-KAMA analysis...")
    print("   (Qua trinh nay co the mat vai phut do training HMM, ARM, va K-Means)")

    try:
        result = hmm_kama(
            df=df,
        )

        print("\n" + "=" * 80)
        print("KET QUA HMM-KAMA")
        print("=" * 80)

        # Map signals to text
        hmm_state_map = {
            -1: "INVALID (Khong hop le)",
            0: "BEARISH STRONG (Giam manh)",
            1: "BEARISH WEAK (Giam yeu)",
            2: "BULLISH WEAK (Tang yeu)",
            3: "BULLISH STRONG (Tang manh)",
        }

        binary_state_map = {-1: "INVALID (Khong hop le)", 0: "BEARISH (Giam)", 1: "BULLISH (Tang)"}

        print("\n--- Du doan trang thai tiep theo (HMM-KAMA) ---")
        next_state_text = hmm_state_map.get(result.next_state_with_hmm_kama, "UNKNOWN")
        print(f"Trang thai: {next_state_text}")

        print("\n--- Trang thai hien tai (Standard Deviation) ---")
        std_state_text = binary_state_map.get(result.current_state_of_state_using_std, "UNKNOWN")
        print(f"Trang thai: {std_state_text}")

        print("\n--- Trang thai hien tai (HMM) ---")
        hmm_state_text = binary_state_map.get(result.current_state_of_state_using_hmm, "UNKNOWN")
        print(f"Trang thai: {hmm_state_text}")

        print("\n--- Trang thai cao xac suat (ARM - Apriori) ---")
        arm_apriori_text = hmm_state_map.get(result.state_high_probabilities_using_arm_apriori, "UNKNOWN")
        print(f"Trang thai: {arm_apriori_text}")

        print("\n--- Trang thai cao xac suat (ARM - FP-Growth) ---")
        arm_fpgrowth_text = hmm_state_map.get(result.state_high_probabilities_using_arm_fpgrowth, "UNKNOWN")
        print(f"Trang thai: {arm_fpgrowth_text}")

        print("\n--- Trang thai hien tai (K-Means) ---")
        kmeans_state_text = binary_state_map.get(result.current_state_of_state_using_kmeans, "UNKNOWN")
        print(f"Trang thai: {kmeans_state_text}")

        print("\n" + "=" * 80)
        print("TEST HOAN TAT!")
        print("=" * 80)

    except Exception as e:
        log_error(f"Loi khi chay HMM-KAMA: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
