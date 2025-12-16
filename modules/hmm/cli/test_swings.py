"""
Test script for HMM-Swings with real BTC data from exchanges.

This script demonstrates how to:
1. Fetch BTC/USDT OHLCV data from exchanges
2. Run HMM-Swings analysis
3. Display results
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

# Add project root to path (3 levels up from cli/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.hmm.core.swings import hmm_swings, SwingsHMM
from modules.common.utils import log_info, log_success, log_error, log_warn

def main():
    """Main test function."""
    print("=" * 80)
    print("HMM-Swings Test voi du lieu BTC/USDT thuc te")
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
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            check_freshness=False
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
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
        else:
            log_warn("DataFrame khong co timestamp index. Tao index mac dinh...")
            df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')
    
    # Run HMM-Swings analysis
    print(f"\n3. Chay HMM-Swings analysis...")
    print("   (Qua trinh nay co the mat vai phut do toi uu so luong states va cross-validation)")
    
    try:
        result = hmm_swings(
            df=df,
            train_ratio=0.8,
            eval_mode=False,
        )
        
        print("\n" + "=" * 80)
        print("KET QUA HMM-SWINGS")
        print("=" * 80)
        
        # Map signal to text
        signal_map = {
            -1: "BEARISH (Giam)",
            0: "NEUTRAL (Trung tinh)",
            1: "BULLISH (Tang)"
        }
        
        signal_text = signal_map.get(result.next_state_with_high_order_hmm, "UNKNOWN")
        
        print(f"\nDu doan trang thai tiep theo: {signal_text}")
        print(f"Xac suat: {result.next_state_probability:.2%}")
        print(f"Thoi gian du kien: {result.next_state_duration} {timeframe}")
        
        # Show analyzer details if available
        print("\n" + "-" * 80)
        print("CHI TIET MODEL")
        print("-" * 80)
        
        # Create analyzer de xem thong tin chi tiet
        analyzer = SwingsHMM(
            train_ratio=0.8,
        )
        analyzer_result = analyzer.analyze(df, eval_mode=False)
        
        if analyzer.optimal_n_states is not None:
            print(f"So trang thai toi uu: {analyzer.optimal_n_states}")
        else:
            print("Khong the xac dinh so trang thai toi uu")
        
        if analyzer.states is not None:
            print(f"Tong so trang thai tu swing points: {len(analyzer.states)}")
            print(f"   - Train: {len(analyzer.train_states) if analyzer.train_states else 0}")
            print(f"   - Test: {len(analyzer.test_states) if analyzer.test_states else 0}")
        
        if analyzer.swing_highs_info is not None and analyzer.swing_lows_info is not None:
            print(f"So swing highs: {len(analyzer.swing_highs_info)}")
            print(f"So swing lows: {len(analyzer.swing_lows_info)}")
        
        print("\n" + "=" * 80)
        print("TEST HOAN TAT!")
        print("=" * 80)
        
    except Exception as e:
        log_error(f"Loi khi chay HMM-Swings: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

