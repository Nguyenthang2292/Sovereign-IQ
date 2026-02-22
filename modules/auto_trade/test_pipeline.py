"""
Auto-Trade Pipeline Test Script

Tests the complete signal pipeline:
1. Rust Build (ATC + XGBoost features)
2. ATC Scanner (Multi-timeframe trend detection)
3. XGBoost Filter (ML signal validation)
4. Gemini Analysis (AI chart analysis)
5. Signal Selection (Final signal with confidence)

Usage:
    python test_pipeline.py
    python test_pipeline.py --symbols BTC/USDT ETH/USDT
    python test_pipeline.py --help
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import ATC_SCANNER_DEFAULTS, SIGNAL_SELECTOR_DEFAULTS, XGBOOST_FILTER_DEFAULTS
from modules.auto_trade.core.atc_scanner import ATCScanner
from modules.auto_trade.core.gemini_integration import GeminiIntegration
from modules.auto_trade.core.signal_pipeline import SignalPipeline
from modules.auto_trade.core.signal_selector import SignalSelector
from modules.auto_trade.core.symbol_manager import SymbolManager
from modules.auto_trade.core.xgboost_filter import XGBoostFilter
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_error, log_warn


def print_banner():
    """Print test banner."""
    print("=" * 80)
    print("🚀 AUTO-TRADE PIPELINE TEST")
    print("=" * 80)
    print()


def check_rust_availability():
    """Check if Rust components are available."""
    print("📦 Checking Rust Components...")
    print("-" * 80)

    try:
        import atc_rust

        print("✅ atc_rust (Rust backend) - AVAILABLE")

        # Test ScanCache
        try:
            cache = atc_rust.ScanCache(capacity=10, ttl_seconds=60.0)
            cache.set("test", {"BTC/USDT"}, set(), {"BTC/USDT": 0.9})
            result = cache.get("test")
            if result:
                print("✅ ScanCache - WORKING")
            else:
                print("⚠️ ScanCache - INITIALIZED BUT NOT WORKING")
        except Exception as e:
            print(f"❌ ScanCache - ERROR: {e}")

        # Test calculate_weighted_score
        try:
            score = atc_rust.calculate_weighted_score("LONG", 0.5, 0.8, False)
            print(f"✅ calculate_weighted_score - WORKING (score={score})")
        except Exception as e:
            print(f"❌ calculate_weighted_score - ERROR: {e}")

        # Test aggregate_signals
        try:
            test_results = {"1h": {"longs": {"BTC/USDT"}, "shorts": set(), "strengths": {"BTC/USDT": 0.9}}}
            aggregated = atc_rust.aggregate_signals(["BTC/USDT"], test_results, {"1h": 1.0}, 0.5, False)
            print(f"✅ aggregate_signals - WORKING (found {len(aggregated)} signals)")
        except Exception as e:
            print(f"❌ aggregate_signals - ERROR: {e}")

        print()
        return True
    except ImportError:
        print("❌ atc_rust (Rust backend) - NOT AVAILABLE")
        print(
            "   Build Rust: cd modules/adaptive_trend_LTS_mini/rust_extensions && python -m maturin develop --release"
        )
        print()
        return False


def find_latest_model() -> str:
    """Find the latest XGBoost model in artifacts."""
    # Check default location first
    default_path = Path("models/xgboost_model.joblib")
    if default_path.exists():
        return str(default_path)

    # Check artifacts directory
    artifacts_dir = Path("artifacts/xgboost/models")
    if artifacts_dir.exists():
        models = list(artifacts_dir.glob("*.joblib"))
        if models:
            # Sort by modification time (newest first)
            latest_model = sorted(models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            print(f"📦 Auto-discovered latest model: {latest_model}")
            return str(latest_model)

    return "models/xgboost_model.joblib"


def check_xgboost_model(model_path: str):
    """Check if XGBoost model exists."""
    print("🤖 Checking XGBoost Model...")
    print("-" * 80)

    path = Path(model_path)
    if path.exists():
        print(f"✅ Model found: {path}")
        print(f"   Size: {path.stat().st_size / 1024 / 1024:.2f} MB")
        print()
        return True
    else:
        print(f"❌ Model not found: {path}")
        print("   Please train and save XGBoost model first")
        print()
        return False


def check_gemini_api():
    """Check if Gemini API is configured."""
    print("🤖 Checking Gemini API...")
    print("-" * 80)

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✅ GEMINI_API_KEY configured: {masked}")
        print()
        return True
    else:
        print("❌ GEMINI_API_KEY not found in environment")
        print("   Set: export GEMINI_API_KEY=your_api_key")
        print()
        return False


def initialize_pipeline(model_path: str, symbols: Optional[List[str]] = None, sample_rate: float = 10.0):
    """Initialize all pipeline components."""
    print("🔧 Initializing Pipeline Components...")
    print("-" * 80)

    # 0. Exchange Manager
    print("0. ExchangeManager...")
    exchange_manager = ExchangeManager()
    print("   ✅ Initialized")

    # 1. Data Fetcher
    print("1. DataFetcher...")
    data_fetcher = DataFetcher(exchange_manager=exchange_manager)
    print("   ✅ Initialized")

    # 2. Symbol Manager
    print("2. SymbolManager...")
    symbol_manager = SymbolManager(data_fetcher=data_fetcher)

    # Determine symbols
    test_symbols = []
    if symbols:
        test_symbols = symbols
        print(f"   ✅ Using explicit list of {len(test_symbols)} symbols")
    else:
        print("   🔄 Fetching all symbols from DataFetcher...")
        all_symbols = data_fetcher.list_binance_futures_symbols()
        total_count = len(all_symbols)

        if total_count == 0:
            log_warn("   ⚠️ No symbols found from DataFetcher!")
        else:
            # Apply random sampling
            sample_size = max(1, int(total_count * (sample_rate / 100.0)))
            sample_size = min(sample_size, total_count)
            test_symbols = random.sample(all_symbols, sample_size)
            print(f"   🎲 Random sampling: {sample_rate}% of {total_count} = {sample_size} symbols")

    # Inject test symbols and disable refresh
    symbol_manager._cached_symbols = test_symbols
    symbol_manager.refresh_symbols = lambda: print("   (Skipping symbol refresh for test)")  # type: ignore[method-assign]
    print(
        f"   ✅ Using {len(test_symbols)} test symbols: {', '.join(test_symbols[:5])}"
        + ("..." if len(test_symbols) > 5 else "")
    )

    # 3. ATC Scanner
    print("3. ATCScanner...")
    # Lower threshold for testing to ensure we get signals
    test_config = ATC_SCANNER_DEFAULTS.copy()
    test_config["threshold"] = 0.01
    test_config["use_rust_cache"] = True

    atc_scanner = ATCScanner(data_fetcher=data_fetcher, config=test_config)  # type: ignore[arg-type]
    print(f"   ✅ Configured: {ATC_SCANNER_DEFAULTS['timeframes']} timeframes")
    print(f"   ✅ Rust cache: {ATC_SCANNER_DEFAULTS['use_rust_cache']}")

    # 4. XGBoost Filter
    print("4. XGBoostFilter...")
    xgboost_filter = XGBoostFilter(data_fetcher=data_fetcher, model_path=model_path, config=XGBOOST_FILTER_DEFAULTS)  # type: ignore[arg-type]
    print(f"   ✅ Model: {model_path}")
    print(f"   ✅ Min confidence: {XGBOOST_FILTER_DEFAULTS['min_confidence']}")

    # 5. Gemini Integration
    print("5. GeminiIntegration...")
    gemini_integration = GeminiIntegration(data_fetcher=data_fetcher, analysis_timeframe="1h")
    if gemini_integration.is_available():
        print("   ✅ API configured")
    else:
        print("   ⚠️ API not configured (will skip Gemini analysis)")

    # 6. Signal Selector
    print("6. SignalSelector...")
    signal_selector = SignalSelector(config=SIGNAL_SELECTOR_DEFAULTS)
    print(
        f"   ✅ Weights: XGBoost={SIGNAL_SELECTOR_DEFAULTS['weight_xgboost']}, "
        f"Gemini={SIGNAL_SELECTOR_DEFAULTS['weight_gemini']}"
    )

    # 7. Signal Pipeline (DynamoDB - persistence optional for tests)
    print("7. SignalPipeline...")
    pipeline = SignalPipeline(
        symbol_manager=symbol_manager,
        atc_scanner=atc_scanner,
        xgboost_filter=xgboost_filter,
        gemini_integration=gemini_integration,
        signal_selector=signal_selector,
        signal_persistence=None,  # Skip persistence for test pipeline
        config={
            "max_symbols_to_scan": len(test_symbols),
            "max_ai_candidates": min(5, len(test_symbols)),
        },
    )
    print("   ✅ Pipeline ready (no persistence)")
    print()

    return pipeline


def display_signal_result(signal):
    """Display final signal with detailed information."""
    if not signal:
        print("❌ NO SIGNAL GENERATED")
        print("   No trading opportunity found that meets criteria")
        return

    print("=" * 80)
    print("🎯 FINAL SIGNAL")
    print("=" * 80)
    print()

    print(f"Symbol:       {signal.symbol}")
    print(f"Signal Type:  {signal.signal_type}")
    print(f"Confidence:   {signal.confidence:.1%}")
    print(f"Score:        {signal.score:.2f}/100")
    print()

    print("📊 Price Levels:")
    print(f"  Entry:       ${signal.entry_price:,.2f}")
    print(f"  Stop Loss:   ${signal.stop_loss:,.2f}")
    print(f"  Take Profit: ${signal.take_profit:,.2f}")
    print()

    # Calculate risk/reward
    if signal.signal_type == "LONG":
        risk = signal.entry_price - signal.stop_loss
        reward = signal.take_profit - signal.entry_price
    else:  # SHORT
        risk = signal.stop_loss - signal.entry_price
        reward = signal.entry_price - signal.take_profit

    rr_ratio = reward / risk if risk > 0 else 0
    print(f"  Risk:        ${risk:,.2f}")
    print(f"  Reward:      ${reward:,.2f}")
    print(f"  R/R Ratio:   {rr_ratio:.2f}")
    print()

    print("🔍 Source Breakdown:")
    print(f"  XGBoost:     {signal.sources.get('xgboost_score', 0):.1%} confidence")
    print(f"  Gemini:      {signal.sources.get('gemini_score', 0):.1%} confidence")
    if signal.sources.get("gemini_reasoning"):
        print(f"  Reasoning:   {signal.sources['gemini_reasoning'][:100]}...")
    print()

    print(f"⏰ Timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(signal.timestamp))}")
    print("=" * 80)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Auto-Trade Pipeline")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to test (if not provided, fetches all symbols from DataFetcher)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=10.0,
        help="Percentage of symbols to sample when fetching all (default: 10.0%%)",
    )
    parser.add_argument("--skip-checks", action="store_true", help="Skip component checks")
    parser.add_argument("--model", help="Path to XGBoost model (optional)")
    parser.add_argument(
        "--force-signal", action="store_true", help="Force a mock ATC signal to test downstream components"
    )
    args = parser.parse_args()

    print_banner()

    # Resolve model path
    model_path = args.model
    if not model_path:
        model_path = find_latest_model()

    # Pre-flight checks
    if not args.skip_checks:
        rust_ok = check_rust_availability()
        model_ok = check_xgboost_model(model_path)
        gemini_ok = check_gemini_api()

        if not rust_ok:
            log_warn("Rust components not available - performance will be degraded")
        if not model_ok:
            log_error("XGBoost model not found - pipeline will fail")
            return 1
        if not gemini_ok:
            log_warn("Gemini API not configured - skipping AI analysis")

    # Initialize pipeline
    try:
        pipeline = initialize_pipeline(model_path=model_path, symbols=args.symbols, sample_rate=args.sample_rate)

        if args.force_signal:
            print("🧪 --force-signal enabled: Patching ATCScanner to return mock signal")
            from modules.auto_trade.core.atc_scanner import SignalResult

            def mock_scan(symbols):
                print("   (Using MOCK ATC results for testing pipeline flow)")
                return [
                    SignalResult(
                        symbol=symbols[0],
                        score=0.85,
                        signal_type="LONG",
                        details={"1h": "LONG", "15m": "LONG", "5m": "NEUTRAL"},
                        strengths={"1h": 0.9, "15m": 0.8, "5m": 0.1},
                    )
                ]

            pipeline.atc_scanner.scan_symbols = mock_scan

            def mock_filter(signals):
                print("   (Using MOCK XGBoost results for testing pipeline flow)")
                # Enhance signals with XGBoost score
                enhanced = []
                for s in signals:
                    # Update details with XGBoost info
                    s.details["xgboost_conf"] = 0.95
                    s.details["xgboost_dir"] = "LONG"
                    # Return list of SignalResult
                    enhanced.append(s)
                return enhanced

            pipeline.xgboost_filter.filter_signals = mock_filter

    except Exception as e:
        log_error(f"Pipeline initialization failed: {e}")
        return 1

    # Run pipeline
    print("🚀 Running Pipeline...")
    print("=" * 80)
    print()

    start_time = time.time()

    try:
        final_signal = pipeline.run_pipeline()
        duration = time.time() - start_time

        print()
        print(f"⏱️ Pipeline completed in {duration:.2f}s")
        print()

        # Display result
        display_signal_result(final_signal)

        # Get pipeline health
        health_status = pipeline.health_registry.check_health()
        print("🏥 System Health:")
        for component, status in health_status.items():
            try:
                # Handle both Tuple(Status, str) and other formats
                if isinstance(status, tuple) and len(status) == 2:
                    status_enum, msg = status
                    status_name = status_enum.name if hasattr(status_enum, "name") else str(status_enum)
                else:
                    status_name = "UNKNOWN"
                    msg = str(status)

                status_emoji = "✅" if status_name == "HEALTHY" else "⚠️"
                print(f"  {status_emoji} {component}: {msg}")
            except Exception:
                print(f"  ❓ {component}: {status}")
        print()

        return 0

    except Exception as e:
        duration = time.time() - start_time
        log_error(f"Pipeline failed after {duration:.2f}s: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
