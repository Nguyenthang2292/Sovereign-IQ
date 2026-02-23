"""
Local smoke-test cho Lambda trainer pipeline.

Chạy ĐÚNG cùng pipeline với handler.py (fetch → indicators → labels → train)
nhưng KHÔNG cần Docker và KHÔNG upload S3.

Dùng trước mỗi lần docker build để catch lỗi ngay tại local.

Usage:
    python scripts/test_trainer_local.py
    python scripts/test_trainer_local.py --symbol ETH/USDT --timeframe 1h --limit 500
    python scripts/test_trainer_local.py --dry-run   # chỉ check imports, không fetch

Env: load từ modules/auto_trade/.env (giống Lambda runtime)
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# ── Thêm project root vào sys.path (giống khi chạy từ root) ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load .env (giống Lambda env vars) ────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402
_ENV_FILE = PROJECT_ROOT / "modules" / "auto_trade" / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
    print(f"[env] Loaded {_ENV_FILE}")
else:
    print(f"[env] WARNING: {_ENV_FILE} not found — dùng env vars hệ thống")

STEP_OK   = "  ✓"
STEP_FAIL = "  ✗"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Kiểm tra import (phát hiện ModuleNotFoundError ngay lập tức)
# ─────────────────────────────────────────────────────────────────────────────
def check_imports() -> bool:
    print("\n[1/5] Checking imports...")
    required = [
        ("xgboost",  "xgboost"),
        ("pandas",   "pandas"),
        ("numpy",    "numpy"),
        ("sklearn",  "scikit-learn"),
        ("ccxt",     "ccxt"),
        ("optuna",   "optuna"),
        ("joblib",   "joblib"),
        ("colorama", "colorama"),
    ]
    ok = True
    for mod, pkg in required:
        try:
            __import__(mod)
            print(f"{STEP_OK} {pkg}")
        except ImportError as e:
            print(f"{STEP_FAIL} {pkg}: {e}")
            ok = False

    # Internal modules
    internal = [
        "modules.common.core.data_fetcher",
        "modules.common.core.exchange_manager",
        "modules.common.core.indicator_engine",
        "modules.xgboost_LTS.core.labeling",
        "modules.xgboost_LTS.core.model",
        "modules.xgboost_LTS.utils.features",
    ]
    for mod in internal:
        try:
            __import__(mod)
            print(f"{STEP_OK} {mod.split('.')[-1]}")
        except Exception as e:
            print(f"{STEP_FAIL} {mod}: {e}")
            ok = False

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full pipeline (giống handler.py, bỏ S3)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(symbol: str, timeframe: str, fetch_limit: int) -> bool:
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.indicator_engine import (
        IndicatorConfig, IndicatorEngine, IndicatorProfile,
    )
    from modules.xgboost_LTS.core.labeling import apply_directional_labels
    from modules.xgboost_LTS.core.model import train_and_predict
    from modules.xgboost_LTS.utils.features import add_advanced_features

    t0 = time.perf_counter()

    # ── 2. Fetch OHLCV ────────────────────────────────────────────────────────
    print(f"\n[2/5] Fetching {fetch_limit} candles — {symbol} {timeframe}...")
    try:
        exchange = ExchangeManager()
        fetcher  = DataFetcher(exchange)
        df = fetcher.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=fetch_limit,
            check_freshness=False,
        )
    except Exception as e:
        print(f"{STEP_FAIL} fetch_ohlcv failed: {e}")
        traceback.print_exc()
        return False

    if df is None or df.empty:
        print(f"{STEP_FAIL} fetch_ohlcv returned empty DataFrame")
        return False
    print(f"{STEP_OK} {len(df)} candles — columns: {list(df.columns)[:6]}...")

    # ── 3. Indicators ─────────────────────────────────────────────────────────
    print(f"\n[3/5] Computing indicators (profile=XGBOOST)...")
    try:
        engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))
        result = engine.compute_features(df)
        df = result[0] if isinstance(result, tuple) else result
        df = add_advanced_features(df)
    except Exception as e:
        print(f"{STEP_FAIL} indicator_engine failed: {e}")
        traceback.print_exc()
        return False
    print(f"{STEP_OK} {len(df)} rows after indicators, {len(df.columns)} columns")

    # ── 4. Labels ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Applying directional labels...")
    try:
        df = apply_directional_labels(df, use_cache=False)
        before = len(df)
        df = df.dropna(subset=["Target"])
        dropped = before - len(df)
    except Exception as e:
        print(f"{STEP_FAIL} apply_directional_labels failed: {e}")
        traceback.print_exc()
        return False

    if df.empty:
        print(f"{STEP_FAIL} No rows left after labeling (dropped {dropped})")
        print("  → Tăng --limit (vd: --limit 1000) để có đủ dữ liệu sau warmup indicators")
        return False
    print(f"{STEP_OK} {len(df)} labeled rows  (dropped {dropped} NaN targets)")
    print(f"  Target distribution: {df['Target'].value_counts().to_dict()}")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training XGBoost model...")
    try:
        model = train_and_predict(df, use_cache=False)
    except Exception as e:
        print(f"{STEP_FAIL} train_and_predict failed: {e}")
        traceback.print_exc()
        return False

    elapsed = time.perf_counter() - t0
    print(f"{STEP_OK} Training done in {elapsed:.1f}s")
    print(f"  Model type: {type(model).__name__}")
    print(f"\n{'='*55}")
    print(f"  PIPELINE OK — {symbol} {timeframe}  [{elapsed:.1f}s]")
    print(f"{'='*55}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Local smoke-test cho Lambda trainer pipeline")
    parser.add_argument("--symbol",    default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="15m",      help="Timeframe (default: 15m)")
    parser.add_argument("--limit",     default=500, type=int,
                        help="Số candles fetch (default: 500; dùng >=500 để có đủ warmup rows)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Chỉ kiểm tra imports, không fetch/train")
    args = parser.parse_args()

    print("=" * 55)
    print("  Trainer Pipeline — Local Smoke Test")
    print("=" * 55)

    # Step 1: imports
    imports_ok = check_imports()
    if not imports_ok:
        print("\n[FAIL] Import errors — fix requirements trước khi docker build\n")
        sys.exit(1)

    if args.dry_run:
        print("\n[OK] dry-run: tất cả imports thành công\n")
        sys.exit(0)

    # Steps 2-5: full pipeline
    success = run_pipeline(args.symbol, args.timeframe, args.limit)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
