import sys
import warnings
from pathlib import Path

# Add project root to sys.path to ensure config module can be imported
# This is needed when running the file directly from subdirectories
if "__file__" in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

import importlib.util

# Import from cli.py file (not cli package) to avoid circular import

import numpy as np
from colorama import Fore, Style
from colorama import init as colorama_init

# Import from xgboost_prediction_ modules (modules specific to xgboost_prediction_main.py)
from config import (
    DEFAULT_EXCHANGE_STRING,
    DEFAULT_EXCHANGES,
    DEFAULT_LIMIT,
    DEFAULT_QUOTE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    ID_TO_LABEL,
    LABEL_TO_ID,
    TARGET_BASE_THRESHOLD,
    TARGET_HORIZON,
    TARGET_LABELS,
)
from modules.common.utils import (
    color_text,
    format_price,
    log_error,
)
from modules.common.domain.symbol_codec import SymbolCodec
from modules.xgboost_LTS.utils.utils import get_prediction_window

cli_file_path = Path(__file__).parent / "argument_parser.py"
spec = importlib.util.spec_from_file_location("xgboost_cli_module", cli_file_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load CLI parser module from {cli_file_path}")
cli_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_module)
parse_args = cli_module.parse_args
resolve_input = cli_module.resolve_input
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.indicator_engine import (
    IndicatorConfig,
    IndicatorEngine,
    IndicatorProfile,
)
from modules.targets import calculate_atr_targets, format_atr_target_display
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.core.model import predict_next_move, train_and_predict
from modules.xgboost_LTS.utils.batch_symbols import batch_train_symbols

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

_SYMBOL_CODEC = SymbolCodec()


def main():
    args = parse_args()
    allow_prompt = not args.no_prompt

    quote = args.quote.upper() if args.quote else DEFAULT_QUOTE
    timeframe = resolve_input(args.timeframe, DEFAULT_TIMEFRAME, "Enter timeframe", str, allow_prompt).lower()
    limit = args.limit if args.limit is not None else DEFAULT_LIMIT
    exchanges_input = args.exchanges if args.exchanges else DEFAULT_EXCHANGE_STRING
    exchanges = [ex.strip() for ex in exchanges_input.split(",") if ex.strip()] or DEFAULT_EXCHANGES

    # Initialize ExchangeManager, DataFetcher, IndicatorEngine
    exchange_manager = ExchangeManager()  # No credentials needed for OHLCV
    data_fetcher = DataFetcher(exchange_manager)
    indicator_engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))

    # Set exchange priority if custom exchanges provided
    if exchanges != DEFAULT_EXCHANGES:
        exchange_manager.public.exchange_priority_for_fallback = exchanges

    def _prepare_symbol_data(raw_symbol):
        symbol = str(_SYMBOL_CODEC.to_ccxt(raw_symbol if "/" in raw_symbol else f"{raw_symbol}/{quote}"))
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
            exchanges=exchanges if exchanges != DEFAULT_EXCHANGES else None,
        )
        if df is None:
            return None

        # Calculate basic indicators without labels first (to preserve latest_data)
        df = indicator_engine.compute_features(df)

        # Calculate advanced features required by XGBoost MODEL_FEATURES
        # This includes: ROC, atr_ratio, price_to_SMA, rolling stats, lag features, time features
        from modules.xgboost_LTS.utils.features import add_advanced_features

        df = add_advanced_features(df)

        # Save latest data before applying labels and dropping NaN
        latest_data = df.iloc[-1:].copy().ffill()

        # Apply directional labels and drop NaN for training data
        df = apply_directional_labels(df)
        latest_threshold = df["DynamicThreshold"].iloc[-1] if len(df) > 0 else TARGET_BASE_THRESHOLD
        df.dropna(inplace=True)
        latest_data["DynamicThreshold"] = latest_threshold

        return {
            "symbol": symbol,
            "exchange_label": exchange_id.upper() if exchange_id else "UNKNOWN",
            "train_df": df,
            "latest_data": latest_data,
            "threshold": latest_threshold,
        }

    def run_once(raw_symbol):
        prepared = _prepare_symbol_data(raw_symbol)
        if prepared is not None:
            symbol = prepared["symbol"]
            exchange_label = prepared["exchange_label"]
            df = prepared["train_df"]
            latest_data = prepared["latest_data"]

            print(color_text(f"Training on {len(df)} samples...", Fore.CYAN))
            model = train_and_predict(df)

            proba = predict_next_move(model, latest_data)
            proba_percent = {label: proba[LABEL_TO_ID[label]] * 100 for label in TARGET_LABELS}
            best_idx = int(np.argmax(proba))
            direction = ID_TO_LABEL[best_idx]
            probability = proba_percent[direction]

            current_price = latest_data["close"].values[0]
            atr = latest_data["ATR_14"].values[0]
            prediction_window = get_prediction_window(timeframe)
            threshold_value = latest_data["DynamicThreshold"].iloc[0]
            prediction_context = f"{prediction_window} | {TARGET_HORIZON} candles >={threshold_value * 100:.2f}% move"

            print("\n" + color_text("=" * 40, Fore.BLUE, Style.BRIGHT))
            print(
                color_text(
                    f"ANALYSIS FOR {symbol} | TF {timeframe} | {exchange_label}",
                    Fore.CYAN,
                    Style.BRIGHT,
                )
            )
            print(color_text(f"Current Price: {format_price(current_price)}", Fore.WHITE))
            print(color_text(f"Market Volatility (ATR): {format_price(atr)}", Fore.WHITE))
            print(color_text("-" * 40, Fore.BLUE))

            if direction == "UP":
                direction_color = Fore.GREEN
            elif direction == "DOWN":
                direction_color = Fore.RED
            else:
                direction_color = Fore.YELLOW

            print(
                color_text(
                    f"PREDICTION ({prediction_context}): {direction}",
                    direction_color,
                    Style.BRIGHT,
                )
            )
            print(color_text(f"Confidence: {probability:.2f}%", direction_color))

            prob_summary = " | ".join(f"{label}: {value:.2f}%" for label, value in proba_percent.items())
            print(color_text(f"Probabilities -> {prob_summary}", Fore.WHITE))

            if direction == "NEUTRAL":
                # Calculate upper and lower price bounds based on threshold
                upper_bound = current_price * (1 + threshold_value)
                lower_bound = current_price * (1 - threshold_value)
                price_range = upper_bound - lower_bound

                print(
                    color_text(
                        "Market expected to stay within +/-{:.2f}% over the next {} candles.".format(
                            threshold_value * 100, TARGET_HORIZON
                        ),
                        Fore.YELLOW,
                    )
                )
                print(
                    color_text(
                        f"Price Range: {format_price(lower_bound)} - {format_price(upper_bound)}",
                        Fore.YELLOW,
                    )
                )
                print(
                    color_text(
                        f"  Upper Bound: {format_price(upper_bound)} (+{threshold_value * 100:.2f}%)",
                        Fore.YELLOW,
                    )
                )
                print(
                    color_text(
                        f"  Lower Bound: {format_price(lower_bound)} (-{threshold_value * 100:.2f}%)",
                        Fore.YELLOW,
                    )
                )
                print(
                    color_text(
                        f"  Range Width: {format_price(price_range)} ({threshold_value * 200:.2f}%)",
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "Estimated Targets via ATR multiples:",
                        Fore.MAGENTA,
                        Style.BRIGHT,
                    )
                )
                # Tính toán ATR targets sử dụng module mới
                atr_targets = calculate_atr_targets(
                    current_price=current_price,
                    atr=atr,
                    direction=direction,
                    multiples=[1, 2, 3],
                )
                # Hiển thị kết quả
                for target_result in atr_targets:
                    display_text = format_atr_target_display(
                        target_result,
                        format_price_func=format_price,
                    )
                    print(
                        color_text(
                            display_text,
                            Fore.MAGENTA,
                        )
                    )
            print(color_text("=" * 40, Fore.BLUE, Style.BRIGHT))
        else:
            print(
                color_text(
                    "Unable to proceed without market data. Please try again later.",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )

    def _symbols_from_file(file_path: str) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Symbols file not found: {file_path}")
        symbols: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                symbols.append(item)
        return symbols

    def _resolve_batch_symbols() -> list[str]:
        symbols: list[str] = []

        if args.symbols:
            symbols.extend([s.strip() for s in args.symbols.split(",") if s.strip()])

        if args.symbols_file:
            symbols.extend(_symbols_from_file(args.symbols_file))

        if args.symbol:
            symbols.append(args.symbol)

        if not symbols and allow_prompt:
            raw = resolve_input(
                None,
                DEFAULT_SYMBOL,
                "Enter comma-separated symbols for batch mode",
                str,
                allow_prompt,
            )
            symbols.extend([s.strip() for s in raw.split(",") if s.strip()])

        if not symbols:
            symbols = [DEFAULT_SYMBOL]

        deduped = []
        seen = set()
        for s in symbols:
            ns = str(_SYMBOL_CODEC.to_ccxt(s if "/" in s else f"{s}/{quote}"))
            if ns not in seen:
                seen.add(ns)
                deduped.append(ns)
        return deduped

    def run_batch_mode():
        symbols = _resolve_batch_symbols()
        print(color_text(f"Batch mode enabled for {len(symbols)} symbols.", Fore.CYAN, Style.BRIGHT))

        prepared: dict[str, dict] = {}
        for raw_symbol in symbols:
            info = _prepare_symbol_data(raw_symbol)
            if info is None:
                log_warn(f"Skipping {raw_symbol}: unable to fetch market data.")
                continue
            if len(info["train_df"]) == 0:
                log_warn(f"Skipping {info['symbol']}: no training rows after labeling/dropna.")
                continue
            prepared[info["symbol"]] = info

        if not prepared:
            print(color_text("No valid symbols to train in batch mode.", Fore.RED, Style.BRIGHT))
            return

        if args.batch_use_dask:
            try:
                import dask.dataframe as dd
                from config import MODEL_FEATURES, XGBOOST_PARAMS
                from modules.xgboost_LTS.core.model_dask import train_and_predict_dask
            except Exception as exc:
                print(color_text(f"Batch Dask mode unavailable: {exc}", Fore.RED, Style.BRIGHT))
                return

            ok_count = 0
            fail_count = 0
            for symbol, info in prepared.items():
                try:
                    npartitions = max(1, min(8, len(info["train_df"]) // 2000 or 1))
                    df_dask = dd.from_pandas(info["train_df"], npartitions=npartitions)
                    train_and_predict_dask(
                        df_dask,
                        model_features=MODEL_FEATURES,
                        params=XGBOOST_PARAMS.copy(),
                        scheduler_address=args.dask_scheduler_address,
                        use_cuda=args.dask_use_cuda,
                        n_workers=args.dask_workers,
                        threads_per_worker=args.dask_threads_per_worker,
                        memory_limit=args.dask_memory_limit,
                    )
                    ok_count += 1
                    print(color_text(f"[OK] {symbol}: Dask training completed.", Fore.GREEN))
                except Exception as exc:
                    fail_count += 1
                    print(color_text(f"[FAIL] {symbol}: {exc}", Fore.RED))
            print(color_text(f"Batch Dask summary -> ok: {ok_count}, fail: {fail_count}", Fore.CYAN, Style.BRIGHT))
            return

        symbols_data = {symbol: info["train_df"] for symbol, info in prepared.items()}
        results = batch_train_symbols(
            symbols_data=symbols_data,
            train_and_predict_fn=train_and_predict,
            max_workers=args.max_workers,
            use_cache=not args.batch_no_cache,
            show_progress=not args.no_batch_progress,
            return_result=False,
        )

        ok_count = sum(1 for v in results.values() if v.get("ok"))
        fail_count = len(results) - ok_count
        print(color_text(f"Batch summary -> ok: {ok_count}, fail: {fail_count}", Fore.CYAN, Style.BRIGHT))

        for symbol, payload in results.items():
            if not payload.get("ok"):
                print(color_text(f"[FAIL] {symbol}: {payload.get('error', 'Unknown error')}", Fore.RED))

    try:
        if args.batch:
            run_batch_mode()
            return

        while True:
            raw_symbol = resolve_input(args.symbol, DEFAULT_SYMBOL, "Enter symbol pair", str, allow_prompt)
            run_once(raw_symbol)
            args.symbol = None  # force prompt next iteration
            if not allow_prompt:
                break
            print(
                color_text(
                    "\nPress Ctrl+C to exit. Provide a new symbol to continue.",
                    Fore.YELLOW,
                )
            )
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
    except Exception as e:
        print(color_text(f"\nCRITICAL ERROR: {str(e)}", Fore.RED, Style.BRIGHT))
        log_error(f"XGBoost CLI encountered a fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
