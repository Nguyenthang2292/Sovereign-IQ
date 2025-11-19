import argparse
import re
import warnings

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from colorama import Fore, Style, init as colorama_init
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_QUOTE = "USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 1500
DEFAULT_EXCHANGES = [
    "binance",
    "kraken",
    "kucoin",
    "gate",
    "okx",
    "bybit",
    "mexc",
    "huobi",
]
DEFAULT_EXCHANGE_STRING = ",".join(DEFAULT_EXCHANGES)
PREDICTION_WINDOWS = {
    "30m": "12h",
    "45m": "18h",
    "1h": "24h",
    "2h": "36h",
    "4h": "48h",
    "6h": "72h",
    "12h": "7d",
    "1d": "7d",
}
TARGET_HORIZON = 24
TARGET_BASE_THRESHOLD = 0.01
TARGET_LABELS = ["DOWN", "NEUTRAL", "UP"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Converts a timeframe string like '30m', '1h', '1d' into minutes.
    """
    match = re.match(r"^\s*(\d+)\s*([mhdw])\s*$", timeframe.lower())
    if not match:
        return 60  # default 1h

    value, unit = match.groups()
    value = int(value)

    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 60 * 24
    if unit == "w":
        return value * 60 * 24 * 7
    return 60


def get_prediction_window(timeframe: str) -> str:
    """
    Returns a textual description of the prediction horizon based on timeframe.
    """
    timeframe = timeframe.lower()
    return PREDICTION_WINDOWS.get(timeframe, "next sessions")


def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    return f"{style}{color}{text}{Style.RESET_ALL}"


def format_price(value: float) -> str:
    """
    Formats prices/indicators with adaptive precision so tiny values remain readable.
    """
    if value is None or pd.isna(value):
        return "N/A"

    abs_val = abs(value)
    if abs_val >= 1:
        precision = 2
    elif abs_val >= 0.01:
        precision = 4
    elif abs_val >= 0.0001:
        precision = 6
    else:
        precision = 8

    return f"{value:.{precision}f}"


def apply_directional_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each row as UP/DOWN/NEUTRAL based on future price movement.
    """
    future_close = df["close"].shift(-TARGET_HORIZON)
    pct_change = (future_close - df["close"]) / df["close"]

    historical_ref = df["close"].shift(TARGET_HORIZON)
    historical_pct = (df["close"] - historical_ref) / historical_ref
    base_threshold = (
        historical_pct.abs()
        .fillna(TARGET_BASE_THRESHOLD)
        .clip(lower=TARGET_BASE_THRESHOLD)
    )
    atr_ratio = (
        df.get("ATR_RATIO_14_50", pd.Series(1.0, index=df.index))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.5, upper=2.0)
    )
    threshold_series = (base_threshold * atr_ratio).clip(lower=TARGET_BASE_THRESHOLD)
    df["DynamicThreshold"] = threshold_series

    df["TargetLabel"] = np.where(
        pct_change >= threshold_series,
        "UP",
        np.where(pct_change <= -threshold_series, "DOWN", "NEUTRAL"),
    )
    df.loc[future_close.isna(), "TargetLabel"] = np.nan
    df["Target"] = df["TargetLabel"].map(LABEL_TO_ID)
    return df


def normalize_symbol(user_input: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Converts user input like 'xmr' into 'XMR/USDT'. Keeps existing slash pairs.
    """
    if not user_input:
        return f"BTC/{quote}"

    norm = user_input.strip().upper()
    if "/" in norm:
        return norm

    if norm.endswith(quote):
        return f"{norm[:-len(quote)]}/{quote}"

    return f"{norm}/{quote}"


def prompt_with_default(message: str, default, cast=str):
    while True:
        raw = input(color_text(f"{message} (default {default}): ", Fore.CYAN))
        value = raw.strip()
        if not value:
            return default
        try:
            return cast(value)
        except ValueError:
            print(color_text("Invalid input. Please try again.", Fore.RED))


def resolve_input(cli_value, default, prompt_message, cast=str, allow_prompt=True):
    if cli_value is not None:
        return cast(cli_value)
    if allow_prompt:
        return prompt_with_default(prompt_message, default, cast)
    return default


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto movement predictor using technical indicators and XGBoost."
    )
    parser.add_argument(
        "-s",
        "--symbol",
        help=f"Trading pair symbol (default: {DEFAULT_SYMBOL}). Accepts formats like 'BTC/USDT' or 'btc'.",
    )
    parser.add_argument(
        "-q",
        "--quote",
        help=f"Quote currency when symbol is given without slash (default: {DEFAULT_QUOTE}).",
    )
    parser.add_argument(
        "-t",
        "--timeframe",
        help=f"Timeframe for OHLCV data (default: {DEFAULT_TIMEFRAME}, e.g., 30m, 1h, 4h).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help=f"Number of candles to fetch (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "-e",
        "--exchanges",
        help=f"Comma-separated list of exchanges to try (default: {DEFAULT_EXCHANGE_STRING}).",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts; rely only on CLI arguments.",
    )
    return parser.parse_args()


def fetch_data(symbol="BTC/USDT", timeframe="1h", limit=1000, exchanges=None):
    """
    Fetches OHLCV data trying multiple exchanges until fresh data is returned.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    freshness_minutes = max(timeframe_to_minutes(timeframe) * 1.5, 5)
    fallback = None

    print(
        color_text(
            f"Fetching {limit} candles for {symbol} ({timeframe})...",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    for exchange_id in exchanges:
        exchange_cls = getattr(ccxt, exchange_id)
        exchange = exchange_cls()
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            if df.empty:
                print(
                    color_text(
                        f"[{exchange_id.upper()}] No data retrieved.", Fore.YELLOW
                    )
                )
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            last_ts = df["timestamp"].iloc[-1]
            now = pd.Timestamp.now(tz="UTC")
            age_minutes = (now - last_ts).total_seconds() / 60.0

            if age_minutes <= freshness_minutes:
                print(
                    color_text(
                        f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (fresh).",
                        Fore.GREEN,
                    )
                )
                return df, exchange_id

            print(
                color_text(
                    f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (stale). Trying next exchange...",
                    Fore.YELLOW,
                )
            )
            fallback = (df, exchange_id)
        except Exception as e:
            print(
                color_text(
                    f"[{exchange_id.upper()}] Error fetching data: {e}", Fore.RED
                )
            )
            continue

    if fallback:
        df, exchange_id = fallback
        print(
            color_text(
                f"Using latest available data from {exchange_id.upper()} despite staleness.",
                Fore.MAGENTA,
            )
        )
        return df, exchange_id

    print(
        color_text("Failed to fetch data from all exchanges.", Fore.RED, Style.BRIGHT)
    )
    return None, None


def add_indicators(df):
    """
    Adds technical indicators using pandas_ta.
    """
    # Trend
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)

    # Momentum
    df["RSI_14"] = ta.rsi(df["close"], length=14)

    # Volatility
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_50"] = ta.atr(df["high"], df["low"], df["close"], length=50)
    df["ATR_RATIO_14_50"] = df["ATR_14"] / df["ATR_50"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = apply_directional_labels(df)

    # Drop NaN values created by indicators
    df.dropna(inplace=True)
    return df


def train_and_predict(df):
    """
    Trains XGBoost model and predicts the next movement.
    """
    # Features to use for prediction
    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "SMA_20",
        "SMA_50",
        "RSI_14",
        "ATR_14",
        "ATR_50",
        "ATR_RATIO_14_50",
    ]

    # Split data: Train on all except the last row (which has no target yet for validation,
    # but in this live scenario we train on everything available up to the second to last candle
    # to predict the movement for the very last known candle, OR we train on history to predict the FUTURE).

    # To predict the FUTURE (next candle after the latest one):
    # We need to train on data where we KNOW the outcome.
    # So we use the dataset where 'Target' is valid.
    # The last row of df currently has Target based on a future candle that doesn't exist yet (shift(-1)).
    # So the last row's Target is False/NaN (pandas shift fills with NaN, but we did dropna).
    # Wait, if we shift(-1), the last row gets NaN. dropna() removes it.
    # So 'df' now contains only historical data where we know the outcome.

    X = df[features]
    y = df["Target"].astype(int)

    def build_model():
        return xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            min_child_weight=3,
            random_state=42,
            objective="multi:softprob",
            num_class=len(TARGET_LABELS),
            eval_metric="mlogloss",
            n_jobs=-1,
        )

    # Train/Test split (80/20) for evaluation metrics (optional, but good practice)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = build_model()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(color_text(f"Holdout Accuracy: {score:.2f}", Fore.YELLOW))

    # Time-series cross validation
    max_splits = min(5, len(df) - 1)
    if max_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=max_splits)
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            cv_model = build_model()
            cv_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = np.argmax(cv_model.predict_proba(X.iloc[test_idx]), axis=1)
            acc = accuracy_score(y.iloc[test_idx], preds)
            cv_scores.append(acc)
            print(color_text(f"CV Fold {fold} Accuracy: {acc:.2f}", Fore.BLUE))
        mean_cv = sum(cv_scores) / len(cv_scores)
        print(
            color_text(
                f"CV Mean Accuracy ({len(cv_scores)} folds): {mean_cv:.2f}",
                Fore.GREEN,
                Style.BRIGHT,
            )
        )
    else:
        print(
            color_text(
                "Not enough data for cross-validation (requires >=3 samples).",
                Fore.YELLOW,
            )
        )

    model.fit(X, y)
    return model


def predict_next_move(model, last_row):
    """
    Predicts the probability for the next candle.
    """
    # Prepare the single row of features
    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "SMA_20",
        "SMA_50",
        "RSI_14",
        "ATR_14",
        "ATR_50",
        "ATR_RATIO_14_50",
    ]
    X_new = last_row[features].values.reshape(1, -1)

    # Predict probability
    # proba[0] is prob of class 0 (Down), proba[1] is prob of class 1 (Up)
    proba = model.predict_proba(X_new)[0]

    return proba


def main():
    args = parse_args()
    allow_prompt = not args.no_prompt

    quote = args.quote.upper() if args.quote else DEFAULT_QUOTE
    timeframe = resolve_input(
        args.timeframe, DEFAULT_TIMEFRAME, "Enter timeframe", str, allow_prompt
    ).lower()
    limit = args.limit if args.limit is not None else DEFAULT_LIMIT
    exchanges_input = args.exchanges if args.exchanges else DEFAULT_EXCHANGE_STRING
    exchanges = [
        ex.strip() for ex in exchanges_input.split(",") if ex.strip()
    ] or DEFAULT_EXCHANGES

    def run_once(raw_symbol):
        symbol = normalize_symbol(raw_symbol, quote)
        df, exchange_id = fetch_data(
            symbol, timeframe, limit=limit, exchanges=exchanges
        )
        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"

        if df is not None:
            df["SMA_20"] = ta.sma(df["close"], length=20)
            df["SMA_50"] = ta.sma(df["close"], length=50)
            df["RSI_14"] = ta.rsi(df["close"], length=14)
            df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["ATR_50"] = ta.atr(df["high"], df["low"], df["close"], length=50)
            df["ATR_RATIO_14_50"] = df["ATR_14"] / df["ATR_50"]
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            latest_data = df.iloc[-1:].copy()
            latest_data.fillna(method="ffill", inplace=True)

            df = apply_directional_labels(df)
            latest_threshold = df["DynamicThreshold"].iloc[-1]
            df.dropna(inplace=True)
            latest_data["DynamicThreshold"] = latest_threshold

            print(color_text(f"Training on {len(df)} samples...", Fore.CYAN))
            model = train_and_predict(df)

            proba = predict_next_move(model, latest_data)
            proba_percent = {
                label: proba[LABEL_TO_ID[label]] * 100 for label in TARGET_LABELS
            }
            best_idx = int(np.argmax(proba))
            direction = ID_TO_LABEL[best_idx]
            probability = proba_percent[direction]

            current_price = latest_data["close"].values[0]
            atr = latest_data["ATR_14"].values[0]
            prediction_window = get_prediction_window(timeframe)
            threshold_value = latest_data["DynamicThreshold"].iloc[0]
            prediction_context = f"{prediction_window} | {TARGET_HORIZON} candles >={threshold_value*100:.2f}% move"

            print("\n" + color_text("=" * 40, Fore.BLUE, Style.BRIGHT))
            print(
                color_text(
                    f"ANALYSIS FOR {symbol} | TF {timeframe} | {exchange_label}",
                    Fore.CYAN,
                    Style.BRIGHT,
                )
            )
            print(
                color_text(f"Current Price: {format_price(current_price)}", Fore.WHITE)
            )
            print(
                color_text(f"Market Volatility (ATR): {format_price(atr)}", Fore.WHITE)
            )
            print(color_text("-" * 40, Fore.BLUE))

            if direction == "UP":
                direction_color = Fore.GREEN
                atr_sign = 1
            elif direction == "DOWN":
                direction_color = Fore.RED
                atr_sign = -1
            else:
                direction_color = Fore.YELLOW
                atr_sign = 0

            print(
                color_text(
                    f"PREDICTION ({prediction_context}): {direction}",
                    direction_color,
                    Style.BRIGHT,
                )
            )
            print(color_text(f"Confidence: {probability:.2f}%", direction_color))

            prob_summary = " | ".join(
                f"{label}: {value:.2f}%" for label, value in proba_percent.items()
            )
            print(color_text(f"Probabilities -> {prob_summary}", Fore.WHITE))

            if direction == "NEUTRAL":
                print(
                    color_text(
                        "Market expected to stay within +/-{:.2f}% over the next {} candles.".format(
                            threshold_value * 100, TARGET_HORIZON
                        ),
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
                for multiple in (1, 2, 3):
                    target_price = current_price + atr_sign * multiple * atr
                    move_abs = abs(target_price - current_price)
                    move_pct = (
                        (move_abs / current_price) * 100 if current_price else None
                    )
                    move_pct_text = (
                        f"{move_pct:.2f}%" if move_pct is not None else "N/A"
                    )
                    print(
                        color_text(
                            f"  ATR x{multiple}: {format_price(target_price)} | Delta {format_price(move_abs)} ({move_pct_text})",
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

    try:
        while True:
            raw_symbol = resolve_input(
                args.symbol, DEFAULT_SYMBOL, "Enter symbol pair", str, allow_prompt
            )
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


if __name__ == "__main__":
    main()
