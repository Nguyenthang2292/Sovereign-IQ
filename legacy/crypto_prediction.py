import ccxt
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import warnings
import logging
from colorama import Fore, Style, init
from imblearn.over_sampling import SMOTE

# Initialize colorama for colored console output
init(autoreset=True)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Custom colored formatter for logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.CYAN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'SUCCESS': Fore.GREEN,
        'RESULT': Fore.MAGENTA,
        'ALERT': Fore.RED,
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            record.msg = f"{self.COLORS[levelname]}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Configure logging
logging.addLevelName(25, 'SUCCESS')  # Custom level between INFO and WARNING
logging.addLevelName(27, 'ALERT')    # Custom level for bearish predictions
logging.addLevelName(35, 'RESULT')   # Custom level between WARNING and ERROR

def setup_logger():
    """Set up and return a configured logger"""
    logger = logging.getLogger('CryptoPrediction')
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter('%(message)s'))
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# Feature columns used for model training
FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'SMA_50', 'SMA_200', 'RSI_9', 'RSI_14', 'RSI_25', 'ATR_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'BBP_5_2.0',
    'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3',
    'OBV',
    # Candlestick patterns
    'DOJI', 'HAMMER', 'INVERTED_HAMMER', 'SHOOTING_STAR',
    'BULLISH_ENGULFING', 'BEARISH_ENGULFING',
    'MORNING_STAR', 'EVENING_STAR', 'PIERCING', 'DARK_CLOUD'
]

def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    """Fetch OHLCV data from multiple exchanges until fresh data is found."""
    exchanges_to_try = ['binance', 'kraken', 'kucoin', 'gate', 'okx', 'bybit', 'mexc', 'huobi']
    for exchange_id in exchanges_to_try:
        logger.info(f"Trying to fetch data from {exchange_id.upper()}...")
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logger.warning(f" - No data found on {exchange_id}.")
                continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # Freshness check using UTC
            last_time = df['timestamp'].iloc[-1]
            time_diff = datetime.now(timezone.utc) - (last_time.tz_localize('UTC') if last_time.tzinfo is None else last_time)
            if time_diff.days > 1:
                logger.warning(f" - Data on {exchange_id} is stale ({time_diff.days} days old). Skipping.")
                continue
            logger.log(25, f"SUCCESS: Found fresh data on {exchange_id.upper()}!")
            return df
        except Exception as e:
            logger.error(f" - Error on {exchange_id}: {str(e)}")
            continue
    logger.error("ERROR: Could not find fresh data on any exchange.")
    return None

def add_candlestick_patterns(df):
    """Detect 10 reliable candlestick patterns and add binary columns."""
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    body = abs(c - o)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - l
    range_hl = h - l
    range_hl = np.where(range_hl == 0, 0.0001, range_hl)
    range_hl = pd.Series(range_hl, index=df.index)
    df['DOJI'] = (body / range_hl < 0.1).astype(int)
    df['HAMMER'] = ((lower_shadow > 2 * body) & (upper_shadow < 0.3 * body) & (body / range_hl < 0.3)).astype(int)
    df['INVERTED_HAMMER'] = ((upper_shadow > 2 * body) & (lower_shadow < 0.3 * body) & (body / range_hl < 0.3)).astype(int)
    df['SHOOTING_STAR'] = ((upper_shadow > 2 * body) & (lower_shadow < 0.3 * body) & (c < o)).astype(int)
    prev_bearish = (o.shift(1) > c.shift(1))
    curr_bullish = (c > o)
    df['BULLISH_ENGULFING'] = (prev_bearish & curr_bullish & (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    prev_bullish = (c.shift(1) > o.shift(1))
    curr_bearish = (o > c)
    df['BEARISH_ENGULFING'] = (prev_bullish & curr_bearish & (o > c.shift(1)) & (c < o.shift(1))).astype(int)
    first_bearish = (o.shift(2) > c.shift(2))
    second_small = (body.shift(1) / range_hl.shift(1) < 0.3)
    third_bullish = (c > o)
    df['MORNING_STAR'] = (first_bearish & second_small & third_bullish & (c > (o.shift(2) + c.shift(2)) / 2)).astype(int)
    first_bullish_es = (c.shift(2) > o.shift(2))
    second_small_es = (body.shift(1) / range_hl.shift(1) < 0.3)
    third_bearish_es = (o > c)
    df['EVENING_STAR'] = (first_bullish_es & second_small_es & third_bearish_es & (c < (o.shift(2) + c.shift(2)) / 2)).astype(int)
    df['PIERCING'] = ((o.shift(1) > c.shift(1)) & (c > o) & (o < c.shift(1)) & (c > (o.shift(1) + c.shift(1)) / 2) & (c < o.shift(1))).astype(int)
    df['DARK_CLOUD'] = ((c.shift(1) > o.shift(1)) & (o > c) & (o > c.shift(1)) & (c < (o.shift(1) + c.shift(1)) / 2) & (c > o.shift(1))).astype(int)
    return df

def add_indicators(df):
    """Add technical indicators and candlestick patterns to the DataFrame."""
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['RSI_9'] = ta.rsi(df['close'], length=9)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df['RSI_25'] = ta.rsi(df['close'], length=25)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    bbands = ta.bbands(df['close'], length=20, std=2.0)
    if bbands is not None and not bbands.empty:
        bbp_cols = [c for c in bbands.columns if c.startswith('BBP')]
        if bbp_cols:
            df['BBP_5_2.0'] = bbands[bbp_cols[0]]
        else:
            logger.warning("BBP column not found in Bollinger Bands output, using zeros.")
            df['BBP_5_2.0'] = 0
    else:
        logger.warning("Bollinger Bands calculation failed, using zeros.")
        df['BBP_5_2.0'] = 0
    stochrsi = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None and not stochrsi.empty:
        df = pd.concat([df, stochrsi], axis=1)
    else:
        logger.warning("Stochastic RSI calculation failed, using zeros.")
        df['STOCHRSIk_14_14_3_3'] = 0
        df['STOCHRSId_14_14_3_3'] = 0
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df = add_candlestick_patterns(df)
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.debug(f"Dropped {dropped} rows due to NaN values from indicators.")
    if len(df) < 200:
        logger.warning(f"Only {len(df)} rows remaining after indicator calculation. Consider increasing data limit.")
    return df

def train_and_predict(X, y):
    """Train XGBoost model with hyperparameter tuning using provided X, y.
    Handles class imbalance via scale_pos_weight.
    """
    class_counts = y.value_counts()
    logger.info(f"Target distribution after balancing: {class_counts.to_dict()}")
    if class_counts.get(1, 0) > 0:
        scale_pos_weight = class_counts.get(0, 0) / class_counts.get(1, 1)
    else:
        scale_pos_weight = 1.0
    logger.info(f"scale_pos_weight set to {scale_pos_weight:.2f}")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    logger.log(35, "Tuning hyperparameters (this may take a moment)...")
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=15,
        scoring='accuracy',
        n_jobs=-1,
        cv=tscv,
        verbose=1,
        random_state=42
    )
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    logger.log(35, f"Best Parameters: {random_search.best_params_}")
    logger.log(35, f"Best Validation Score: {random_search.best_score_:.2f}")
    return best_model

def predict_next_move(model, last_row):
    """Predict probability for next movement using the trained model."""
    X_new = last_row[FEATURES].values.reshape(1, -1)
    proba = model.predict_proba(X_new)[0]
    return proba

def main():
    symbol = input("Enter symbol (e.g. BTC/USDT): ").strip().upper()
    if not symbol:
        symbol = 'BTC/USDT'
        logger.info(f"No input, using default: {symbol}")
    elif '/' not in symbol:
        symbol += '/USDT'
        logger.info(f"Auto-appended USDT: {symbol}")
    print("\nSelect Timeframe:")
    print("1. 15m (Predict next 6h)")
    print("2. 30m (Predict next 12h)")
    print("3. 1h  (Predict next 24h)")
    print("4. 4h  (Predict next 4 days)")
    print("5. 1d  (Predict next 24 days)")
    tf_choice = input("Enter choice (default 1h): ").strip()
    timeframe_map = {
        '1': '15m', '2': '30m', '3': '1h', '4': '4h', '5': '1d',
        '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h', '1d': '1d'
    }
    timeframe = timeframe_map.get(tf_choice, '1h')
    logger.info(f"Using timeframe: {timeframe}")
    df = fetch_data(symbol, timeframe=timeframe, limit=1500)
    if df is None:
        logger.error("Failed to retrieve data. Exiting.")
        return
    df = add_indicators(df)
    HORIZON = 24
    df['Target'] = (df['close'].shift(-HORIZON) > df['close']).astype(int)
    df.dropna(inplace=True)
    logger.info(f"Prepared training data with {len(df)} rows.")
    X = df[FEATURES]
    y = df['Target']
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    logger.info(f"After SMOTE balancing: class distribution {pd.Series(y_bal).value_counts().to_dict()}")
    model = train_and_predict(X_bal, y_bal)
    latest_row = df.iloc[-1:]
    proba = predict_next_move(model, latest_row)
    prob_down, prob_up = proba[0] * 100, proba[1] * 100
    current_price = latest_row['close'].values[0]
    atr = latest_row['ATR_14'].values[0]
    last_time = pd.to_datetime(latest_row['timestamp'].values[0])
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize('UTC')
    time_diff = datetime.now(timezone.utc) - last_time
    if time_diff.days > 1:
        logger.warning(f"\n[WARNING] Data is stale ({time_diff.days} days old).")
        logger.warning("The coin might be delisted or the exchange is under maintenance.")
        logger.warning("Prediction results are NOT valid for now.")
    logger.info(f"\n{'='*30}")
    logger.info(f"ANALYSIS FOR {symbol} ({timeframe})")
    logger.info(f"Latest Candle Time: {latest_row['timestamp'].values[0]}")
    logger.info(f"Current Price: {Fore.YELLOW}{current_price:.6f}{Style.RESET_ALL}")
    logger.info(f"Market Volatility (ATR): {Fore.YELLOW}{atr:.6f}{Style.RESET_ALL}")
    logger.info(f"{'-'*30}")
    target_dist = 1.5 * atr * np.sqrt(HORIZON)
    if prob_up > prob_down:
        direction = "UP"
        probability = prob_up
        target = current_price + target_dist
    else:
        direction = "DOWN"
        probability = prob_down
        target = current_price - target_dist
    prediction_msg = f"PREDICTION: {direction}"
    prediction_level = 25 if direction == "UP" else 27
    logger.log(prediction_level, prediction_msg)
    logger.info(f"Probability: {probability:.2f}%")
    logger.info(f"Estimated Target ({HORIZON}h): {target:.6f}")
    logger.info(f"Potential Move: {abs(target - current_price):.6f} ({(abs(target - current_price) / current_price * 100):.2f}%)")
    logger.info(f"{'='*30}")

if __name__ == "__main__":
    main()
