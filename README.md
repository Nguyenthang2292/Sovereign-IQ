# Crypto Prediction AI

This project uses Machine Learning (XGBoost) to predict the next movement of cryptocurrency pairs.

## Features

-   **Multi-Exchange Support**: Automatically fetches data from Binance, Kraken, KuCoin, Gate.io, etc.
-   **Smart Fallback**: If data is stale (e.g., delisted coin), it switches to another exchange.
-   **Advanced Indicators**: Uses SMA, RSI, ATR, MACD, Bollinger Bands, Stochastic RSI, and OBV.
-   **Auto-Tuning**: Automatically finds the best hyperparameters for each coin using `RandomizedSearchCV`.

## Installation

1.  Install Python 3.8+.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script:
```bash
python crypto_prediction.py
```

Enter the symbol (e.g., `BTC/USDT`) and select the timeframe when prompted.
