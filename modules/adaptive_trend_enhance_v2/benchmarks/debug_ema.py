import numpy as np
import pandas as pd
import pandas_ta as ta


def debug_ema():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    series = pd.Series(prices)
    length = 5

    # Pandas TA
    ta_ema = ta.ema(series, length=length)
    print("Pandas TA EMA:")
    print(ta_ema.values)

    # Pandas EWM (adjust=False) - Rust Logic
    pd_ewm = series.ewm(span=length, adjust=False).mean()
    print("\nPandas EWM (adjust=False):")
    print(pd_ewm.values)

    # Pandas EWM (adjust=True)
    pd_ewm_adj = series.ewm(span=length, adjust=True).mean()
    print("\nPandas EWM (adjust=True):")
    print(pd_ewm_adj.values)

    # Manual Loop (Rust logic simulation)
    alpha = 2.0 / (length + 1.0)
    manual_ema = np.zeros_like(prices)
    manual_ema[0] = prices[0]
    for i in range(1, len(prices)):
        manual_ema[i] = alpha * prices[i] + (1 - alpha) * manual_ema[i - 1]

    print("\nManual Loop (Rust logic):")
    print(manual_ema)


if __name__ == "__main__":
    debug_ema()
