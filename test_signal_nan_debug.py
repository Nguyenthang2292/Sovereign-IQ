"""Debug script to test signal generation with NaN values."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.signal_detection import (
    crossover,
    crossunder,
    generate_signal_from_ma,
)

# Create test data
np.random.seed(42)
bars = 100
price = pd.Series(np.random.randn(bars).cumsum() + 100, name="close")

# Create MA with NaN at start (realistic scenario)
ma = price.rolling(window=20).mean()

print("=" * 60)
print("DEBUG: Signal Generation with NaN Values")
print("=" * 60)
print(f"\nPrice length: {len(price)}")
print(f"MA length: {len(ma)}")
print(f"Price NaN count: {price.isna().sum()}")
print(f"MA NaN count: {ma.isna().sum()}")
print(f"MA first 25 values:\n{ma.head(25)}")

print("\n" + "=" * 60)
print("Testing crossover detection...")
print("=" * 60)
up = crossover(price, ma)
print(f"Crossover result length: {len(up)}")
print(f"Crossover True count: {up.sum()}")
print(f"Crossover result (first 30):\n{up.head(30)}")

print("\n" + "=" * 60)
print("Testing crossunder detection...")
print("=" * 60)
down = crossunder(price, ma)
print(f"Crossunder result length: {len(down)}")
print(f"Crossunder True count: {down.sum()}")
print(f"Crossunder result (first 30):\n{down.head(30)}")

print("\n" + "=" * 60)
print("Testing signal generation...")
print("=" * 60)
signal = generate_signal_from_ma(price, ma)
print(f"Signal length: {len(signal)}")
print(f"Signal dtype: {signal.dtype}")
print(f"Signal value counts:\n{signal.value_counts()}")
print(f"Signal first 30 values:\n{signal.head(30)}")
print(f"Signal last 30 values:\n{signal.tail(30)}")

# Check if all signals are 0
if (signal == 0).all():
    print("\n⚠️ WARNING: ALL SIGNALS ARE 0!")
    print("This indicates a regression in signal generation logic.")
else:
    print(f"\n✅ Signals generated successfully:")
    print(f"   - Long signals (1): {(signal == 1).sum()}")
    print(f"   - Short signals (-1): {(signal == -1).sum()}")
    print(f"   - Neutral signals (0): {(signal == 0).sum()}")
