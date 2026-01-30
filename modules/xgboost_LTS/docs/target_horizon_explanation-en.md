# TARGET_HORIZON Explanation

## Definition

`TARGET_HORIZON = 24` (defined in `modules/config.py`)

This is the number of **candles** in the future that the model will predict. The default value is **24 candles**.

## Usage in the System

### 1. **Creating Labels (Labeling) - `modules/labeling.py`**

`TARGET_HORIZON` is used to create target labels based on future prices:

```python
# Get the closing price after TARGET_HORIZON candles
future_close = df["close"].shift(-TARGET_HORIZON)  # shift(-24) = shift back 24 candles

# Calculate percentage price change
pct_change = (future_close - df["close"]) / df["close"]

# Get the closing price from TARGET_HORIZON candles ago to calculate dynamic threshold
historical_ref = df["close"].shift(TARGET_HORIZON)  # shift(24) = shift forward 24 candles
```

**Example with TARGET_HORIZON = 24:**
- At candle #100, the model will look at the price at candle #124 (100 + 24) to create the label
- If the price at candle 124 > price at candle 100 + threshold → Label = "UP"
- If the price at candle 124 < price at candle 100 - threshold → Label = "DOWN"
- Otherwise → Label = "NEUTRAL"

### 2. **Preventing Data Leakage (Model Training) - `modules/model.py`**

`TARGET_HORIZON` is used to create a **gap** between the training set and test set:

```python
# Train/Test split with gap
split = int(len(df) * 0.8)  # 80% of data
train_end = split - TARGET_HORIZON  # End training before TARGET_HORIZON candles
test_start = split  # Start testing after the gap
```

**Why is a gap needed?**

When creating labels, each row in the training set uses the price from **TARGET_HORIZON candles later** to create the label. Without a gap:

```
❌ WRONG (Data Leakage):
Train: [1, 2, 3, ..., 100]
Test:  [101, 102, 103, ...]
→ Row 100 in train uses price from row 124 to create label
→ Rows 101-124 in test were already "seen" when creating label for row 100
→ Model already "knows" the test data → Data Leakage!

✅ CORRECT (With gap):
Train: [1, 2, 3, ..., 76]  (ends at 80 - 24 = 76)
Gap:   [77, 78, 79, ..., 100]  (24 candles gap)
Test:  [101, 102, 103, ...]
→ Row 76 in train uses price from row 100 to create label
→ Rows 101+ in test are completely independent, no leakage
```

### 3. **Cross-Validation with Gap - `modules/model.py`**

In cross-validation, a similar gap is needed:

```python
# Remove the last TARGET_HORIZON indices from the training set
train_idx_filtered = train_idx_array[:-TARGET_HORIZON]

# Ensure test set starts after the gap
min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
```

### 4. **Display in Output - `main.py`**

`TARGET_HORIZON` is displayed in the prediction context:

```python
prediction_context = f"{prediction_window} | {TARGET_HORIZON} candles >={threshold_value*100:.2f}% move"
# Example: "24h | 24 candles >=1.50% move"
```

## Concrete Examples

### With timeframe = "1h" and TARGET_HORIZON = 24:

- Model predicts the price after **24 hours** (24 candles × 1h = 24h)
- At the current moment, the model will predict the price 24 hours from now
- Labels are created by comparing the current price with the price 24 candles later

### With timeframe = "4h" and TARGET_HORIZON = 24:

- Model predicts the price after **96 hours** (24 candles × 4h = 96h = 4 days)
- At the current moment, the model will predict the price 4 days from now

## Impact of Changing TARGET_HORIZON

### Increasing TARGET_HORIZON (e.g., 24 → 48):
- ✅ Predicts further into the future
- ❌ Requires more data (loses an additional 48 rows due to gap)
- ❌ Labels may be less accurate (predicting further = harder)
- ❌ Loses more data at the end of dataset (cannot create labels for the last 48 candles)

### Decreasing TARGET_HORIZON (e.g., 24 → 12):
- ✅ Requires less data
- ✅ Closer predictions (easier to predict)
- ❌ Shorter-term predictions
- ✅ Loses less data (only loses the last 12 rows)

## Recommendations

- **TARGET_HORIZON = 24** is a reasonable value for most timeframes
- For short timeframes (30m, 1h): can keep 24 or increase to 48
- For long timeframes (4h, 1d): can decrease to 12 or 6
- Ensure sufficient data: need at least `TARGET_HORIZON * 2 + 200` rows to have enough data after creating gap and calculating indicators

## Formula for Minimum Data Required

```
Minimum rows needed = TARGET_HORIZON + TARGET_HORIZON + 200
                    = 2 * TARGET_HORIZON + 200
                    = Gap + Training data + Indicators requirement
```

With `TARGET_HORIZON = 24`:
- Minimum = 2 × 24 + 200 = **248 rows**
- Recommended: **500+ rows** for good training data
