# Đề Xuất Cải Tiến Hệ Thống Scoring HMM

## Tổng Quan

Tài liệu này mô tả các cải tiến đề xuất cho hệ thống scoring trong `modules/hmm/signal_combiner.py`.

## Vấn Đề Hiện Tại

1. **High-Order HMM chỉ dùng threshold đơn giản**: Không có scoring system, chỉ check `probability >= threshold`
2. **Scoring không có confidence weighting**: Tất cả signals đều có weight cố định
3. **Không có normalization**: Scores có thể khác nhau về scale
4. **Không có conflict resolution**: Khi 2 signals mâu thuẫn
5. **Không có combined confidence score**: Không có cách đánh giá tổng thể

## Đề Xuất Cải Tiến

### 1. Thêm Scoring System cho High-Order HMM

**Hiện tại:**
```python
signal_high_order_hmm: Signal = HOLD
if probability >= HMM_PROBABILITY_THRESHOLD:
    if next_state == -1:
        signal_high_order_hmm = SHORT
    elif next_state == 1:
        signal_high_order_hmm = LONG
```

**Cải tiến:**
```python
# Tính score dựa trên probability và state strength
high_order_score_long = 0.0
high_order_score_short = 0.0

if probability >= HMM_PROBABILITY_THRESHOLD:
    # Score = probability * state_strength_multiplier
    if next_state == -1:  # BEARISH
        high_order_score_short = probability * HMM_HIGH_ORDER_BEARISH_STRENGTH
    elif next_state == 1:  # BULLISH
        high_order_score_long = probability * HMM_HIGH_ORDER_BULLISH_STRENGTH
    # next_state == 0 (NEUTRAL): không cộng score
```

### 2. Confidence-Weighted Scoring

**Hiện tại:**
```python
if primary_state in {1, 3}:
    score_long += HMM_SIGNAL_PRIMARY_WEIGHT  # Fixed weight
```

**Cải tiến:**
```python
# Tính confidence từ các indicators
confidence = _calculate_model_confidence(hmm_kama_result, high_order_hmm_result)

# Apply confidence multiplier
if primary_state in {1, 3}:
    score_long += HMM_SIGNAL_PRIMARY_WEIGHT * confidence
```

### 3. Normalized Scoring

**Cải tiến:**
```python
# Normalize scores về range 0-100
max_possible_score = (
    HMM_SIGNAL_PRIMARY_WEIGHT +
    (HMM_SIGNAL_TRANSITION_WEIGHT * 3) +
    (HMM_SIGNAL_ARM_WEIGHT * 2) +
    HMM_HIGH_ORDER_MAX_SCORE
)

normalized_score_long = (score_long / max_possible_score) * 100
normalized_score_short = (score_short / max_possible_score) * 100
```

### 4. Conflict Resolution

**Cải tiến:**
```python
# Khi 2 signals mâu thuẫn
if signal_high_order_hmm != signal_hmm_kama and signal_high_order_hmm != HOLD:
    # Ưu tiên signal có confidence cao hơn
    high_order_confidence = probability
    kama_confidence = _calculate_kama_confidence(score_long, score_short)
    
    if high_order_confidence > kama_confidence * HMM_CONFLICT_RESOLUTION_THRESHOLD:
        # High-Order HMM có confidence cao hơn, giữ nguyên
        pass
    else:
        # KAMA có confidence cao hơn, downgrade High-Order signal
        signal_high_order_hmm = HOLD
```

### 5. Combined Confidence Score

**Cải tiến:**
```python
def _calculate_combined_confidence(
    high_order_prob: float,
    kama_scores: Tuple[int, int],
    signal_agreement: bool
) -> float:
    """Calculate combined confidence from both models."""
    # Base confidence từ probability
    base_confidence = high_order_prob
    
    # KAMA confidence từ score ratio
    total_kama_score = kama_scores[0] + kama_scores[1]
    kama_confidence = (
        max(kama_scores) / total_kama_score if total_kama_score > 0 else 0.5
    )
    
    # Agreement bonus: nếu 2 signals đồng ý, tăng confidence
    agreement_bonus = 1.2 if signal_agreement else 1.0
    
    # Weighted average
    combined = (base_confidence * 0.4 + kama_confidence * 0.6) * agreement_bonus
    return min(combined, 1.0)  # Cap at 1.0
```

### 6. Dynamic Threshold Adjustment

**Cải tiến:**
```python
# Điều chỉnh threshold dựa trên market volatility
volatility = df['close'].pct_change().std()
if volatility > HMM_HIGH_VOLATILITY_THRESHOLD:
    # Market biến động cao, tăng threshold (conservative)
    adjusted_threshold = HMM_SIGNAL_MIN_THRESHOLD * 1.2
else:
    adjusted_threshold = HMM_SIGNAL_MIN_THRESHOLD
```

### 7. State Strength Multipliers

**Cải tiến:**
```python
# Primary state có strength khác nhau
STATE_STRENGTH_MULTIPLIERS = {
    0: 1.0,  # Bearish strong
    1: 0.7,  # Bearish weak
    2: 0.7,  # Bullish weak
    3: 1.0,  # Bullish strong
}

if primary_state in {1, 3}:
    strength = STATE_STRENGTH_MULTIPLIERS.get(primary_state, 1.0)
    score_long += HMM_SIGNAL_PRIMARY_WEIGHT * strength
```

## Constants Cần Thêm vào config.py

```python
# High-Order HMM Scoring
HMM_HIGH_ORDER_BEARISH_STRENGTH = 1.0
HMM_HIGH_ORDER_BULLISH_STRENGTH = 1.0
HMM_HIGH_ORDER_MAX_SCORE = 1.0  # Max score từ High-Order HMM

# Confidence & Conflict Resolution
HMM_CONFLICT_RESOLUTION_THRESHOLD = 1.2  # Ratio để ưu tiên High-Order
HMM_HIGH_VOLATILITY_THRESHOLD = 0.03  # Volatility threshold
HMM_VOLATILITY_ADJUSTMENT_FACTOR = 1.2  # Multiplier cho high volatility

# State Strength Multipliers
HMM_STATE_STRENGTH_STRONG = 1.0
HMM_STATE_STRENGTH_WEAK = 0.7
```

## Lợi Ích

1. **Chính xác hơn**: Confidence weighting giúp đánh giá tốt hơn
2. **Linh hoạt hơn**: Dynamic threshold adapt với market conditions
3. **Rõ ràng hơn**: Normalized scores dễ so sánh và hiểu
4. **Ổn định hơn**: Conflict resolution tránh signals mâu thuẫn
5. **Toàn diện hơn**: Combined confidence cho overview tổng thể

## Implementation Priority

1. **High Priority**: 
   - Confidence-weighted scoring
   - Normalized scoring
   - Combined confidence score

2. **Medium Priority**:
   - High-Order HMM scoring system
   - Conflict resolution

3. **Low Priority**:
   - Dynamic threshold adjustment
   - State strength multipliers

