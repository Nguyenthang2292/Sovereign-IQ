# Code Review: signal_selector.py

**Date**: 2026-02-01
**Reviewer**: Claude Code
**File**: `modules/auto_trade/core/signal_selector.py`

## Overview

This module aggregates trading signals from multiple sources (XGBoost and Gemini) and selects the optimal trade setup based on weighted confidence scoring. The code is well-structured with clear separation between data models and selection logic.

## Code Quality Analysis

### ✅ Strengths

1. **Clean Architecture**: Good use of dataclasses and type hints
2. **Clear Documentation**: Well-written docstrings explain purpose and parameters
3. **Conflict Resolution**: Implements direction conflict detection (lines 111-119)
4. **Configurable Weights**: Supports external configuration for scoring weights
5. **Modular Design**: Separate evaluation logic in `_evaluate_candidate`

### ⚠️ Issues & Suggestions

#### 1. **Price Level Validation** (Critical)
**Location**: Lines 98-109, 131-134

```python
entry = 0.0
tp = 0.0
sl = 0.0
```

**Issue**: Allows zero values for critical price levels, relying on downstream execution phase to calculate them.

**Suggestion**:
```python
# Add validation after price extraction
if entry == 0.0 or tp == 0.0 or sl == 0.0:
    log_warn(
        f"Missing price levels for {xb_signal.symbol}. "
        f"Entry: {entry}, TP: {tp}, SL: {sl}"
    )
    # Option 1: Discard the signal
    return None
    # Option 2: Calculate fallback levels here
```

#### 2. **Leverage Validation** (High Priority)
**Location**: Line 26

```python
leverage: int = 2
```

**Issue**: No validation for leverage values. Could accept dangerous values (0, negative, or too high).

**Suggestion**:
```python
@dataclass
class FinalSignal:
    # ... other fields ...
    leverage: int = 2

    def __post_init__(self):
        if not (1 <= self.leverage <= 10):  # Adjust max based on risk policy
            raise ValueError(f"Invalid leverage: {self.leverage}. Must be 1-10")
        # Validate price relationships
        if self.signal_type == "LONG":
            if not (self.stop_loss < self.entry < self.take_profit):
                raise ValueError("Invalid LONG price levels")
        elif self.signal_type == "SHORT":
            if not (self.take_profit < self.entry < self.stop_loss):
                raise ValueError("Invalid SHORT price levels")
```

#### 3. **Configuration Hardcoding** (Medium Priority)
**Location**: Lines 38-40

```python
self.weight_xgboost = self.config.get("weight_xgboost", 0.4)
self.weight_gemini = self.config.get("weight_gemini", 0.6)
self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.7)
```

**Issue**: Magic numbers scattered in code. Should reference centralized config.

**Suggestion**: Move defaults to `config/` directory following project conventions:
```python
# In config/auto_trade.py
SIGNAL_SELECTOR_DEFAULTS = {
    "weight_xgboost": 0.4,
    "weight_gemini": 0.6,
    "min_confidence_threshold": 0.7,
}
```

#### 4. **Error Handling** (Medium Priority)
**Location**: Lines 89-92

```python
try:
    xb_conf = float(xb_signal.details.get("xgboost_conf", 0.0))
except (ValueError, TypeError):
    xb_conf = 0.0
```

**Issue**: Silent failure. If parsing fails, using 0.0 confidence could hide data quality issues.

**Suggestion**:
```python
try:
    xb_conf = float(xb_signal.details.get("xgboost_conf", 0.0))
except (ValueError, TypeError) as e:
    log_warn(
        f"Failed to parse XGBoost confidence for {xb_signal.symbol}: {e}. "
        f"Details: {xb_signal.details}"
    )
    xb_conf = 0.0
```

#### 5. **Confidence Normalization Logic** (Low Priority)
**Location**: Lines 124-127

```python
if gemini_signal:
    final_conf = (xb_conf * self.weight_xgboost) + (gemini_conf * self.weight_gemini)
    # Normalize to 0-1 range roughly, ensuring we don't exceed 1.0
    final_conf = min(1.0, final_conf)
```

**Issue**: Comment says "normalize" but it's just capping. If weights don't sum to 1.0, scores aren't truly normalized.

**Suggestion**:
```python
# Ensure weights are normalized
total_weight = self.weight_xgboost + self.weight_gemini
if gemini_signal:
    final_conf = (
        (xb_conf * self.weight_xgboost) +
        (gemini_conf * self.weight_gemini)
    ) / total_weight
else:
    final_conf = xb_conf
```

#### 6. **Type Safety** (Low Priority)
**Location**: Line 28, 143-147

```python
sources: Dict[str, Any] = field(default_factory=dict)
```

**Issue**: Using `Any` reduces type safety.

**Suggestion**:
```python
from typing import TypedDict

class SignalSources(TypedDict, total=False):
    xgboost_score: float
    gemini_score: float
    gemini_reasoning: str

@dataclass
class FinalSignal:
    sources: SignalSources = field(default_factory=dict)
```

#### 7. **Missing Risk Metrics** (Enhancement)
**Location**: Throughout module

**Suggestion**: Add risk/reward ratio calculation:
```python
def _calculate_risk_reward_ratio(self, signal: FinalSignal) -> float:
    """Calculate risk/reward ratio for the signal."""
    if signal.signal_type == "LONG":
        risk = signal.entry - signal.stop_loss
        reward = signal.take_profit - signal.entry
    else:  # SHORT
        risk = signal.stop_loss - signal.entry
        reward = signal.entry - signal.take_profit

    return reward / risk if risk > 0 else 0.0
```

## Security Considerations

✅ No direct security risks identified
✅ No SQL injection or XSS vectors
✅ Proper use of type hints helps prevent type-related bugs

## Performance Implications

✅ Efficient O(n log n) sorting for candidate selection
✅ No obvious performance bottlenecks
⚠️ Consider adding caching if Gemini API calls are expensive

## Test Coverage

**Missing Test Cases**:
1. Conflict scenarios (XGBoost LONG vs Gemini SHORT)
2. Missing Gemini data handling
3. Zero/invalid price levels
4. Edge cases: empty signal lists, all candidates below threshold
5. Weight configuration validation

## Recommendations Summary

### Priority 1 (Critical):
- Add price level validation (don't allow zeros for critical fields)
- Implement leverage range validation
- Add `__post_init__` validation for FinalSignal

### Priority 2 (High):
- Move default configs to centralized config file
- Enhance error logging with context
- Add risk/reward ratio calculation
- Implement proper weight normalization

### Priority 3 (Nice to Have):
- Improve type safety with TypedDict
- Add comprehensive unit tests
- Document the "Phase 3 fallback" mentioned in comments

## Overall Assessment

**Score: 7.5/10**

The code is well-structured and follows good practices, but has critical gaps in validation that could lead to invalid trade signals being executed. The reliance on downstream phases for price calculation adds unnecessary coupling and risk.

## Next Steps

1. Implement Priority 1 recommendations immediately before production use
2. Create comprehensive test suite covering edge cases
3. Add integration tests with actual XGBoost and Gemini signal data
4. Document the complete signal selection pipeline including Phase 3 fallback behavior
