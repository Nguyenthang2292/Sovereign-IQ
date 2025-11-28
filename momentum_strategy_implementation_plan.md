# Implementation Plan: Long-Short Momentum Strategy

## 1. Objective
Extend the current Pairs Trading system to support a **Long-Short Momentum** strategy alongside the existing **Mean Reversion** strategy.

- **Mean Reversion (Current)**: Expects divergence to converge. Longs the underperformer (Worst), Shorts the overperformer (Best).
- **Momentum (New)**: Expects divergence to continue. Longs the overperformer (Best), Shorts the underperformer (Worst).

## 2. Implementation Steps

### Step 1: Modify `main_pairs_trading.py`

#### A. Add CLI Argument
Update `parse_args()` to include a new `--strategy` parameter.
- **Flag**: `--strategy`
- **Choices**: `['reversion', 'momentum']`
- **Default**: `'reversion'`
- **Description**: Choose trading logic: 'reversion' for mean reversion (default), 'momentum' for trend following.

#### B. Update Main Logic (`main` function)
Refactor the candidate selection section (Step 3 & 4) to handle the strategy switch.

**Logic Mapping:**

| Strategy | Long Candidate | Short Candidate | Logic |
| :--- | :--- | :--- | :--- |
| **Reversion** | `worst_performers` | `best_performers` | Bet on spread narrowing (Convergence) |
| **Momentum** | `best_performers` | `worst_performers` | Bet on spread widening (Divergence) |

**Code Changes:**
1.  **Switch Candidates**:
    ```python
    if args.strategy == "momentum":
        long_candidates = best_performers  # Buy the winners
        short_candidates = worst_performers # Sell the losers
        
        # Momentum doesn't strictly require cointegration (we want them to drift apart)
        # But we might still want correlation to hedge market risk.
        # For now, we can disable strict cointegration checks or make them optional.
        args.require_cointegration = False 
    else:
        long_candidates = worst_performers # Buy the losers (dip)
        short_candidates = best_performers # Sell the winners (peak)
    ```

2.  **Update Display Labels**:
    - Update `display_performers` calls to show dynamic titles based on strategy (e.g., "LONG CANDIDATES (Momentum Leaders)" vs "LONG CANDIDATES (Weak performers)").

### Step 2: Adjust `PairsTradingAnalyzer` Interaction

The `analyze_pairs_opportunity` method typically takes `best` and `worst` dataframes. We need to ensure we pass the correct "Long side" and "Short side" to it.

- **Current Signature**: `analyze_pairs_opportunity(best_performers, worst_performers, ...)`
- **Action**: It might be clearer to rename arguments or ensure the caller maps them correctly.
    - If `strategy == momentum`: Call with `long_side=best`, `short_side=worst`.
    - If `strategy == reversion`: Call with `long_side=worst`, `short_side=best`.

### Step 3: Validation & Testing

1.  **Run Reversion (Default)**:
    - Command: `python main_pairs_trading.py`
    - Verify: Longs are negative score (Red), Shorts are positive score (Green).

2.  **Run Momentum**:
    - Command: `python main_pairs_trading.py --strategy momentum`
    - Verify: Longs are positive score (Green), Shorts are negative score (Red).

## 3. Future Enhancements (Optional)

- **Trend Metrics**: For Momentum, consider adding ADX or Slope filters to ensure the "Best" are trending strongly, not just overbought.
- **Stop Loss**: Momentum strategies are susceptible to sharp reversals. Implement tighter stop-loss logic based on ATR.
