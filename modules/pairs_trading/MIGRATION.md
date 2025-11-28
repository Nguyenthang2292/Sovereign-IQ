# Migration Guide: Pairs Trading Module Refactoring

## Overview

The pairs trading module has been refactored from a flat structure to a well-organized package hierarchy with clear separation of concerns.

## What Changed

### Directory Structure

**Before:**
```
pairs_trading/
├── __init__.py
├── pairs_analyzer.py
├── pair_metrics_computer.py
├── opportunity_scorer.py
├── performance_analyzer.py
├── statistical_tests.py
├── hedge_ratio.py
├── zscore_metrics.py
├── risk_metrics.py
├── display.py
├── utils.py
├── cli.py
├── prompts.py
└── parsers.py
```

**After:**
```
pairs_trading/
├── README.md                    # NEW: Comprehensive documentation
├── __init__.py                  # UPDATED: New import structure
├── core/                        # Core analysis components
│   ├── pairs_analyzer.py
│   ├── pair_metrics_computer.py
│   └── opportunity_scorer.py
├── metrics/                     # Statistical and quantitative metrics
│   ├── statistical_tests.py
│   ├── hedge_ratio.py
│   ├── zscore_metrics.py
│   └── risk_metrics.py
├── analysis/                    # Performance analysis
│   └── performance_analyzer.py
├── utils/                       # NEW: Business logic utilities
│   ├── pair_selector.py         # RENAMED from pair_selection.py
│   ├── pair_transformer.py      # RENAMED from pair_manipulation.py
│   └── candidate_manager.py     # RENAMED from candidate_pools.py
└── cli/                         # RENAMED from ui/
    ├── argument_parser.py       # RENAMED from cli.py
    ├── interactive_prompts.py   # RENAMED from prompts.py
    ├── input_parsers.py         # RENAMED from parsers.py
    └── formatters/              # NEW: Display formatters sub-package
        ├── performance_formatter.py  # SPLIT from display.py
        └── pairs_formatter.py        # SPLIT from display.py
```

### File Renames

| Old Location | New Location | Reason |
|--------------|--------------|--------|
| `ui/` | `cli/` | More accurate - it's CLI not UI |
| `ui/cli.py` | `cli/argument_parser.py` | Descriptive name |
| `ui/prompts.py` | `cli/interactive_prompts.py` | More specific |
| `ui/parsers.py` | `cli/input_parsers.py` | More specific |
| `ui/display.py` | `cli/formatters/` (split) | Better organization |
| `ui/pair_selection.py` | `utils/pair_selector.py` | Singular, consistent |
| `ui/pair_manipulation.py` | `utils/pair_transformer.py` | Clearer intent |
| `ui/candidate_pools.py` | `utils/candidate_manager.py` | More descriptive |

### Import Changes

**Old imports:**
```python
from modules.pairs_trading.ui import (
    display_performers,
    select_top_unique_pairs,
    reverse_pairs,
)
```

**New imports:**
```python
# Display functions moved to cli.formatters
from modules.pairs_trading.cli import (
    display_performers,
    display_pairs_opportunities,
)

# Utility functions moved to utils
from modules.pairs_trading.utils import (
    select_top_unique_pairs,
    select_pairs_for_symbols,
    reverse_pairs,
    ensure_symbols_in_candidate_pools,
)
```

**However, for backward compatibility, you can still use:**
```python
# This still works! (imports from main __init__.py)
from modules.pairs_trading import (
    display_performers,
    select_top_unique_pairs,
    reverse_pairs,
)
```

## Migration Steps

### Option 1: No Changes Required (Recommended)

The main `__init__.py` re-exports everything, so existing code should work without changes:

```python
# This still works!
from modules.pairs_trading import (
    PairsTradingAnalyzer,
    display_performers,
    select_top_unique_pairs,
    reverse_pairs,
    parse_args,
)
```

### Option 2: Update to New Structure (Optional)

If you want to use the new structure explicitly:

**Before:**
```python
from modules.pairs_trading.ui import display_performers
from modules.pairs_trading.ui import select_top_unique_pairs
from modules.pairs_trading.ui import reverse_pairs
```

**After:**
```python
from modules.pairs_trading.cli import display_performers
from modules.pairs_trading.utils import select_top_unique_pairs
from modules.pairs_trading.utils import reverse_pairs
```

## Benefits of New Structure

### 1. Clear Separation of Concerns
- **core/**: Core business logic
- **metrics/**: Statistical calculations
- **analysis/**: Performance analysis
- **utils/**: Utility functions
- **cli/**: User interface components

### 2. Better Scalability
- Easy to add new metrics in `metrics/`
- Easy to add new formatters in `cli/formatters/`
- Easy to add new utilities in `utils/`

### 3. Improved Maintainability
- Smaller, focused files
- Clear responsibilities
- Better organization

### 4. Professional Structure
- Follows Python best practices
- Similar to popular libraries (sklearn, pandas, etc.)
- Easy for new developers to understand

## Deprecated Files (Old ui/ directory)

The old `ui/` directory is still present but should not be used. It will be removed in a future version.

**Files to avoid:**
- `modules.pairs_trading.ui.*` (use `cli` or `utils` instead)

## Testing

After migration, test your code:

```python
# Test imports
from modules.pairs_trading import (
    PairsTradingAnalyzer,
    display_performers,
    select_top_unique_pairs,
)

# Test functionality
analyzer = PairsTradingAnalyzer()
# ... your code ...
```

## Rollback Plan

If you encounter issues, you can temporarily revert by:

1. Keep using old imports from `ui/` package
2. The old files are still present
3. Report issues to the development team

## Questions?

Refer to:
- `README.md` - Comprehensive module documentation
- `__init__.py` - See all available exports
- Individual package `__init__.py` files for sub-package exports

## Timeline

- **Now**: New structure available, old structure deprecated
- **Next release**: Old `ui/` directory will be removed
- **Action required**: Update imports if using explicit `ui` imports

## Summary

✅ **No immediate action required** - backward compatibility maintained
✅ **New structure available** - use for new code
✅ **Better organization** - easier to maintain and extend
✅ **Comprehensive docs** - see README.md

The refactoring improves code organization without breaking existing functionality!
