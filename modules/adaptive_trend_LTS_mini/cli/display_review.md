 Code Review: modules/adaptive_trend_LTS_mini/cli/display.py

  Overview

  This module provides CLI display utilities for the Adaptive Trend Component (ATC) system. It handles formatting and rendering of trading signals, MA (Moving Average) indicators, equity weights, and scan results using colored terminal output via Colorama.

  Code Quality & Style

  Strengths:
  - ✅ Well-structured with clear separation of concerns (private helper functions)
  - ✅ Comprehensive docstrings following NumPy/Google style
  - ✅ Type hints for all function signatures (modern Python 3.10+ syntax with tuple[str, str])
  - ✅ Consistent naming conventions and code formatting
  - ✅ Good use of constants for display widths (maintainability)
  - ✅ Defensive programming with empty series checks

  Style Issues:
  - ⚠️ Line 47: Modern type hint syntax tuple[str, str] requires Python 3.9+, consistent with project requirements
  - ⚠️ Minor inconsistency: Some functions have blank lines between sections, others don't

  Specific Suggestions for Improvements

  1. Code Duplication (DRY Principle)

  Issue: Lines 80-88 duplicate the logic from _get_trend_direction() (lines 47-61)

  # Lines 80-88 duplicate trend direction logic
  if ma_trend_value > 0:
      ma_color = Fore.GREEN
      ma_dir = "^"
  elif ma_trend_value < 0:
      ma_color = Fore.RED
      ma_dir = "v"
  else:
      ma_color = Fore.YELLOW
      ma_dir = "-"

  Recommendation: Extend _get_trend_direction() to return direction symbol:

  def _get_trend_direction(trend_value: float) -> tuple[str, str, str]:
      """Get trend direction label, color, and symbol based on trend value."""
      if trend_value > 0:
          return "BULLISH", Fore.GREEN, "^"
      elif trend_value < 0:
          return "BEARISH", Fore.RED, "v"
      else:
          return "NEUTRAL", Fore.YELLOW, "-"

  Then refactor line 77-88 in _display_ma_signals:

  ma_dir, ma_color, ma_symbol = _get_trend_direction(ma_trend_value)
  print(color_text(f"  {ma_name:6s}: {latest_ma_sig:8.4f} {ma_symbol}", ma_color))

  2. Repeated DataFrame Iteration Pattern

  Issue: Lines 261-270 and 292-301 have nearly identical iteration logic for displaying signal rows.

  Recommendation: Extract into a helper function:

  def _display_signal_rows(signals: pd.DataFrame, color: str) -> None:
      """Display signal rows with consistent formatting."""
      for _, row in signals.iterrows():
          signal_str = f"{row['signal']:+.6f}"
          price_str = format_price(row["price"])
          row_text = (
              f"{row['symbol']:<{COL_SYMBOL_WIDTH}} "
              f"{signal_str:>{COL_SIGNAL_WIDTH}} "
              f"{price_str:>{COL_PRICE_WIDTH}} "
              f"{row['exchange']:<{COL_EXCHANGE_WIDTH}}"
          )
          print(color_text(row_text, color))

  3. Magic Numbers and Hardcoded Values

  Issue:
  - Line 340: cols = 4 is hardcoded
  - Line 262, 293: Format string +.6f for signals (why 6 decimals?)
  - Line 92: Format string 8.4f for MA signals

  Recommendation: Define as module-level constants:

  # Display formatting constants
  DISPLAY_WIDTH = 80
  COL_SYMBOL_WIDTH = 15
  COL_SIGNAL_WIDTH = 12
  COL_PRICE_WIDTH = 15
  COL_EXCHANGE_WIDTH = 10
  SIGNAL_DECIMAL_PLACES = 6
  MA_SIGNAL_DECIMAL_PLACES = 4
  SYMBOLS_PER_ROW = 4

  4. Error Handling in list_futures_symbols

  Issue: Lines 348-352 catch specific exceptions but then raise on generic Exception, which could expose stack traces to CLI users.

  Recommendation:
  - Log all exceptions consistently without re-raising (CLI context)
  - Or remove the generic exception handler if you want to let errors bubble up

  except Exception as e:
      log_error(f"Unexpected error listing symbols: {type(e).__name__}: {e}")
      # Don't re-raise in CLI context - user sees the error message

  5. Unused Import

  Issue: Line 30 imports DataFetcher but it's only used as a type hint in line 314. Consider using string annotations if not needed at runtime.

  Minor: This is actually fine for type checking, but could use from __future__ import annotations for consistency.

  6. Repeated Header Display Pattern

  Issue: Lines 252-259 and 283-290 display identical table headers.

  Recommendation: Extract header display:

  def _display_signal_table_header() -> None:
      """Display signal table header."""
      header = (
          f"{'Symbol':<{COL_SYMBOL_WIDTH}} "
          f"{'Signal':>{COL_SIGNAL_WIDTH}} "
          f"{'Price':>{COL_PRICE_WIDTH}} "
          f"{'Exchange':<{COL_EXCHANGE_WIDTH}}"
      )
      print(color_text(header, Fore.MAGENTA))
      print(color_text("-" * DISPLAY_WIDTH, Fore.CYAN))

  7. Type Safety

  Issue: atc_results.get() returns Any - consider defining a TypedDict or dataclass for ATC results structure.

  Recommendation:

  from typing import TypedDict

  class ATCResults(TypedDict, total=False):
      Average_Signal: pd.Series
      EMA_Signal: pd.Series
      HMA_Signal: pd.Series
      # ... etc

  Potential Issues & Risks

  1. Performance

  - ⚠️ Line 149: trend_sign(average_signal) calculates trend every time - if this is expensive and called frequently, consider caching
  - ⚠️ Line 77: Same for individual MA trends in loops

  2. Empty Series Handling

  - ✅ Good: Empty checks before .iloc[-1] access
  - ⚠️ Line 150: latest_trend.empty check but still accesses .iloc[-1] - could use ternary consistently

  Safer pattern:
  latest_trend_value = latest_trend.iloc[-1] if len(latest_trend) > 0 else 0

  3. Data Validation

  - No validation that atc_results dictionary contains expected keys with correct types
  - current_price could be 0, negative, or NaN - no validation

  4. Display Width Overflow

  - Fixed width formatting could break if symbol names exceed COL_SYMBOL_WIDTH (15 chars)
  - Consider using .ljust() or textwrap for truncation

  Testing Considerations

  Missing test coverage for:
  - Edge cases: empty DataFrames, None values, NaN prices
  - Display formatting with very long symbol names
  - Colorama output (could use snapshot testing)
  - Error handling paths in list_futures_symbols

  Security

  ✅ No major security concerns - this is display-only code with no external I/O or user input processing beyond function parameters.

  Summary

  Overall: Well-written, maintainable code with good documentation. Main improvements would be reducing code duplication and extracting repeated patterns. The code follows project conventions from CLAUDE.md (PEP 8, type hints, docstrings).

  Priority fixes:
  1. High: Refactor duplicate trend direction logic
  2. Medium: Extract repeated display patterns
  3. Low: Move magic numbers to constants

  Estimated effort: 30-45 minutes to implement all suggestions.