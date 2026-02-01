# Code Review: interactive_prompts.py

Overview

This module provides interactive CLI prompts for the Adaptive Trend Classification (ATC) system.
It handles user input for timeframe selection and analysis mode configuration with validation,
colored output, and graceful error handling.

Score: 8.5/10 ✅ Good Quality

Strengths

✅ Excellent Structure & Design

1. Clean Separation of Concerns
    - Private helper functions (_validate_input_length,_find_timeframe_index,
      _prompt_custom_timeframe,_display_timeframe_menu)
    - Public API functions (prompt_timeframe, prompt_interactive_mode)
    - Clear function boundaries and responsibilities
2. Strong Type Safety
    - Uses TypedDict for return types (InteractiveModeResult)
    - Type hints on all function parameters and returns
    - Custom exception class for user exit (UserExitRequested)
3. Comprehensive Documentation
    - Module-level docstring explains purpose and key functionalities
    - All functions have detailed docstrings with Args/Returns sections
    - Clear inline comments for complex logic
4. Good Input Validation
    - Length validation to prevent memory exhaustion
    - Format validation using regex patterns
    - Numeric input validation before conversion
    - Graceful error handling with user-friendly messages
5. User Experience
    - Colorized output using colorama (Fore.CYAN, Fore.MAGENTA, Style.BRIGHT)
    - Default values clearly highlighted
    - Helpful error messages with format examples
    - Consistent prompt formatting with constants (PROMPT_DISPLAY_WIDTH)
6. Robust Error Handling
    - Try-except blocks for import errors (DEFAULT_TIMEFRAME fallback)
    - Validation errors with helpful retry loops
    - Custom exception for graceful exit

Issues & Suggestions

✅ Medium Priority

1. ✅ Inconsistent Return Type in _prompt_custom_timeframe (Line 80-115) - **COMPLETED**

**Status**: Fixed - Added MAX_INPUT_ATTEMPTS loop with retry limit (lines 75-97 in interactive_prompts.py)

**Issue**: Function signature says it returns str but has an implicit None return path if the loop never returns.

**Recommendation**: Add explicit exit condition or document that it's guaranteed to return:

  def _prompt_custom_timeframe(default_timeframe: str) -> str:
      """Prompt for custom timeframe with validation.

      Note: This function loops until valid input is provided.
      Users can press Ctrl+C to exit.

      Args:
          default_timeframe: Default timeframe if user enters empty input

      Returns:
          Validated and normalized timeframe string
      """

2. ✅ Magic Numbers for Menu Options (Lines 167-168) - **COMPLETED**

**Status**: Fixed - Using descriptive variable names `num_tf`, `custom_opt`, `def_opt` (lines 128-130 in interactive_prompts.py)

Issue: Menu option numbers are hardcoded, making maintenance difficult if timeframes list changes.

  CUSTOM_TIMEFRAME_OPTION = len(timeframes) + 1
  DEFAULT_TIMEFRAME_OPTION = len(timeframes) + 2

  Better approach: Already good! But could be more explicit:

# Menu options (calculated from timeframe count)

NUM_TIMEFRAME_OPTIONS = len(timeframes)
CUSTOM_TIMEFRAME_OPTION = NUM_TIMEFRAME_OPTIONS + 1
DEFAULT_TIMEFRAME_OPTION = NUM_TIMEFRAME_OPTIONS + 2

1. ✅ Potential Infinite Loop Risk (Lines 176-204) - **COMPLETED**

**Status**: Fixed - Added MAX_INPUT_ATTEMPTS constant and retry counters in all loops:

- `_prompt_custom_timeframe`: lines 75-97
- `prompt_timeframe`: lines 137-161
- `prompt_interactive_mode`: lines 176-191

  Issue: The while True loop in prompt_timeframe has no explicit exit condition
  beyond user providing valid input. If prompt_user_input has issues, this could hang.

  Recommendation: Add maximum retry limit:

  MAX_RETRIES = 10
  attempts = 0

  while attempts < MAX_RETRIES:
      attempts += 1
      choice = prompt_user_input(...)
      # ... validation logic

  if attempts >= MAX_RETRIES:
      log_error("Maximum retry attempts reached. Using default timeframe.")
      return default_timeframe

  1. ✅ Missing Test Coverage Check - **COMPLETED**

  **Status**: Implemented in `tests/adaptive_trend_LTS_mini/test_interactive_prompts.py`

Issue: No unit tests visible for this module.
Interactive prompts are notoriously difficult to test but critical for UX.

  Recommendation: Create tests with mocked prompt_user_input:

# tests/adaptive_trend_LTS_mini/test_interactive_prompts.py

  from unittest.mock import patch
  from modules.adaptive_trend_LTS_mini.cli.interactive_prompts import (
      prompt_timeframe,
      UserExitRequested,
  )

  @patch('modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input')
  def test_prompt_timeframe_valid_selection(mock_input):
      mock_input.return_value = "1"  # Select 15m
      result = prompt_timeframe()
      assert result == "15m"

  @patch('modules.adaptive_trend_LTS_mini.cli.interactive_prompts.prompt_user_input')
  def test_prompt_timeframe_custom(mock_input):
      mock_input.side_effect = ["6", "2h"]  # Custom option, then "2h"
      result = prompt_timeframe()
      assert result == "2h"

🟢 Low Priority

1. ✅ Hardcoded Timeframe List (Lines 158-164) - **COMPLETED**

**Status**: Fixed - `prompt_timeframe` now accepts `available_timeframes` argument

  Issue: Timeframe options are hardcoded. Could be configurable or imported from common config.

  Recommendation:

  from config import COMMON_TIMEFRAMES, DEFAULT_TIMEFRAME

  def prompt_timeframe(
      default_timeframe: str = DEFAULT_TIMEFRAME,
      available_timeframes: list[tuple[str, str]] = None
  ) -> str:
      if available_timeframes is None:
          available_timeframes = COMMON_TIMEFRAMES
      # ... rest of function

1. ✅ Color Constants Could Be Centralized - **COMPLETED**

**Status**: Fixed - All color constants centralized at module level (lines 40-43 in interactive_prompts.py)

```python
MENU_COLOR = Fore.CYAN
  HIGHLIGHT_COLOR = Fore.MAGENTA
  ERROR_COLOR = Fore.RED
  ```

  Issue: Color choices (Fore.CYAN, Fore.MAGENTA) are scattered throughout.
  If branding changes, multiple locations need updates.

  Recommendation:

# At module level

  MENU_COLOR = Fore.CYAN
  HIGHLIGHT_COLOR = Fore.MAGENTA
  ERROR_COLOR = Fore.RED  # If used elsewhere

  1. ✅ _find_timeframe_index Returns -1 on Not Found - **COMPLETED**

  **Status**: Fixed - Now returns -1 instead of 0 for "not found" (line 70 in interactive_prompts.py)

  Issue: Returning 0 as default is misleading - it implies first item when it should indicate "not found".

  def _find_timeframe_index(timeframes: list[tuple[str, str]], target: str) -> int:
      """Find index of timeframe in list.

      Args:
          timeframes: List of (timeframe, description) tuples
          target: Target timeframe string

      Returns:
          Index of timeframe, or 0 if not found
      """
      for idx, (tf, _) in enumerate(timeframes):
          if tf == target:
              return idx
      return 0  # Should this be -1 or raise ValueError?

  Recommendation:

  def _find_timeframe_index(timeframes: list[tuple[str, str]], target: str) -> int:
      """Find index of timeframe in list.

      Returns:
          Index of timeframe, or 0 (first option) if not found
      """
      for idx, (tf, _) in enumerate(timeframes):
          if tf == target:
              return idx
      # Default to first option if target not found
      return 0

  Or use -1 and handle it explicitly in the caller.

  Security Considerations

  ✅ Secure Practices

  1. Input Length Validation - Prevents DoS via memory exhaustion (MAX_INPUT_LENGTH = 100)
  2. No eval() or exec() - No arbitrary code execution risks
  3. Regex Validation - Uses TIMEFRAME_NORMALIZED_RE for format validation
  4. Safe Type Conversions - Pre-validates numeric input before int() conversion

  ⚠️ Minor Concerns

  1. Regex Complexity - External dependency on TIMEFRAME_NORMALIZED_RE.
     Ensure it's not vulnerable to ReDoS (Regular Expression Denial of Service).
  2. No Rate Limiting - Infinite retry loops could be abused by malicious scripts.
     (Though CLI context mitigates this somewhat).

  Performance Considerations

  ✅ Efficient

- Minimal memory footprint (small lists, no unnecessary allocations)
- No blocking I/O beyond user input (which is expected)
- No expensive computations

  📝 Notes

- Interactive prompts are inherently slow (waiting for user input), so performance is not critical here

  Project Convention Adherence

  ✅ Follows Conventions

  1. Type Hints - Complete type annotations following Python 3.9+ syntax (list[tuple[str, str]])
  2. Docstrings - Comprehensive docstrings with Args/Returns
  3. Logging - Uses project's log_* utilities from modules.common.utils
  4. Error Handling - Custom exception class for application-level errors
  5. Constants - Module-level constants in UPPER_CASE
  6. Private Functions - Leading underscore convention (_validate_input_length)

  ✅ Potential Improvements

  1. ✅ Test Coverage - Add unit tests with mocked inputs **[COMPLETED]**
  2. ✅ Config Integration - Consider importing timeframe lists from config **[COMPLETED]**
  3. ✅ Retry Limits - Add maximum retry attempts to prevent infinite loops **[COMPLETED]**

  Specific Recommendations

  High Priority (Reliability)

  1. ✅ Add retry limits to all while True loops to prevent potential hangs:
     MAX_INPUT_ATTEMPTS = 10 **[COMPLETED - lines 38, 75-97, 137-161, 176-191]**
  2. ✅ Create unit tests using unittest.mock.patch:
     tests/adaptive_trend_LTS_mini/test_interactive_prompts.py **[COMPLETED]**

  Medium Priority (Maintainability)

  1. ✅ Centralize color constants:
     MENU_COLOR = Fore.CYAN
     HIGHLIGHT_COLOR = Fore.MAGENTA **[COMPLETED - lines 40-43]**
  2. ✅ Document infinite loop behavior in docstrings:
  - Note that Ctrl+C exits **[COMPLETED]**
  - Mention that loops continue until valid input **[COMPLETED]**

  Low Priority (Nice-to-Have)

  1. ✅ Make timeframe list configurable via function parameter **[COMPLETED]**
  2. ✅ Consider returning -1 instead of 0 in _find_timeframe_index for "not found" **[COMPLETED - line 70]**

  Summary

  Overall Assessment: 9.0/10 ✅ Production Quality

  ## Implementation Progress: 8/8 Completed (100.0%)

  ### ✅ Completed Improvements

  1. Add unit tests with mocked prompt_user_input (HIGH PRIORITY)
  2. Make timeframe list configurable via parameter (LOW PRIORITY)
  3. All docstring notes and retry limits implemented

  Strengths:

- Excellent code structure with clear separation of concerns
- Strong type safety and comprehensive documentation
- Good user experience with validation and colored output
- Robust error handling with retry limits
- Centralized constants for maintainability
- High test coverage

  Areas for Improvement:

- None identified at this stage.

  Recommendation: This code is updated and verified.

  1. ✅ Retry limits implemented
  2. ✅ Unit tests created
  3. ✅ Color constants centralized
  4. ✅ Fixed _find_timeframe_index return value

  The code follows best practices and project conventions well. No critical bugs found.
