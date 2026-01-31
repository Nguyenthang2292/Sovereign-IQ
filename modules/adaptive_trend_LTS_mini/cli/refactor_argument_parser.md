 Code Review: modules/adaptive_trend_LTS_mini/cli/argument_parser.py

  Overview

  This module defines the command-line argument parser for the Adaptive Trend Classification (ATC) analysis tool. It provides a clean interface for configuring analysis parameters, MA lengths, robustness settings, and various operational modes.

  ---
  Code Quality Analysis

  ✅ Strengths

  1. Clean Structure: Well-organized, single-purpose module
  2. Comprehensive Options: 19 arguments covering all major use cases
  3. Good Defaults: Sensible default values for all parameters
  4. User-Friendly: Clear help text for each argument
  5. Graceful Import: Fallback defaults if config import fails
  6. Type Annotations: Correct types for all arguments (str, int, float, bool)
  7. Descriptive Naming: Clear argument names (--ema-len, --min-signal, etc.)

  ⚠️ Issues & Suggestions

  1. Missing Return Type Hint (Line 24)

  def parse_args():
      """Parse command-line arguments for ATC analysis."""
  Issue: No return type annotation.

  Suggestion:
  def parse_args() -> argparse.Namespace:
      """Parse command-line arguments for ATC analysis."""

  2. Hardcoded Magic Numbers (Lines 58, 64, 70, 76, 82, 88, 101, 108, 114, 141, 152)

  default=28,        # Lines 58, 64, 70, 76, 82, 88
  default=0.02,      # Line 101
  default=0.03,      # Line 108
  default=0,         # Line 114
  default=0.01,      # Line 141
  default=100,       # Line 152
  Issue: Multiple magic numbers scattered throughout.

  Suggestion:
  # Default parameter values
  DEFAULT_MA_LENGTH = 28
  DEFAULT_LAMBDA = 0.02
  DEFAULT_DECAY = 0.03
  DEFAULT_CUTOUT = 0
  DEFAULT_MIN_SIGNAL = 0.01
  DEFAULT_BATCH_SIZE = 100

  # Use in arguments
  parser.add_argument("--ema-len", type=int, default=DEFAULT_MA_LENGTH, ...)

  3. Inconsistent Naming Convention (Line 99-102)

  parser.add_argument(
      "--lambda",
      type=float,
      default=0.02,
      dest="lambda_param",
      help="Lambda parameter for exponential growth (default: 0.02)",
  )
  Issue: CLI argument --lambda maps to lambda_param internally. This inconsistency can confuse users/developers.

  Suggestion: Either:
  - Use --lambda-param in CLI and lambda_param internally (consistent)
  - Document why lambda can't be used directly (Python keyword)

  Better approach:
  parser.add_argument(
      "--lambda-param",  # Consistent naming
      type=float,
      default=DEFAULT_LAMBDA,
      help="Lambda parameter for exponential growth (default: 0.02)",
  )

  4. Repetitive MA Argument Pattern (Lines 56-90)

  parser.add_argument("--ema-len", type=int, default=28, help="EMA length (default: 28)")
  parser.add_argument("--hma-len", type=int, default=28, help="HMA length (default: 28)")
  parser.add_argument("--wma-len", type=int, default=28, help="WMA length (default: 28)")
  # ... repeats 6 times
  Issue: Highly repetitive code pattern.

  Suggestion: Extract to helper function:
  def _add_ma_arguments(parser: argparse.ArgumentParser, ma_length: int = DEFAULT_MA_LENGTH) -> None:
      """Add Moving Average length arguments.

      Args:
          parser: Argument parser to add arguments to
          ma_length: Default MA length for all indicators
      """
      ma_types = ["ema", "hma", "wma", "dema", "lsma", "kama"]
      for ma_type in ma_types:
          parser.add_argument(
              f"--{ma_type}-len",
              type=int,
              default=ma_length,
              help=f"{ma_type.upper()} length (default: {ma_length})",
          )

  5. Missing Input Validation (Various lines)

  Issue: No validation for:
  - Negative values for lengths, limits, batch_size
  - Invalid timeframe formats
  - min_signal outside reasonable range (e.g., > 1.0)

  Suggestion: Add custom validation:
  def parse_args() -> argparse.Namespace:
      """Parse and validate command-line arguments."""
      parser = argparse.ArgumentParser(...)
      # ... add arguments ...

      args = parser.parse_args()

      # Validation
      if args.limit <= 0:
          parser.error("--limit must be positive")
      if args.batch_size <= 0:
          parser.error("--batch-size must be positive")
      if not (0 < args.min_signal <= 1.0):
          parser.error("--min-signal must be between 0 and 1.0")
      if any(getattr(args, f'{ma}_len', 0) <= 0 for ma in ['ema', 'hma', 'wma', 'dema', 'lsma', 'kama']):
          parser.error("MA lengths must be positive")

      return args

  6. No Mutually Exclusive Groups (Lines 118-147)

  parser.add_argument("--no-prompt", ...)
  parser.add_argument("--no-menu", ...)
  parser.add_argument("--list-symbols", ...)
  parser.add_argument("--auto", ...)
  Issue: Some combinations don't make sense (e.g., --list-symbols with --auto).

  Suggestion: Use argument groups:
  mode_group = parser.add_mutually_exclusive_group()
  mode_group.add_argument("--auto", action="store_true", help="Force auto mode")
  mode_group.add_argument("--list-symbols", action="store_true", help="List symbols and exit")

  7. Inconsistent Help Text Format (Lines 35, 41, 47, 53)

  help=f"Symbol pair to analyze (default: {DEFAULT_SYMBOL})",
  help=f"Quote currency (default: {DEFAULT_QUOTE})",
  vs
  help="EMA length (default: 28)",
  help="KAMA length (default: 28)",
  Issue: Some use f-strings with constants, others hardcode defaults in help text.

  Suggestion: Be consistent - use f-strings everywhere:
  help=f"EMA length (default: {DEFAULT_MA_LENGTH})",

  8. Missing Argument Grouping (All arguments)

  Issue: All arguments in one flat list is hard to navigate with --help.

  Suggestion: Group related arguments:
  # Basic options
  basic_group = parser.add_argument_group('Basic Options')
  basic_group.add_argument('--symbol', ...)
  basic_group.add_argument('--timeframe', ...)

  # MA parameters
  ma_group = parser.add_argument_group('Moving Average Parameters')
  ma_group.add_argument('--ema-len', ...)

  # Advanced parameters
  advanced_group = parser.add_argument_group('Advanced Parameters')
  advanced_group.add_argument('--lambda-param', ...)

  # Mode options
  mode_group = parser.add_argument_group('Mode Options')
  mode_group.add_argument('--auto', ...)

  9. Missing Type Definitions (Entire file)

  Issue: No type alias or dataclass for parsed arguments.

  Suggestion: Create a typed namespace:
  from dataclasses import dataclass
  from typing import Optional

  @dataclass
  class ATCArguments:
      """Typed arguments for ATC analysis."""
      symbol: Optional[str]
      quote: str
      timeframe: str
      limit: int
      ema_len: int
      # ... all other fields

      @classmethod
      def from_namespace(cls, ns: argparse.Namespace) -> 'ATCArguments':
          """Convert argparse.Namespace to typed ATCArguments."""
          return cls(**vars(ns))

  10. No Version Information (Missing)

  Issue: No --version argument.

  Suggestion:
  parser.add_argument(
      "--version",
      action="version",
      version="%(prog)s 1.0.0",
  )

  ---
  Specific Recommendations

  High Priority

  1. Add return type hint -> argparse.Namespace
  2. Add input validation for numerical arguments
  3. Extract magic numbers to constants

  Medium Priority

  4. Refactor repetitive MA arguments using helper function
  5. Fix naming inconsistency (--lambda → --lambda-param)
  6. Add mutually exclusive groups for mode options

  Low Priority (Code Quality)

  7. Group arguments for better --help output
  8. Standardize help text format (always use f-strings)
  9. Add version argument
  10. Create typed argument dataclass for better type safety

  ---
  Performance Considerations

  - ✅ No performance issues - argument parsing is one-time operation
  - ✅ Efficient default value handling

  ---
  Security Considerations

  - ⚠️ Input Validation Missing: No validation for:
    - Very large --limit values (potential memory exhaustion)
    - Negative values that don't make sense
    - Very large --batch-size (memory issues)
  - ✅ No command injection risks (proper type handling)
  - ✅ No file system access

  Recommendation: Add validation to prevent:
  if args.limit > 10000:  # Arbitrary reasonable limit
      parser.error("--limit too large (max: 10000)")
  if args.batch_size > 1000:
      parser.error("--batch-size too large (max: 1000)")

  ---
  Testing Recommendations

  The module lacks tests. Consider adding:

  1. Test default values:
    - All arguments have correct defaults
    - Fallback defaults work when config import fails
  2. Test argument parsing:
    - Valid inputs parse correctly
    - Type conversions work (str, int, float, bool)
    - Choices work (--robustness)
  3. Test validation:
    - Invalid values rejected (negative numbers)
    - Out-of-range values handled
    - Mutually exclusive options enforced
  4. Test help text:
    - --help output contains all arguments
    - Default values shown correctly
  5. Test edge cases:
    - Empty command line (all defaults)
    - Mix of flags and values
    - Unknown arguments rejected

  ---
  Code Smells

  1. Long Function: parse_args() is 134 lines
  2. Magic Numbers: 28, 0.02, 0.03, 0, 0.01, 100
  3. Repetitive Code: 6 nearly identical MA argument definitions
  4. No Validation: Accepts invalid inputs without checking

  ---
  Summary

  Overall Quality: 7/10

  This is a functional argument parser with good coverage of options. The main issues are:
  - Lack of validation - accepts invalid inputs
  - Repetitive code - MA arguments could be DRY
  - Magic numbers - should be constants
  - Missing type hints - return type needed
  - No tests - critical for CLI tools

  The code is production-ready but would benefit from:
  - Input validation to prevent errors/security issues
  - Refactoring to reduce repetition
  - Comprehensive test coverage
  - Better organization (argument groups)

  ---