"""
Command-line argument parser for pairs trading analysis.

This module provides the main argument parser for the pairs trading CLI,
defining all command-line options and their default values.
"""

import argparse

try:
    from config import (
        PAIRS_TRADING_TOP_N,
        PAIRS_TRADING_MIN_SPREAD,
        PAIRS_TRADING_MAX_SPREAD,
        PAIRS_TRADING_MIN_CORRELATION,
        PAIRS_TRADING_MAX_CORRELATION,
        PAIRS_TRADING_MAX_HALF_LIFE,
        PAIRS_TRADING_WEIGHT_PRESETS,
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
        PAIRS_TRADING_KALMAN_PRESETS,
        PAIRS_TRADING_OPPORTUNITY_PRESETS,
    )
except ImportError:
    PAIRS_TRADING_TOP_N = 5
    PAIRS_TRADING_MIN_SPREAD = 0.01
    PAIRS_TRADING_MAX_SPREAD = 0.50
    PAIRS_TRADING_MIN_CORRELATION = 0.3
    PAIRS_TRADING_MAX_CORRELATION = 0.9
    PAIRS_TRADING_MAX_HALF_LIFE = 50
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0
    PAIRS_TRADING_WEIGHT_PRESETS = {
        "momentum": {"1d": 0.5, "3d": 0.3, "1w": 0.2},
        "balanced": {"1d": 0.3, "3d": 0.4, "1w": 0.3},
    }
    PAIRS_TRADING_KALMAN_PRESETS = {
        "balanced": {"delta": 1e-5, "obs_cov": 1.0, "description": "Default balanced profile"},
    }
    PAIRS_TRADING_OPPORTUNITY_PRESETS = {
        "balanced": {
            "description": "Default balanced scoring",
        }
    }


def parse_args():
    """Parse command-line arguments for pairs trading analysis."""
    parser = argparse.ArgumentParser(
        description="Pairs Trading Analysis - Identify trading opportunities from best/worst performers"
    )
    parser.add_argument(
        "--pairs-count",
        type=int,
        default=PAIRS_TRADING_TOP_N,
        help=f"Number of tradeable pairs to return (default: {PAIRS_TRADING_TOP_N})",
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=50,
        help="Number of top/bottom symbols to consider per side when forming pairs",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["reversion", "momentum"],
        default="reversion",
        help="Chọn chiến lược giao dịch: reversion (long yếu, short mạnh) hoặc momentum (long mạnh, short yếu).",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to analyze (default: all available)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights for timeframes in format '1d:0.5,3d:0.3,1w:0.2' (default: from config)",
    )
    parser.add_argument(
        "--weight-preset",
        type=str,
        choices=list(PAIRS_TRADING_WEIGHT_PRESETS.keys()),
        default="balanced",
        help="Choose predefined weight preset (momentum or balanced). Ignored if --weights is provided.",
    )
    parser.add_argument(
        "--ols-fit-intercept",
        dest="ols_fit_intercept",
        action="store_true",
        default=PAIRS_TRADING_OLS_FIT_INTERCEPT,
        help="Use intercept term when fitting OLS hedge ratio (default: enabled).",
    )
    parser.add_argument(
        "--no-ols-fit-intercept",
        dest="ols_fit_intercept",
        action="store_false",
        help="Disable intercept (beta forced through origin) when fitting OLS hedge ratio.",
    )
    parser.add_argument(
        "--kalman-delta",
        type=float,
        default=PAIRS_TRADING_KALMAN_DELTA,
        help=f"State noise factor delta for Kalman hedge ratio (default: {PAIRS_TRADING_KALMAN_DELTA}).",
    )
    parser.add_argument(
        "--kalman-obs-cov",
        type=float,
        default=PAIRS_TRADING_KALMAN_OBS_COV,
        help=f"Observation covariance for Kalman hedge ratio (default: {PAIRS_TRADING_KALMAN_OBS_COV}).",
    )
    parser.add_argument(
        "--kalman-preset",
        type=str,
        choices=list(PAIRS_TRADING_KALMAN_PRESETS.keys()),
        default="balanced",
        help="Choose predefined Kalman parameter preset (fast_react / balanced / stable).",
    )
    parser.add_argument(
        "--opportunity-preset",
        type=str,
        choices=list(PAIRS_TRADING_OPPORTUNITY_PRESETS.keys()),
        default="balanced",
        help="Choose opportunity scoring profile (e.g. balanced/aggressive/conservative).",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=PAIRS_TRADING_MIN_SPREAD,
        help=f"Minimum spread percentage (default: {PAIRS_TRADING_MIN_SPREAD*100:.2f}%)",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=PAIRS_TRADING_MAX_SPREAD,
        help=f"Maximum spread percentage (default: {PAIRS_TRADING_MAX_SPREAD*100:.2f}%)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=PAIRS_TRADING_MIN_CORRELATION,
        help=f"Minimum correlation (default: {PAIRS_TRADING_MIN_CORRELATION:.2f})",
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=PAIRS_TRADING_MAX_CORRELATION,
        help=f"Maximum correlation (default: {PAIRS_TRADING_MAX_CORRELATION:.2f})",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help="Maximum number of pairs to display (default: 10)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip pairs validation (show all opportunities)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Manual mode: comma/space separated symbols to focus on (e.g., 'BTC/USDT,ETH/USDT')",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Skip interactive launcher (retain legacy CLI flag workflow)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["opportunity_score", "quantitative_score"],
        default="opportunity_score",
        help="Sort pairs by opportunity_score or quantitative_score (default: opportunity_score)",
    )
    parser.add_argument(
        "--require-cointegration",
        action="store_true",
        help="Only accept cointegrated pairs (filter out non-cointegrated pairs)",
    )
    parser.add_argument(
        "--max-half-life",
        type=float,
        default=PAIRS_TRADING_MAX_HALF_LIFE,
        help=f"Maximum acceptable half-life for mean reversion (default: {PAIRS_TRADING_MAX_HALF_LIFE})",
    )
    parser.add_argument(
        "--min-quantitative-score",
        type=float,
        default=None,
        help="Minimum quantitative score (0-100) to accept a pair (default: no threshold)",
    )
    
    return parser.parse_args()
