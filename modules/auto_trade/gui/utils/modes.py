"""
Trading Modes Module

Defines constants for different trading modes.
"""

from enum import StrEnum


class TradingMode(StrEnum):
    """
    Trading mode constants.

    - PRODUCTION: Live trading with real money
    - DEMO: Testnet trading with test funds
    - DRY_RUN: Simulated trading without real exchange connection

    Because TradingMode extends StrEnum, each member IS a str and can be
    passed directly to any parameter typed as ``str``.
    """

    PRODUCTION = "PRODUCTION"
    DEMO = "DEMO"
    DRY_RUN = "DRY_RUN"
