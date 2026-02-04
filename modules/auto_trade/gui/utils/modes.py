"""
Trading Modes Module

Defines constants for different trading modes.
"""


class TradingMode:
    """
    Trading mode constants.

    - PRODUCTION: Live trading with real money
    - DEMO: Testnet trading with test funds
    - DRY_RUN: Simulated trading without real exchange connection
    """

    PRODUCTION: str = "PRODUCTION"
    DEMO: str = "DEMO"
    DRY_RUN: str = "DRY_RUN"
