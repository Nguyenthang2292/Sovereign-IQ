"""
Mock Price Feed Module

Provides simulated cryptocurrency prices for testing and dry-run mode.
Prices fluctuate randomly within a realistic range.
"""

import random
from typing import Dict


class MockPriceFeed:
    """
    Mock price feed for simulating cryptocurrency prices.

    Maintains base prices for common symbols and simulates
    realistic price movements with random fluctuations.
    """

    def __init__(self) -> None:
        """Initialize mock price feed with base prices."""
        self.base_prices: Dict[str, float] = {
            "BTC/USDT": 42000.0,
            "ETH/USDT": 2500.0,
            "SOL/USDT": 95.0,
            "BNB/USDT": 380.0,
            "XRP/USDT": 0.55,
        }

        self.current_prices: Dict[str, float] = self.base_prices.copy()

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol with simulated movement.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")

        Returns:
            Current price with +/- 2% random fluctuation
        """
        if symbol in self.current_prices:
            current = self.current_prices[symbol]
            change_percent = random.uniform(-0.02, 0.02)
            new_price = current * (1 + change_percent)
            self.current_prices[symbol] = new_price
            return new_price
        else:
            return random.uniform(0.5, 50000.0)

    def update_prices(self) -> None:
        """
        Update all prices with small random movements.

        Simulates market price changes with +/- 1% fluctuations.
        """
        for symbol in self.current_prices:
            current = self.current_prices[symbol]
            change_percent = random.uniform(-0.01, 0.01)
            self.current_prices[symbol] = current * (1 + change_percent)

    def set_price(self, symbol: str, price: float) -> None:
        """
        Manually set price for a symbol.

        Args:
            symbol: Trading symbol
            price: New price to set
        """
        self.current_prices[symbol] = price

    def get_all_prices(self) -> Dict[str, float]:
        """
        Get all current prices.

        Returns:
            Dictionary mapping symbols to their current prices
        """
        return self.current_prices.copy()
