import random
from typing import Dict


class MockPriceFeed:
    def __init__(self):
        self.base_prices = {
            "BTC/USDT": 42000.0,
            "ETH/USDT": 2500.0,
            "SOL/USDT": 95.0,
            "BNB/USDT": 380.0,
            "XRP/USDT": 0.55,
        }

        self.current_prices = self.base_prices.copy()

    def get_current_price(self, symbol: str) -> float:
        if symbol in self.current_prices:
            current = self.current_prices[symbol]
            change_percent = random.uniform(-0.02, 0.02)
            new_price = current * (1 + change_percent)
            self.current_prices[symbol] = new_price
            return new_price
        else:
            return random.uniform(0.5, 50000.0)

    def update_prices(self):
        for symbol in self.current_prices:
            current = self.current_prices[symbol]
            change_percent = random.uniform(-0.01, 0.01)
            self.current_prices[symbol] = current * (1 + change_percent)

    def set_price(self, symbol: str, price: float):
        self.current_prices[symbol] = price

    def get_all_prices(self) -> Dict[str, float]:
        return self.current_prices.copy()
