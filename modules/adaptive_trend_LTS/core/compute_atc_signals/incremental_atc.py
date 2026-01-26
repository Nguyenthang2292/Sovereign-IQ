"""Incremental ATC computation for live trading.

Instead of recalculating entire signal series, this module updates
only the last bar based on stored state (MA values, equity, signals).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_info, log_warn, log_error
except ImportError:

    def log_debug(msg: str) -> None:
        print(f"[DEBUG] {msg}")

    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:
        print(f"[ERROR] {msg}")


class IncrementalATC:
    """Incremental ATC calculator that maintains state between updates.

    Usage:
        atc = IncrementalATC(config)
        atc.initialize(prices)  # Full calculation for initial state
        signal = atc.update(new_price)  # O(1) update for new bar
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize incremental ATC with configuration.

        Args:
            config: ATC configuration parameters (same as compute_atc_signals)
        """
        self.config = config

        # State variables
        self.ma_length = {
            "ema": config.get("ema_len", 28),
            "hma": config.get("hull_len", 28),
            "wma": config.get("wma_len", 28),
            "dema": config.get("dema_len", 28),
            "lsma": config.get("lsma_len", 28),
            "kama": config.get("kama_len", 28),
        }

        max_history = max(self.ma_length.values()) + 1
        self.state = {
            "ma_values": {},  # Last MA values (EMA, HMA, WMA, DEMA, LSMA, KAMA)
            "ema2_values": {},  # EMA(EMA) values for DEMA
            "equity": None,  # Last equity value
            "signal": None,  # Last signal value
            "price_history": deque(maxlen=max_history),  # Price window
            "initialized": False,
        }

    def initialize(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Initialize state with full calculation on historical data.

        Args:
            prices: Historical price series

        Returns:
            Full ATC results (same format as compute_atc_signals)
        """
        from .compute_atc_signals import compute_atc_signals

        log_debug("Initializing incremental ATC with full calculation")

        # Full calculation to establish baseline state
        results = compute_atc_signals(prices, **self.config)

        # Extract and store state from last bar
        self._extract_state(results, prices)
        self.state["initialized"] = True

        log_debug("Incremental ATC initialized successfully")
        return results

    def update(self, new_price: float) -> float:
        """Update ATC signal with new price bar (O(1) operation).

        Args:
            new_price: New price value

        Returns:
            Updated signal value
        """
        if not self.state["initialized"]:
            raise RuntimeError("Must call initialize() before update()")

        # Add to history
        self.state["price_history"].append(new_price)

        # Update MA states incrementally
        self._update_mas(new_price)

        # Update Layer 1 signal
        signal_l1 = self._update_layer1_signal()

        # Update Layer 2 equity
        self._update_equity(signal_l1)

        # Calculate final signal
        signal = self._calculate_final_signal()

        self.state["signal"] = signal
        return signal

    def reset(self):
        """Reset state (for new symbol or configuration change)."""
        log_debug("Resetting incremental ATC state")
        max_history = max(self.ma_length.values()) + 1
        self.state = {
            "ma_values": {},
            "ema2_values": {},
            "equity": None,
            "signal": None,
            "price_history": deque(maxlen=max_history),
            "initialized": False,
        }

    def _extract_state(self, results: Dict[str, pd.Series], prices: pd.Series):
        """Extract state from full calculation results."""
        # Get moving averages from results
        mas = ["ema", "hma", "wma", "dema", "lsma", "kama"]
        for ma_type in mas:
            if ma_type in results:
                self.state["ma_values"][ma_type] = results[ma_type].iloc[-1]

        # Extract EMA2 for DEMA
        if "dema" in results and "ema" in results:
            # DEMA = 2*EMA - EMA2 -> EMA2 = 2*EMA - DEMA
            ema_val = results["ema"].iloc[-1]
            dema_val = results["dema"].iloc[-1]
            self.state["ema2_values"]["dema"] = 2 * ema_val - dema_val

        # Get Layer 1 equities
        if "L1_Equity_EMA" in results:
            self.state["equity"] = {
                "EMA": results["L1_Equity_EMA"].iloc[-1],
                "HMA": results["L1_Equity_HMA"].iloc[-1],
                "WMA": results["L1_Equity_WMA"].iloc[-1],
                "DEMA": results["L1_Equity_DEMA"].iloc[-1],
                "LSMA": results["L1_Equity_LSMA"].iloc[-1],
                "KAMA": results["L1_Equity_KAMA"].iloc[-1],
            }

        # Populate price history
        self.state["price_history"].clear()
        self.state["price_history"].extend(prices.values)

    def _update_mas(self, new_price: float):
        """Update all MA states incrementally."""
        self._update_ema(new_price, self.ma_length["ema"])
        self._update_hma(new_price, self.ma_length["hma"])
        self._update_wma(new_price, self.ma_length["wma"])
        self._update_dema(new_price, self.ma_length["dema"])
        self._update_lsma(new_price, self.ma_length["lsma"])
        self._update_kama(new_price, self.ma_length["kama"])

    def _update_ema(self, new_price: float, length: int):
        """Update EMA incrementally."""
        alpha = 2.0 / (length + 1.0)
        prev_ema = self.state["ma_values"].get("ema", new_price)
        new_ema = alpha * new_price + (1 - alpha) * prev_ema
        self.state["ma_values"]["ema"] = new_ema

    def _update_wma(self, new_price: float, length: int, ma_key: str = "wma"):
        """Update WMA incrementally."""
        prices = list(self.state["price_history"])
        if len(prices) < length:
            self.state["ma_values"][ma_key] = new_price
            return

        window = prices[-length:]
        weights = np.arange(1, length + 1)
        wma = np.dot(window, weights) / weights.sum()
        self.state["ma_values"][ma_key] = wma

    def _update_hma(self, new_price: float, length: int):
        """Update HMA incrementally."""
        half_len = max(1, length // 2)
        sqrt_len = max(1, int(np.sqrt(length)))

        self._update_wma(new_price, half_len, "wma_half")
        self._update_wma(new_price, length, "wma_full")

        wma_half = self.state["ma_values"].get("wma_half", new_price)
        wma_full = self.state["ma_values"].get("wma_full", new_price)

        hma_input_val = 2 * wma_half - wma_full

        if "hma_input_history" not in self.state:
            self.state["hma_input_history"] = deque(maxlen=sqrt_len)
        self.state["hma_input_history"].append(hma_input_val)

        if len(self.state["hma_input_history"]) >= sqrt_len:
            weights = np.arange(1, sqrt_len + 1)
            hma = np.dot(list(self.state["hma_input_history"]), weights) / weights.sum()
            self.state["ma_values"]["hma"] = hma
        else:
            self.state["ma_values"]["hma"] = hma_input_val

    def _update_dema(self, new_price: float, length: int):
        """Update DEMA incrementally."""
        alpha = 2.0 / (length + 1.0)

        prev_ema = self.state["ma_values"].get("ema", new_price)
        new_ema = alpha * new_price + (1 - alpha) * prev_ema
        self.state["ma_values"]["ema"] = new_ema

        prev_ema2 = self.state["ema2_values"].get("dema", new_ema)
        new_ema2 = alpha * new_ema + (1 - alpha) * prev_ema2
        self.state["ema2_values"]["dema"] = new_ema2

        self.state["ma_values"]["dema"] = 2 * new_ema - new_ema2

    def _update_lsma(self, new_price: float, length: int):
        """Update LSMA incrementally."""
        prices = list(self.state["price_history"])
        if len(prices) < length:
            self.state["ma_values"]["lsma"] = new_price
            return

        window = prices[-length:]
        x = np.arange(length)
        y = np.array(window)

        n = length
        sum_x = n * (n - 1) / 2
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6
        sum_y = np.sum(y)
        sum_xy = np.dot(x, y)

        denom = n * sum_x2 - sum_x**2
        if denom == 0:
            self.state["ma_values"]["lsma"] = new_price
            return

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        lsma = intercept + slope * (n - 1)
        self.state["ma_values"]["lsma"] = lsma

    def _update_kama(self, new_price: float, length: int):
        """Update KAMA incrementally."""
        prev_kama = self.state["ma_values"].get("kama", new_price)

        prices = list(self.state["price_history"])
        if len(prices) < length + 1:
            self.state["ma_values"]["kama"] = new_price
            return

        window = prices[-(length + 1) :]
        change = abs(window[-1] - window[0])
        volatility = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))

        er = change / volatility if volatility != 0 else 0
        fast_sc = 2 / (2.0 + 1)
        slow_sc = 2 / (30.0 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        new_kama = prev_kama + sc * (new_price - prev_kama)
        self.state["ma_values"]["kama"] = new_kama

    def _update_layer1_signal(self) -> float:
        """Calculate Layer 1 signal from current MA states."""
        from modules.adaptive_trend_enhance.core.process_layer1.layer1_signal import _layer1_signal_for_ma

        ma_values = self.state["ma_values"]
        price_history = list(self.state["price_history"])

        if not price_history:
            return 0.0

        current_price = price_history[-1]
        decay = self.config.get("De", 0.03) / 100.0
        la = self.config.get("La", 0.02) / 1000.0

        signals = []
        for ma_type in ["ema", "hma", "wma", "dema", "lsma", "kama"]:
            if ma_type in ma_values:
                ma_val = ma_values[ma_type]
                dummy_ma_series = pd.Series([ma_val])
                ma_tuple = tuple([dummy_ma_series] * 9)

                signal, _, _ = _layer1_signal_for_ma(pd.Series([current_price]), ma_tuple, L=la, De=decay)
                signals.append(signal.iloc[-1])

        return np.mean(signals) if signals else 0.0

    def _update_equity(self, signal_l1: float):
        """Update equity incrementally."""
        if self.state["equity"] is None:
            self.state["equity"] = {m: 1.0 for m in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]}
            return

        decay = self.config.get("De", 0.03) / 100.0
        la = self.config.get("La", 0.02) / 1000.0

        for ma_type in self.state["equity"]:
            prev_equity = self.state["equity"][ma_type]
            new_equity = prev_equity * (1 - decay) + signal_l1 * la
            self.state["equity"][ma_type] = new_equity

    def _calculate_final_signal(self) -> float:
        """Calculate final Average_Signal."""
        if self.state["equity"] is None:
            return 0.0

        long_threshold = self.config.get("long_threshold", 0.1)
        short_threshold = self.config.get("short_threshold", -0.1)

        ma_signals = {}
        for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
            ma_val = self.state["ma_values"].get(ma_type.lower())
            price_history = list(self.state["price_history"])
            if ma_val is not None and price_history:
                current_price = price_history[-1]
                ma_signals[ma_type] = self._get_layer1_signal(ma_val, current_price, long_threshold, short_threshold)
            else:
                ma_signals[ma_type] = 0.0

        equities = np.array([self.state["equity"].get(ma, 1.0) for ma in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]])
        C = np.array([ma_signals[ma] for ma in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]])

        nom = np.sum(C * equities)
        den = np.sum(equities)

        return nom / den if den != 0 else 0.0

    def _get_layer1_signal(self, ma_val: float, price: float, long_threshold: float, short_threshold: float) -> float:
        """Get Layer 1 signal for a single MA."""
        signal_l1 = (price - ma_val) / ma_val if ma_val != 0 else 0.0
        if signal_l1 > long_threshold:
            return 1.0
        if signal_l1 < short_threshold:
            return -1.0
        return 0.0
