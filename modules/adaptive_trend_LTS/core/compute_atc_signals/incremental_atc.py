"""Incremental ATC computation for live trading.

Instead of recalculating entire signal series, this module updates
only the last bar based on stored state (MA values, equity, signals).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, Union, Optional
import json
import msgpack
import datetime

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_error, log_info, log_warn
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

        # Use O(1) MA implementations if configured
        self.use_o1_mas = config.get("use_o1_mas", True)

        # Use Rust backend for incremental updates if configured
        self.use_rust_incremental = config.get("use_rust_incremental", True)

        max_history = max(self.ma_length.values()) + 1
        self.state = {
            "ma_values": {},  # Last MA values (EMA, HMA, WMA, DEMA, LSMA, KAMA)
            "ema2_values": {},  # EMA(EMA) values for DEMA
            "equity": None,  # Last equity value
            "signal": None,  # Last signal value
            "price_history": deque(maxlen=max_history),  # Price window
            "initialized": False,
        }

        # Initialize O(1) MA objects
        self._init_o1_mas()

    def save_state(self, path: Union[str, Path]) -> None:
        """Save current state to file (MessagePack).

        Args:
            path: File path to save state to
        """
        path = Path(path)

        # Convert deques to lists for serialization
        state_to_save = self.state.copy()
        state_to_save["price_history"] = list(self.state["price_history"])
        if "hma_input_history" in self.state:
            state_to_save["hma_input_history"] = list(self.state["hma_input_history"])

        # Collect O(1) MA states if used
        if self.use_o1_mas:
            state_to_save["o1_mas_state"] = {}
            for k, v in self.o1_mas.items():
                state_to_save["o1_mas_state"][k] = v.get_state()

        payload = {
            "version": "1.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self.config,
            "state": state_to_save,
        }

        with open(path, "wb") as f:
            f.write(msgpack.packb(payload))

        log_debug(f"State saved to {path}")

    @classmethod
    def load_state(cls, path: Union[str, Path]) -> "IncrementalATC":
        """Load state from file and create restored IncrementalATC instance.

        Args:
            path: File path to load state from

        Returns:
            Restored IncrementalATC instance
        """
        path = Path(path)
        with open(path, "rb") as f:
            # raw=False ensures strings are decoded from bytes
            payload = msgpack.unpackb(f.read(), raw=False)

        # Verify version (simple check for now)
        if "version" not in payload:
            log_warn("State file missing version, assuming compatible")

        config = payload["config"]
        state_data = payload["state"]

        # Create instance
        instance = cls(config)

        # Restore state
        instance.state = state_data

        # Convert lists back to deques
        max_history = max(instance.ma_length.values()) + 1
        instance.state["price_history"] = deque(state_data["price_history"], maxlen=max_history)

        if "hma_input_history" in state_data:
            sqrt_len = max(1, int(np.sqrt(instance.ma_length["hma"])))
            instance.state["hma_input_history"] = deque(state_data["hma_input_history"], maxlen=sqrt_len)

        # Restore O(1) MA internal states
        if instance.use_o1_mas:
            log_debug("Restoring O(1) MA internal states...")
            if "o1_mas_state" in state_data:
                # Direct restoration (preferred)
                for k, v in state_data["o1_mas_state"].items():
                    if k in instance.o1_mas:
                        instance.o1_mas[k].set_state(v)
                log_debug("O(1) MA states restored directly.")
            else:
                # Fallback to replay (legacy compatibility or if not saved)
                log_debug("No O(1) state found, falling back to replay...")

                # Reset O(1) objects to be sure
                for ma_obj in instance.o1_mas.values():
                    ma_obj.reset()

                # Replay history
                for price in instance.state["price_history"]:
                    for ma_obj in instance.o1_mas.values():
                        ma_obj.update(price)

                log_debug("O(1) MA states restored via replay.")

        instance.state["initialized"] = True
        return instance

    def _init_o1_mas(self):
        """Initialize O(1) MA objects for WMA, HMA, LSMA, KAMA."""
        if not self.use_o1_mas:
            self.o1_mas = {}
            return

        try:
            from .incremental_mas_o1 import TrueO1WMA, TrueO1HMA, TrueO1LSMA, TrueO1KAMA

            self.o1_mas = {
                "wma": TrueO1WMA(self.ma_length["wma"]),
                "hma": TrueO1HMA(self.ma_length["hma"]),
                "lsma": TrueO1LSMA(self.ma_length["lsma"]),
                "kama": TrueO1KAMA(self.ma_length["kama"]),
            }
            log_debug("O(1) MA objects initialized")
        except ImportError as e:
            log_warn(f"Could not import O(1) MA implementations: {e}, falling back to legacy")
            self.use_o1_mas = False
            self.o1_mas = {}

    def initialize(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Initialize state with full calculation on historical data.

        Args:
            prices: Historical price series

        Returns:
            Full ATC results (same format as compute_atc_signals)
        """
        from modules.adaptive_trend_LTS.core.compute_moving_averages import set_of_moving_averages

        from .compute_atc_signals import compute_atc_signals

        log_debug("Initializing incremental ATC with full calculation")

        # Compute MAs directly to get actual MA values
        ma_tuples = {}
        for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
            length = self.ma_length[ma_type.lower()]
            ma_tuple = set_of_moving_averages(
                length=length,
                source=prices,
                ma_type=ma_type,
                robustness="Medium",
                use_cache=False,
                use_rust=True,
                use_cuda=False,
            )
            ma_tuples[ma_type] = ma_tuple

        # Filter out incremental-specific config parameters
        compute_config = {k: v for k, v in self.config.items() if k not in ["use_o1_mas", "use_rust_incremental"]}

        # Full calculation to establish baseline state
        results = compute_atc_signals(prices, **compute_config)

        # Extract and store state from last bar (include ma_tuples)
        self._extract_state(results, prices, ma_tuples)
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

        log_debug(f"Updating with new_price={new_price}")

        # Add to history
        self.state["price_history"].append(new_price)

        # Try Rust backend first if configured
        if self.use_rust_incremental:
            try:
                from .incremental_backend import update_incremental_auto

                signal, updated_state = update_incremental_auto(self.state, new_price, self.config)

                # Update state from Rust response
                self.state = updated_state
                self.state["signal"] = signal
                log_debug(f"Rust update complete, signal={signal}")
                return signal
            except ImportError:
                log_warn("Rust backend not available, falling back to Python")
                self.use_rust_incremental = False
            except Exception as e:
                log_warn(f"Rust backend failed: {e}, falling back to Python")
                self.use_rust_incremental = False

        # Update MA states incrementally
        self._update_mas(new_price)
        log_debug(f"After MAs update: {self.state['ma_values']}")

        # Update Layer 1 signal
        signal_l1 = self._update_layer1_signal()
        log_debug(f"Layer 1 signal: {signal_l1}")

        # Update Layer 2 equity
        self._update_equity(signal_l1)
        log_debug(f"Equity: {self.state['equity']}")

        # Calculate final signal
        signal = self._calculate_final_signal()
        log_debug(f"Final signal: {signal}")

        self.state["signal"] = signal
        return signal

    def batch_update(self, new_prices: Any) -> list[float]:
        """Update ATC signal with multiple new price bars.

        Args:
            new_prices: Sequence of new price values

        Returns:
            List of updated signal values corresponding to each price
        """
        signals = []
        for price in new_prices:
            signals.append(self.update(price))
        return signals

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

        # Reset O(1) MA objects
        if self.use_o1_mas:
            for ma_key, ma_obj in self.o1_mas.items():
                ma_obj.reset()

    def _extract_state(self, results: Dict[str, pd.Series], prices: pd.Series, ma_tuples: Dict[str, tuple]):
        """Extract state from full calculation results."""
        log_debug(f"Extracting state from results. Available keys: {list(results.keys())}")

        # Extract MA values from ma_tuples (primary MA is at index 0)
        for ma_type, ma_tuple in ma_tuples.items():
            if ma_tuple is not None:
                ma_values = ma_tuple[0]  # Primary MA is at index 0
                self.state["ma_values"][ma_type.lower()] = ma_values.iloc[-1]
                log_debug(f"Extracted {ma_type.lower()}: {self.state['ma_values'][ma_type.lower()]}")

        # Extract EMA2 for DEMA
        ema_val = self.state["ma_values"].get("ema")
        dema_val = self.state["ma_values"].get("dema")
        if ema_val is not None and dema_val is not None:
            # DEMA = 2*EMA - EMA2 -> EMA2 = 2*EMA - DEMA
            self.state["ema2_values"]["dema"] = 2 * ema_val - dema_val
            log_debug(f"Extracted ema2_values[dema]: {self.state['ema2_values']['dema']}")

        # Get Layer 2 equities (stored as {MA_TYPE}_S in results)
        equity_keys = [k for k in results.keys() if k.endswith("_S")]
        log_debug(f"Found equity keys: {equity_keys}")

        if "EMA_S" in results:
            self.state["equity"] = {
                "EMA": results["EMA_S"].iloc[-1],
                "HMA": results["HMA_S"].iloc[-1],
                "WMA": results["WMA_S"].iloc[-1],
                "DEMA": results["DEMA_S"].iloc[-1],
                "LSMA": results["LSMA_S"].iloc[-1],
                "KAMA": results["KAMA_S"].iloc[-1],
            }
            log_debug(f"Extracted equity: {self.state['equity']}")
        else:
            log_warn("EMA_S not found in results")

        # Populate price history
        self.state["price_history"].clear()
        self.state["price_history"].extend(prices.tolist())
        log_debug(f"Price history populated with {len(self.state['price_history'])} prices")

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
        if self.use_o1_mas and ma_key == "wma" and ma_key in self.o1_mas:
            self.state["ma_values"][ma_key] = self.o1_mas[ma_key].update(new_price)
            return

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
        if self.use_o1_mas and "hma" in self.o1_mas:
            self.state["ma_values"]["hma"] = self.o1_mas["hma"].update(new_price)
            return

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
        if self.use_o1_mas and "lsma" in self.o1_mas:
            self.state["ma_values"]["lsma"] = self.o1_mas["lsma"].update(new_price)
            return

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
        if self.use_o1_mas and "kama" in self.o1_mas:
            self.state["ma_values"]["kama"] = self.o1_mas["kama"].update(new_price)
            return

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

        # Create new dictionary to ensure object identity changes
        new_equity = {}
        for ma_type in self.state["equity"]:
            prev_equity = self.state["equity"][ma_type]
            new_equity[ma_type] = prev_equity * (1 - decay) + signal_l1 * la

        self.state["equity"] = new_equity

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


TF_RESOLUTION_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


class MultiTimeframeIncrementalATC:
    """Multi-timeframe wrapper for IncrementalATC.

    Maintains one IncrementalATC instance per timeframe and synchronizes
    updates from a base timeframe (e.g., 1m) to higher timeframes only when
    their bars complete.

    Usage:
        mtf = MultiTimeframeIncrementalATC(config, timeframes=["1m", "5m", "15m"])
        mtf.initialize({"1m": prices_1m, "5m": prices_5m, "15m": prices_15m})
        signals = mtf.update(new_price, timeframe="1m")
    """

    def __init__(self, config: Dict[str, Any], timeframes: list[str] | None = None):
        """Initialize multi-timeframe ATC.

        Args:
            config: ATC configuration parameters (same as IncrementalATC)
            timeframes: List of timeframe strings, ordered from base to highest
                       (default: ["1m", "5m", "15m"])
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m"]

        self.config = config
        self.timeframes = timeframes
        self.base_tf = timeframes[0]
        self.higher_tfs = timeframes[1:]

        log_debug(f"Initializing MTF with timeframes: {self.timeframes}, base: {self.base_tf}")

        # Create one IncrementalATC per timeframe
        self.atcs = {tf: IncrementalATC(config) for tf in timeframes}

        # Bar counters for each timeframe
        self.bar_counters = {tf: 0 for tf in timeframes}

        # Track last bar prices for each higher timeframe
        self.last_bar_prices: Dict[str, float | None] = {tf: None for tf in self.higher_tfs}

    def _bars_per_tf(self, tf: str) -> int:
        """Get number of minutes in a timeframe bar."""
        return TF_RESOLUTION_MAP.get(tf, 1)

    def _is_bar_completed(self, base_bar_index: int, target_tf: str) -> bool:
        """Check if a higher timeframe bar has completed.

        Args:
            base_bar_index: Current bar index in base timeframe
            target_tf: Target timeframe to check

        Returns:
            True if the target timeframe bar has completed
        """
        base_minutes = self._bars_per_tf(self.base_tf)
        target_minutes = self._bars_per_tf(target_tf)

        if base_minutes == 0 or target_minutes == 0:
            return False

        ratio = target_minutes // base_minutes
        if ratio == 0:
            return False

        return (base_bar_index + 1) % ratio == 0

    def initialize(self, historical_data):
        """Initialize all timeframe ATCs with historical data.

        Args:
            historical_data: Either:
                - Dict mapping timeframe to price series: {"1m": prices_1m, "5m": prices_5m}
                - Single price series (will be used for all timeframes)

        Returns:
            Dict of initialization results per timeframe
        """
        log_debug("Initializing MTF ATC with historical data")

        results = {}

        if isinstance(historical_data, dict):
            for tf, prices in historical_data.items():
                if tf in self.atcs:
                    log_debug(f"Initializing {tf} ATC with {len(prices)} bars")
                    results[tf] = self.atcs[tf].initialize(prices)
        else:
            log_debug(f"Single dataset provided, initializing all TFs with {len(historical_data)} bars")
            for tf in self.timeframes:
                results[tf] = self.atcs[tf].initialize(historical_data)

        # Initialize last bar prices for higher TFs with last prices from data
        for tf in self.higher_tfs:
            if isinstance(historical_data, dict) and tf in historical_data:
                self.last_bar_prices[tf] = historical_data[tf].iloc[-1]
            elif isinstance(historical_data, pd.Series):
                self.last_bar_prices[tf] = historical_data.iloc[-1]

        log_debug("MTF initialization complete")
        return results

    def update(self, new_price: float, timeframe: str | None = None) -> Dict[str, float]:
        """Update ATC signals across all timeframes.

        Args:
            new_price: New price value
            timeframe: Timeframe of the update (default: base_tf)

        Returns:
            Dict of signal values per timeframe
        """
        if timeframe is None:
            timeframe = self.base_tf

        signals = {}

        if timeframe == self.base_tf:
            signals = self._update_from_base(new_price)
        else:
            log_warn(f"Direct updates to non-base timeframe {timeframe} not supported")
            return {tf: self.atcs[tf].state.get("signal", 0.0) for tf in self.timeframes}

        return signals

    def _update_from_base(self, new_price: float) -> Dict[str, float]:
        """Update from base timeframe and sync to higher timeframes.

        Args:
            new_price: New price from base timeframe

        Returns:
            Dict of signal values per timeframe
        """
        signals = {}

        # Update base TF
        base_bar_index = self.bar_counters[self.base_tf]
        base_signal = self.atcs[self.base_tf].update(new_price)
        self.bar_counters[self.base_tf] += 1
        signals[self.base_tf] = base_signal

        # Store last bar price for higher TFs
        for tf in self.higher_tfs:
            self.last_bar_prices[tf] = new_price

        # Check which higher TFs have completed bars
        for tf in self.higher_tfs:
            if self._is_bar_completed(base_bar_index, tf):
                log_debug(f"Bar completed for {tf} at base bar {base_bar_index}")

                # Push the last closed bar price to higher TF
                bar_price = self.last_bar_prices[tf]
                if bar_price is not None:
                    self.bar_counters[tf] += 1
                    tf_signal = self.atcs[tf].update(bar_price)
                    signals[tf] = tf_signal
                    log_debug(f"{tf} updated with signal {tf_signal}")
            else:
                # Return last known signal for this TF
                signals[tf] = self.atcs[tf].state.get("signal", 0.0)

        log_debug(f"MTF signals: {signals}")
        return signals

    def reset(self):
        """Reset all timeframe ATCs."""
        log_debug("Resetting MTF ATC")
        for tf in self.timeframes:
            self.atcs[tf].reset()
            self.bar_counters[tf] = 0
            self.last_bar_prices[tf] = None

    def get_state(self, tf: str | None = None) -> Dict:
        """Get state for specific timeframe or all timeframes.

        Args:
            tf: Specific timeframe to get state for (default: all)

        Returns:
            State dictionary for requested timeframe(s)
        """
        if tf is not None:
            return self.atcs[tf].state
        return {tf: self.atcs[tf].state for tf in self.timeframes}

    def get_signal(self, tf: str | None = None) -> float | Dict[str, float]:
        """Get current signal for specific timeframe or all timeframes.

        Args:
            tf: Specific timeframe to get signal for (default: all)

        Returns:
            Signal value or dict of signals per timeframe
        """
        if tf is not None:
            return self.atcs[tf].state.get("signal", 0.0)
        return {tf: self.atcs[tf].state.get("signal", 0.0) for tf in self.timeframes}
