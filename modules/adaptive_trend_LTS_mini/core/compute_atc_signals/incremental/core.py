"""Core Incremental ATC Logic."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:

    def log_debug(msg: str, *args: object) -> None:
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str, *args: object) -> None:
        print(f"[WARN] {msg}")


from . import ma_updaters, signal_calculator
from ..execution_shift import apply_execution_shift_value
from .state_manager import StateManager


class IncrementalATC:
    """Incremental ATC calculator that maintains state between updates.

    Usage:
        atc = IncrementalATC(config)
        atc.initialize(prices)  # Full calculation for initial state
        signal = atc.update(new_price)  # O(1) update for new bar
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize incremental ATC with configuration."""
        self.config = config
        self._lock = threading.RLock()

        # State management
        self.state_manager = StateManager(config, self._lock)

        # Shortcuts for cleaner access
        self.ma_length = self.state_manager.ma_length
        self.robustness = self.state_manager.robustness
        self.use_o1_mas = self.state_manager.use_o1_mas

        # Rust backend configuration
        self.use_rust_incremental = config.get("use_rust_incremental", True)

        # Initialize O(1) MA objects
        self._init_o1_mas()

    @property
    def state(self) -> Dict[str, Any]:
        """Access state dictionary."""
        return self.state_manager.state

    @state.setter
    def state(self, value: Dict[str, Any]):
        """Set state dictionary."""
        self.state_manager.state = value

    def save_state(self, path: Union[str, Path]) -> None:
        """Save current state to file."""
        self.state_manager.save_state(path, self.o1_mas)

    @classmethod
    def load_state(cls, path: Union[str, Path]) -> "IncrementalATC":
        """Load state from file and create restored IncrementalATC instance."""
        # Create a dummy instance first to get config?
        # Actually StateManager.load_state reads the file which contains config.
        # But we need an instance of IncrementalATC.
        # Let's read the file first to get config, then create instance, then load state.
        import msgpack  # type: ignore[import-untyped]

        path = Path(path)
        # SECURITY: Only load state files from trusted sources. Do not deserialize
        # msgpack from untrusted input (e.g. user uploads or network); use raw=False
        # for str keys/values in trusted payloads only.
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        config = payload["config"]
        instance = cls(config)

        # Now let state manager load the state properly
        instance.state_manager.load_state(path, instance.o1_mas)

        return instance

    def _init_o1_mas(self):
        """Initialize O(1) MA objects for WMA, HMA, LSMA, KAMA."""
        if not self.use_o1_mas:
            self.o1_mas = {}
            return

        try:
            # Import from parent directory (../incremental_mas_o1.py)
            from ..incremental_mas_o1 import TrueO1HMA, TrueO1KAMA, TrueO1LSMA, TrueO1WMA

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
        """Initialize state with full calculation on historical data."""
        from modules.adaptive_trend_LTS_mini.core.compute_moving_averages import (
            set_of_moving_averages,
        )

        from ..compute_atc_signals import compute_atc_signals

        log_debug("Initializing incremental ATC with full calculation")

        # Compute MAs directly to get actual MA values
        ma_tuples = {}
        for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
            length = self.ma_length[ma_type.lower()]
            ma_tuple = set_of_moving_averages(
                length=length,
                source=prices,
                ma_type=ma_type,
                robustness=self.robustness,
                use_cache=False,
                use_rust=self.config.get("use_rust_backend", True),
            )
            ma_tuples[ma_type] = ma_tuple

        # Only pass kwargs that compute_atc_signals() accepts (ignore incremental-only and extra keys)
        _allowed = {
            "ema_len", "hma_len", "wma_len", "dema_len", "lsma_len", "kama_len",
            "ema_w", "hma_w", "wma_w", "dema_w", "lsma_w", "kama_w",
            "robustness", "lambda_param", "decay", "cutout",
            "long_threshold", "short_threshold", "strategy_mode",
            "parallel_l1", "parallel_l2", "precision",
            "use_rust_backend", "use_cache", "fast_mode",
            "use_approximate", "approximate_threshold",
            "use_adaptive_approximate", "approximate_volatility_window", "approximate_volatility_factor",
            "equity_floor",
        }
        compute_config = {k: v for k, v in self.config.items() if k in _allowed}
        if "decay" not in compute_config:
            if "decay" in self.config:
                compute_config["decay"] = self.config["decay"]
            elif "decay_rate" in self.config:
                compute_config["decay"] = self.config["decay_rate"]
            elif "De" in self.config:
                compute_config["decay"] = self.config["De"]

        # Full calculation to establish baseline state
        results = compute_atc_signals(prices, **compute_config)

        # Extract and store state from last bar
        self.state_manager.extract_state_from_results(results, prices, ma_tuples)
        self.state["initialized"] = True

        log_debug("Incremental ATC initialized successfully")
        return results

    def update(self, new_price: float) -> float:
        """Update ATC signal with new price bar (O(1) operation)."""
        with self._lock:
            if not self.state["initialized"]:
                raise RuntimeError("Must call initialize() before update()")

            # Input validation
            if not isinstance(new_price, (int, float, np.integer, np.floating)):
                raise TypeError(f"new_price must be numeric, got {type(new_price)}")

            new_price = float(new_price)

            if np.isnan(new_price):
                raise ValueError("new_price cannot be NaN")
            if np.isinf(new_price):
                raise ValueError("new_price cannot be infinite")
            if new_price <= 0:
                raise ValueError(f"new_price must be positive, got {new_price}")

            # Validate price history
            min_required_history = max(self.ma_length.values())
            if len(self.state["price_history"]) < min_required_history - 1:
                raise RuntimeError(
                    f"Insufficient price history. Need at least {min_required_history - 1} bars before update(), "
                    f"but only have {len(self.state['price_history'])}. Call initialize() with sufficient data first."
                )

            log_debug(f"Updating with new_price={new_price}")

            prev_price = self.state["price_history"][-1] if self.state["price_history"] else new_price

            # Add to history
            self.state["price_history"].append(new_price)

            # Update bar index
            if self.state.get("bar_index") is None:
                self.state["bar_index"] = len(self.state["price_history"]) - 1
            else:
                self.state["bar_index"] += 1
                if self.state["bar_index"] > 2**30:
                    log_warn(f"Bar index reached {self.state['bar_index']}, resetting to prevent overflow.")
                    self.state["bar_index"] = 2**30

            # Try Rust backend first if configured
            if self.use_rust_incremental:
                try:
                    from modules.adaptive_trend_LTS_mini.core.incremental_backend import (
                        check_rust_available,
                        update_incremental_auto,
                    )

                    if check_rust_available():
                        prev_avg = self.state.get("average_signal")
                        signal, updated_state = update_incremental_auto(self.state, new_price, self.config)

                        # Update state from Rust response
                        self.state = updated_state
                        raw_signal = float(signal)
                        self.state["average_signal_prev"] = prev_avg
                        self.state["average_signal"] = raw_signal
                        self.state["average_signal_exec"] = apply_execution_shift_value(prev_avg)
                        self.state["signal_raw"] = raw_signal

                        strategy_mode = bool(self.config.get("strategy_mode", False))
                        output_signal = (
                            float(self.state["average_signal_exec"]) if strategy_mode else raw_signal
                        )
                        self.state["signal"] = output_signal
                        log_debug(
                            f"Rust update complete, raw_signal={raw_signal}, "
                            f"output_signal={output_signal}, strategy_mode={strategy_mode}"
                        )
                        return output_signal

                    log_warn("Rust backend not available, falling back to Python")
                except Exception as e:
                    # In case of import error or runtime error
                    log_warn(f"Rust backend failed: {e}, falling back to Python")

            prev_ma_values = {k: list(v) for k, v in self.state["ma_values"].items() if isinstance(v, list)}

            # Update MA states incrementally
            self._update_mas(new_price, prev_ma_values=prev_ma_values)
            log_debug(f"After MAs update: {self.state['ma_values']}")

            # Calculate signal using Python implementation
            raw_signal = self._update_python_incremental(prev_price=prev_price, prev_ma_values=prev_ma_values)
            strategy_mode = bool(self.config.get("strategy_mode", False))
            output_signal = (
                float(self.state.get("average_signal_exec", 0.0))
                if strategy_mode
                else raw_signal
            )
            log_debug(f"Final signal (raw={raw_signal}, output={output_signal}, strategy_mode={strategy_mode})")

            self.state["signal_raw"] = raw_signal
            self.state["signal"] = output_signal
            return output_signal

    def batch_update(self, new_prices: Any) -> list[float]:
        """Update ATC signal with multiple new price bars."""
        signals = []
        for price in new_prices:
            signals.append(self.update(price))
        return signals

    def reset(self):
        """Reset state."""
        log_debug("Resetting incremental ATC state")
        self.state_manager.reset(self.o1_mas)

    def _update_mas(self, new_price: float, prev_ma_values: Optional[Dict[str, list]] = None):
        """Update all MA states incrementally."""
        prev_emas = None
        if prev_ma_values:
            prev_emas = prev_ma_values.get("ema")

        # Update EMA
        new_emas = ma_updaters.update_ema(self.state, new_price, self.ma_length["ema"], self.robustness, prev_emas)

        # Update HMA
        ma_updaters.update_hma(self.state, new_price, self.ma_length["hma"], self.robustness)

        # Update WMA
        ma_updaters.update_wma(
            self.state, new_price, self.ma_length["wma"], self.robustness, self.o1_mas, self.use_o1_mas
        )

        # Update DEMA
        ma_updaters.update_dema(
            self.state, new_price, self.ma_length["dema"], self.robustness, prev_emas=prev_emas, new_emas=new_emas
        )

        # Update LSMA
        ma_updaters.update_lsma(self.state, new_price, self.ma_length["lsma"], self.robustness)

        # Update KAMA
        ma_updaters.update_kama(self.state, new_price, self.ma_length["kama"], self.robustness)

    def _update_python_incremental(self, prev_price: float, prev_ma_values: Dict[str, list]) -> float:
        """Python incremental update."""
        if not self.state["price_history"]:
            return 0.0

        new_price = self.state["price_history"][-1]

        # FIX #1: Prevent Look-ahead Bias
        # Capture Layer 1 signals from the PREVIOUS state before updating them.
        # Layer 2 equity calculation for the current bar must rely on the performance
        # of the signals generated at the previous bar (T-1), not the current bar (T).
        # We need a copy because update_layer1_signals will modify the state in-place.
        prev_layer1_signals = self.state.get("layer1_signals", {}).copy()

        # Update Layer 1 Signals and Equities
        # This updates state["layer1_signals"] to the current bar's values
        signal_calculator.update_layer1_signals(self.state, self.config, prev_price, new_price, prev_ma_values)

        # Update Layer 2 Equities
        # Pass the CAPTURED previous signals to ensure we weight based on past predictive performance
        signal_calculator.update_layer2_equities(self.state, self.config, prev_price, new_price, prev_layer1_signals)

        # Calculate Final Signal
        avg_current = signal_calculator.calculate_average_signal(self.state, self.config)

        # Persist raw average signal and derive execution-view signal in adapter layer.
        prev_avg = self.state.get("average_signal")
        self.state["average_signal_prev"] = prev_avg
        self.state["average_signal"] = avg_current
        self.state["average_signal_exec"] = apply_execution_shift_value(prev_avg)

        return avg_current
