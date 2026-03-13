"""State management for incremental ATC."""

from __future__ import annotations

import datetime
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Union

import msgpack
import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:

    def log_debug(msg: str, *args: object) -> None:
        print(f"[DEBUG] {msg % args if args else msg}")

    def log_warn(msg: str, *args: object) -> None:
        print(f"[WARN] {msg % args if args else msg}")


class StateManager:
    """Manages state for IncrementalATC."""

    def __init__(self, config: Dict[str, Any], lock: threading.RLock):
        self.config = config
        self._lock = lock
        self.ma_length = {
            "ema": config.get("ema_len", 28),
            "hma": config.get("hma_len", 28),
            "wma": config.get("wma_len", 28),
            "dema": config.get("dema_len", 28),
            "lsma": config.get("lsma_len", 28),
            "kama": config.get("kama_len", 28),
        }
        self.robustness = config.get("robustness", "Medium")
        self.use_o1_mas = config.get("use_o1_mas", True)

        # Initialize state structure
        self.state = self._create_initial_state()

    def _create_initial_state(self) -> Dict[str, Any]:
        """Create empty state dictionary."""
        # Calculate max_history
        from .constants import ROBUSTNESS_OFFSETS

        max_base_length = max(self.ma_length.values())
        max_offset = ROBUSTNESS_OFFSETS.get(self.robustness, 6)
        max_length_with_offset = max_base_length + max_offset
        max_history = max_length_with_offset + 10

        return {
            "ma_values": {},
            "ema2_values": {},
            "signals_l1": {},
            "equity_l1": {},
            "layer1_signals": {},
            "equity": None,
            "signal": None,
            "average_signal": None,
            "price_history": deque(maxlen=max_history),
            "initialized": False,
            "bar_index": None,
        }

    def reset(self, o1_mas: Optional[Dict[str, Any]] = None):
        """Reset state."""
        self.state = self._create_initial_state()

        # Clear HMA input histories
        for key in list(self.state.keys()):
            if key.startswith("hma_input_history_"):
                del self.state[key]

        # Reset O(1) MA objects
        if self.use_o1_mas and o1_mas:
            for ma_obj in o1_mas.values():
                ma_obj.reset()

    def save_state(self, path: Union[str, Path], o1_mas: Optional[Dict[str, Any]] = None) -> None:
        """Save current state to file (MessagePack)."""
        with self._lock:
            path = Path(path)

            # Convert deques to lists for serialization
            state_to_save = self.state.copy()
            state_to_save["price_history"] = list(self.state["price_history"])

            # Handle HMA input history deques
            for key, value in self.state.items():
                if key.startswith("hma_input_history_") and isinstance(value, deque):
                    state_to_save[key] = list(value)

            # Collect O(1) MA states if used
            if self.use_o1_mas and o1_mas:
                state_to_save["o1_mas_state"] = {}
                for k, v in o1_mas.items():
                    try:
                        if v is not None:
                            state_to_save["o1_mas_state"][k] = v.get_state()
                        else:
                            log_warn(f"O(1) MA object for {k} is None, skipping state save")
                    except Exception as e:
                        log_warn(f"Failed to save O(1) MA state for {k}: {e}")

            payload = {
                "version": "1.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "config": self.config,
                "state": state_to_save,
            }

            with open(path, "wb") as f:
                f.write(msgpack.packb(payload))

            log_debug(f"State saved to {path}")

    def load_state(self, path: Union[str, Path], o1_mas: Optional[Dict[str, Any]] = None) -> None:
        """Load state from file."""
        path = Path(path)
        with open(path, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=False)

        state_data = payload["state"]
        self.state = state_data

        # Ensure new keys exist
        self.state.setdefault("signals_l1", {})
        self.state.setdefault("equity_l1", {})
        self.state.setdefault("layer1_signals", {})
        self.state.setdefault("equity", None)
        self.state.setdefault("signal", None)
        self.state.setdefault("average_signal", None)
        self.state.setdefault("bar_index", None)

        # Convert lists back to deques
        max_base_length = max(self.ma_length.values())
        from .constants import ROBUSTNESS_OFFSETS

        max_offset = ROBUSTNESS_OFFSETS.get(self.robustness, 6)
        max_history = max_base_length + max_offset + 10

        self.state["price_history"] = deque(state_data["price_history"], maxlen=max_history)

        # Restore HMA input history deques
        for key, value in state_data.items():
            if key.startswith("hma_input_history_"):
                # Determine sqrt_len logic if needed, but for generic restore:
                # We need maxlen. Ideally we re-calculate it or just use the list len as hint?
                # Using maxlen based on implementation logic:
                # From update_hma: sqrt_len = int(np.sqrt(ln))
                # It's safer to re-create deque with maxlen from current config if possible,
                # but since we have multiple variations, we might just assume the saved list is sufficient.
                # Let's try to infer maxlen from the list length + buffer or just use a large enough one.
                # Or better: Recalculate it properly if we knew which variation 'i' it corresponds to.
                # Since we don't easily know 'i' to 'length' mapping here without re-doing diflen,
                # we will trust the list length for now or set a safe upper bound.
                # Actually, in incremental_atc.py it used:
                # sqrt_len = max(1, int(np.sqrt(instance.ma_length["hma"]))) -> THIS WAS WRONG/SIMPLIFIED in original code?
                # Original code only had one 'hma_input_history' (no index).
                # My new code uses `hma_input_history_{i}`.
                # Let's iterate and restore.

                # NOTE: HMA State Restore Heuristic
                # Ideally, we should recalculate maxlen = sqrt(length) for each variation.
                # However, since we don't have easy access to variation lengths here,
                # we use a heuristic: len(value) + 10 buffer.
                # sqrt(length) is small (e.g. sqrt(28) ~ 5), so +10 is plenty.
                # This ensures the deque doesn't overflow prematurely upon restore.
                self.state[key] = deque(value, maxlen=len(value) + 10)  # +10 buffer

        # Restore O(1) MA internal states
        if self.use_o1_mas and o1_mas:
            if "o1_mas_state" in state_data:
                for k, v in state_data["o1_mas_state"].items():
                    if k in o1_mas:
                        o1_mas[k].set_state(v)
            else:
                # Fallback to replay
                log_debug("No O(1) state found, falling back to replay...")
                for ma_obj in o1_mas.values():
                    ma_obj.reset()
                for price in self.state["price_history"]:
                    for ma_obj in o1_mas.values():
                        ma_obj.update(price)

        self.state["initialized"] = True

    def extract_state_from_results(
        self,
        results: Dict[str, pd.Series],
        prices: pd.Series,
        ma_tuples: Dict[str, tuple],
    ):
        """Extract state from full calculation results."""
        # Extract MA values
        for ma_type, ma_tuple in ma_tuples.items():
            if ma_tuple is not None and len(ma_tuple) == 9:
                self.state["ma_values"][ma_type.lower()] = [ma.iloc[-1] for ma in ma_tuple]
            elif ma_tuple is not None:
                self.state["ma_values"][ma_type.lower()] = [ma_tuple[0].iloc[-1]] * 9

        # Extract EMA2 for DEMA
        ema_list = self.state["ma_values"].get("ema")
        dema_list = self.state["ma_values"].get("dema")
        if ema_list is not None and dema_list is not None:
            self.state["ema2_values"]["dema"] = [2 * ema_list[i] - dema_list[i] for i in range(9)]

        # Extract per-variation Layer 1 signals and equities
        try:
            from modules.adaptive_trend_LTS_mini.core.process_layer1.layer1_signal import (
                _layer1_signal_for_ma,
            )
            from modules.adaptive_trend_LTS_mini.utils.rate_of_change import rate_of_change

            from .constants import get_scaled_params

            L_scaled, De_scaled = get_scaled_params(self.config)
            R = rate_of_change(prices)

            for ma_type, ma_tuple in ma_tuples.items():
                if ma_tuple is None:
                    continue
                try:
                    signal_series, signals_tuple, equity_tuple = _layer1_signal_for_ma(
                        prices=prices,
                        ma_tuple=ma_tuple,
                        L=L_scaled,
                        De=De_scaled,
                        R=R,
                    )
                    self.state["signals_l1"][ma_type.lower()] = [float(s.iloc[-1]) for s in signals_tuple]
                    self.state["equity_l1"][ma_type.lower()] = [float(e.iloc[-1]) for e in equity_tuple]
                    if signal_series is not None and len(signal_series) > 0:
                        self.state["layer1_signals"][ma_type] = float(signal_series.iloc[-1])
                except Exception as e:
                    log_warn(f"Failed to extract layer1 state for {ma_type}: {e}")
        except Exception as e:
            log_warn(f"Failed to extract per-variation layer1 state: {e}")

        # Get Layer 2 equities
        if "EMA_S" in results:
            self.state["equity"] = {
                "EMA": results["EMA_S"].iloc[-1],
                "HMA": results["HMA_S"].iloc[-1],
                "WMA": results["WMA_S"].iloc[-1],
                "DEMA": results["DEMA_S"].iloc[-1],
                "LSMA": results["LSMA_S"].iloc[-1],
                "KAMA": results["KAMA_S"].iloc[-1],
            }

        if "Average_Signal" in results and len(results["Average_Signal"]) > 0:
            self.state["average_signal"] = float(results["Average_Signal"].iloc[-1])
            self.state["signal"] = self.state["average_signal"]

        # Populate price history
        self.state["price_history"].clear()
        self.state["price_history"].extend(prices.tolist())
        self.state["bar_index"] = len(prices) - 1

        # Reconstruct HMA internal state (input history)
        self._reconstruct_hma_state(prices)

    def _reconstruct_hma_state(self, prices: pd.Series):
        """Reconstruct HMA internal state from price history."""
        try:
            import pandas_ta as ta

            from modules.adaptive_trend_LTS_mini.utils.diflen import diflen

            length = self.ma_length["hma"]
            diflen_res = diflen(length, robustness=self.robustness)
            assert diflen_res is not None, "diflen returned None"
            L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen_res
            lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
            sqrt_lengths = [max(1, int(np.sqrt(ln))) for ln in lengths]
            half_lengths = [max(1, ln // 2) for ln in lengths]

            # We need enough history to calculate WMA.
            # For each variation i, we need the last sqrt_lengths[i] values of the 'raw' series.
            # Each 'raw' value at time t depends on WMA(n) and WMA(n/2) at time t.
            # So we need to calculate WMA(n) and WMA(n/2) for the last max(sqrt_len) bars.

            max_sqrt_len = max(sqrt_lengths)
            # Ensure we don't go out of bounds
            available_bars = len(prices)
            bars_to_recalc = min(available_bars, max_sqrt_len + max(lengths))  # Safety buffer

            if bars_to_recalc < 1:
                return

            recent_prices = prices.iloc[-bars_to_recalc:]

            for i, ln in enumerate(lengths):
                half_len = half_lengths[i]
                sqrt_len = sqrt_lengths[i]

                # We need the last 'sqrt_len' inputs
                # raw = 2 * wma(n/2) - wma(n)

                # Calculate WMA(n) and WMA(n/2) on recent prices
                # using pandas_ta for correctness matching batch
                wma_n = ta.wma(recent_prices, length=ln)
                wma_half = ta.wma(recent_prices, length=half_len)

                if wma_n is None or wma_half is None:
                    continue

                raw_series = 2 * wma_half - wma_n

                # Get the last 'sqrt_len' valid values
                # Note: wma will have NaNs at the start.
                raw_valid = raw_series.dropna()

                if len(raw_valid) > 0:
                    last_values = raw_valid.iloc[-sqrt_len:].tolist()

                    hma_hist_key = f"hma_input_history_{i}"
                    self.state[hma_hist_key] = deque(last_values, maxlen=sqrt_len)

        except Exception as e:
            log_warn(f"Failed to reconstruct HMA state: {e}")
