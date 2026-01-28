"""JIT Specialization for ATC Configurations.

This module provides code generation and JIT specialization for common
ATC configurations to reduce configuration overhead and improve performance
for frequently used configs.
"""

from __future__ import annotations

# Import specialized implementations
import importlib.util
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.utils.config import ATCConfig

# Check if specialized implementations are available
# We use find_spec to avoid importing the module (and triggering unused import warnings)
# effectively checking if both numba and the specialization module exist.
_numba_spec = importlib.util.find_spec("numba")
_specialized_spec = importlib.util.find_spec("modules.adaptive_trend_LTS.core.codegen.numba_specialized")

NUMBA_SPECIALIZATION_AVAILABLE = (_numba_spec is not None) and (_specialized_spec is not None)


@dataclass
class SpecializedConfigKey:
    """Hashable key for identifying specialized configurations.

    Attributes:
        ma_type: Primary MA type (e.g., "EMA", "HMA", or "ALL" for default)
        length: MA length
        robustness: Robustness level
        mode: Specialization mode ("default", "ema_only", "short_length", etc.)
    """

    ma_type: str
    length: int
    robustness: str
    mode: str

    def __hash__(self) -> int:
        return hash((self.ma_type, self.length, self.robustness, self.mode))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpecializedConfigKey):
            return False
        return (
            self.ma_type == other.ma_type
            and self.length == other.length
            and self.robustness == other.robustness
            and self.mode == other.mode
        )


def _get_config_key(config: ATCConfig, mode: str = "default") -> SpecializedConfigKey:
    """Extract specialization key from ATCConfig.

    Args:
        config: ATC configuration
        mode: Specialization mode (default, ema_only, short_length, etc.)

    Returns:
        SpecializedConfigKey for caching/lookup
    """
    if mode == "ema_only":
        return SpecializedConfigKey("EMA", config.ema_len, config.robustness, mode)
    elif mode == "short_length":
        return SpecializedConfigKey("ALL", config.ema_len, config.robustness, mode)
    else:
        return SpecializedConfigKey("ALL", config.ema_len, config.robustness, mode)


def get_specialized_compute_fn(
    config: ATCConfig,
    mode: str = "default",
    use_codegen: bool = True,
) -> Optional[Callable[[pd.Series], dict[str, pd.Series]]]:
    """Get or create a specialized compute function for given config.

    This factory function returns a specialized compute function that is
    JIT-compiled and optimized for specific configuration. Specialized
    functions are cached for reuse.

    Args:
        config: ATC configuration to specialize for
        mode: Specialization mode (default, ema_only, short_length, etc.)
        use_codegen: If False, returns None (use generic path)

    Returns:
        Specialized compute function or None if specialization not enabled
        or not available for this config

    Example:
        >>> config = ATCConfig(ema_len=28, robustness="Medium")
        >>> compute_fn = get_specialized_compute_fn(config, mode="ema_only")
        >>> if compute_fn:
        ...     result = compute_fn(prices)
        ... else:
        ...     result = compute_atc_signals(prices, **config_to_dict(config))
    """
    if not use_codegen:
        return None

    if not NUMBA_SPECIALIZATION_AVAILABLE:
        return None

    # Import specialized implementations only when needed
    try:
        from modules.adaptive_trend_LTS.core.codegen.numba_specialized import (
            compute_ema_only_atc,
        )
    except ImportError:
        return None

    config_key = _get_config_key(config, mode)

    # Check if we have a specialized function for this config
    if mode == "ema_only":
        # Return EMA-only specialized function
        def _ema_only_specialized(prices: pd.Series) -> dict[str, pd.Series]:
            # Convert to numpy array
            prices_arr = prices.values.astype(np.float64)

            # Compute using JIT-compiled function
            ema_signal, ema_equity = compute_ema_only_atc(
                prices_arr,
                ema_len=config.ema_len,
                lambda_param=config.lambda_param,
                decay=config.decay,
                long_threshold=config.long_threshold,
                short_threshold=config.short_threshold,
                cutout=config.cutout,
                strategy_mode=config.strategy_mode,
            )

            # Return in expected format
            result: dict[str, pd.Series] = {}
            result["EMA_Signal"] = pd.Series(ema_signal, index=prices.index)
            result["EMA_S"] = pd.Series(ema_equity, index=prices.index)
            result["Average_Signal"] = pd.Series(ema_signal, index=prices.index)

            return result

        return _ema_only_specialized

    # For other modes, return None (not yet implemented)
    return None

    config_key = _get_config_key(config, mode)

    # Check if we have a specialized function for this config
    # For now, return None - specialization will be implemented in Task 3
    # TODO: Implement actual specialized functions
    return None


def compute_atc_specialized(
    prices: pd.Series,
    config: ATCConfig,
    mode: str = "default",
    use_codegen_specialization: bool = True,
    fallback_to_generic: bool = True,
    **kwargs: Any,
) -> dict[str, pd.Series]:
    """Compute ATC signals using specialized or generic path.

    This is the main entrypoint for codegen specialization. It attempts to
    use a specialized, JIT-compiled path for known configurations, with a
    safe fallback to the generic path.

    Args:
        prices: Price series
        config: ATC configuration
        mode: Specialization mode (default, ema_only, short_length, etc.)
        use_codegen_specialization: If True, try to use specialized path
        fallback_to_generic: If True, fall back to generic path if specialization fails
        **kwargs: Additional parameters for compute_atc_signals (if using generic path)

    Returns:
        Dictionary with ATC signals and equities

    Raises:
        ValueError: If specialization fails and fallback_to_generic=False

    Example:
        >>> config = ATCConfig(ema_len=28, robustness="Medium")
        >>> result = compute_atc_specialized(
        ...     prices,
        ...     config,
        ...     use_codegen_specialization=True,
        ...     fallback_to_generic=True
        ... )
    """
    if use_codegen_specialization:
        specialized_fn = get_specialized_compute_fn(config, mode)

        if specialized_fn is not None:
            try:
                # Use specialized path
                return specialized_fn(prices)
            except Exception as e:
                if fallback_to_generic:
                    # Log warning and fall back to generic
                    pass
                else:
                    raise ValueError(f"Specialized path failed: {e}")

    # Fallback to generic path
    from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import (
        compute_atc_signals as generic_compute,
    )

    # Convert ATCConfig to dict
    config_dict = {
        "ema_len": config.ema_len,
        "hull_len": config.hma_len,
        "wma_len": config.wma_len,
        "dema_len": config.dema_len,
        "lsma_len": config.lsma_len,
        "kama_len": config.kama_len,
        "ema_w": config.ema_w,
        "hma_w": config.hma_w,
        "wma_w": config.wma_w,
        "dema_w": config.dema_w,
        "lsma_w": config.lsma_w,
        "kama_w": config.kama_w,
        "robustness": config.robustness,
        "lambda_param": config.lambda_param,
        "decay": config.decay,
        "cutout": config.cutout,
        "long_threshold": config.long_threshold,
        "short_threshold": config.short_threshold,
        "strategy_mode": config.strategy_mode,
        "precision": config.precision,
        "use_rust_backend": config.use_rust_backend,
    }

    # Merge with any additional kwargs
    config_dict.update(kwargs)

    return generic_compute(prices, **config_dict)


def is_config_specializable(config: ATCConfig, mode: str = "default") -> bool:
    """Check if a configuration can be specialized.

    Args:
        config: ATC configuration to check
        mode: Specialization mode

    Returns:
        True if configuration can be specialized, False otherwise
    """
    if not NUMBA_SPECIALIZATION_AVAILABLE:
        return False

    # For now, only certain configs are specializable
    # This will be expanded in future tasks
    specializable_modes = ["ema_only"]

    if mode not in specializable_modes:
        return False

    # Check if it matches known hot path configs
    if mode == "ema_only":
        # EMA-only: all lengths are specializable (Task 3 implementation)
        return True
    else:
        # Other modes not yet implemented
        return False


__all__ = [
    "SpecializedConfigKey",
    "get_specialized_compute_fn",
    "compute_atc_specialized",
    "is_config_specializable",
]
