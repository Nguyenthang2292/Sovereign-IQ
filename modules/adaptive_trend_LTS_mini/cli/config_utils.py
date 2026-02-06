"""
Configuration utilities for ATC CLI.
"""

from argparse import Namespace
from typing import TypedDict, cast

from modules.common.utils import extract_dict_from_namespace


class ATCParams(TypedDict, total=False):
    """Type definition for ATC parameters dictionary.

    This provides type safety for the parameters passed to ATC analysis functions.
    All fields are optional (total=False) to support flexible parameter extraction.
    """

    limit: int
    ema_len: int
    hma_len: int
    wma_len: int
    dema_len: int
    lsma_len: int
    kama_len: int
    robustness: str
    lambda_param: float
    decay: float
    cutout: int
    long_threshold: float
    short_threshold: float
    # Performance & Backend
    use_rust_backend: bool
    batch_processing: bool
    fast_mode: bool
    precision: str
    use_cache: bool
    # Approximate Scanning
    use_approximate: bool
    use_adaptive_approximate: bool
    approximate_volatility_window: int
    approximate_volatility_factor: float
    approximate_threshold: float


def get_atc_params(args: Namespace) -> ATCParams:
    """Extract ATC parameters from arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        ATCParams: Typed dictionary containing ATC configuration parameters
    """
    atc_param_keys = [
        "limit",
        "ema_len",
        "hma_len",
        "wma_len",
        "dema_len",
        "lsma_len",
        "kama_len",
        "robustness",
        "lambda_param",
        "decay",
        "cutout",
        "long_threshold",
        "short_threshold",
        # Performance & Backend
        "use_rust_backend",
        "batch_processing",
        "fast_mode",
        "precision",
        "use_cache",
        # Approximate Scanning
        "use_approximate",
        "use_adaptive_approximate",
        "approximate_volatility_window",
        "approximate_volatility_factor",
        "approximate_threshold",
    ]
    return cast(ATCParams, extract_dict_from_namespace(args, atc_param_keys))
