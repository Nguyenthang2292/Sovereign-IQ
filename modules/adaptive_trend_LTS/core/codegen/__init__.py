"""Code generation and JIT specialization for ATC.

This package provides code generation capabilities for optimizing ATC
computations through JIT compilation and configuration specialization.
"""

from modules.adaptive_trend_LTS.core.codegen.specialization import (
    SpecializedConfigKey,
    compute_atc_specialized,
    get_specialized_compute_fn,
    is_config_specializable,
)

__all__ = [
    "SpecializedConfigKey",
    "get_specialized_compute_fn",
    "compute_atc_specialized",
    "is_config_specializable",
]
