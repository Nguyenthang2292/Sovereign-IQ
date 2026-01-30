"""Base types for sampling strategies."""

from enum import Enum


class SamplingStrategy(str, Enum):
    """Available sampling strategies for Stage 0."""

    RANDOM = "random"
    VOLUME_WEIGHTED = "volume_weighted"
    STRATIFIED = "stratified"
    TOP_N_HYBRID = "top_n_hybrid"
    SYSTEMATIC = "systematic"
    LIQUIDITY_WEIGHTED = "liquidity_weighted"
