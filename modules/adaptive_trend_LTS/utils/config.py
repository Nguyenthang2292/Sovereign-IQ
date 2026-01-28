from dataclasses import dataclass
from typing import Any, Dict

"""Configuration for Adaptive Trend Classification (ATC) analysis."""


@dataclass
class ATCConfig:
    """Configuration for Adaptive Trend Classification (ATC) analysis."""

    # Moving Average lengths
    ema_len: int = 28
    hma_len: int = 28
    wma_len: int = 28
    dema_len: int = 28
    lsma_len: int = 28
    kama_len: int = 28

    # Moving Average weights
    ema_w: float = 1.0
    hma_w: float = 1.0
    wma_w: float = 1.0
    dema_w: float = 1.0
    lsma_w: float = 1.0
    kama_w: float = 1.0

    # ATC parameters
    robustness: str = "Medium"  # "Narrow", "Medium", or "Wide"
    lambda_param: float = 0.02
    decay: float = 0.03
    cutout: int = 0
    strategy_mode: bool = False  # Set to True for shifted, non-repainting signals (Strategy View)

    @property
    def lambda_scaled(self) -> float:
        """Lambda scaled for calculations (divided by 1000 to match PineScript)."""
        return self.lambda_param / 1000

    @property
    def decay_scaled(self) -> float:
        """Decay scaled for calculations (divided by 100 to match PineScript)."""
        return self.decay / 100

    # Signal threshold parameters
    long_threshold: float = 0.1
    short_threshold: float = -0.1

    # Calculation source for Moving Averages
    calculation_source: str = "close"  # "close", "open", "high", "low"

    # Data parameters
    limit: int = 1500
    timeframe: str = "15m"

    # Performance optimization parameters
    batch_size: int = 100  # Number of symbols to process in each batch before forcing GC
    precision: str = "float64"  # "float64" or "float32"
    parallel_l1: bool = True  # Level 1 parallelism (intra-symbol)
    parallel_l2: bool = True  # Level 2 parallelism (inter-symbol)
    use_rust_backend: bool = True  # Use Rust backend (CPU parallelism with Rayon)

    # Cache compression parameters
    use_compression: bool = False  # Enable blosc compression for disk cache
    compression_level: int = 5  # Compression level (0-9, higher = more compression)
    compression_algorithm: str = "blosclz"  # Compression algorithm name

    # Memory optimization parameters
    use_memory_mapped: bool = False  # Enable memory-mapped arrays for large datasets


def create_atc_config_from_dict(
    params: Dict[str, Any],
    timeframe: str = "15m",
) -> ATCConfig:
    """
    Create ATCConfig from a dictionary of parameters.

    Args:
        params: Dictionary containing ATC parameters
        timeframe: Timeframe for data (default: "15m")

    Returns:
        ATCConfig instance with parameters from dict
    """
    return ATCConfig(
        timeframe=timeframe,
        limit=params.get("limit", 1500),
        ema_len=params.get("ema_len", 28),
        hma_len=params.get("hma_len", 28),
        wma_len=params.get("wma_len", 28),
        dema_len=params.get("dema_len", 28),
        lsma_len=params.get("lsma_len", 28),
        kama_len=params.get("kama_len", 28),
        ema_w=params.get("ema_w", 1.0),
        hma_w=params.get("hma_w", 1.0),
        wma_w=params.get("wma_w", 1.0),
        dema_w=params.get("dema_w", 1.0),
        lsma_w=params.get("lsma_w", 1.0),
        kama_w=params.get("kama_w", 1.0),
        robustness=params.get("robustness", "Medium"),
        lambda_param=params.get("lambda_param", 0.02),
        decay=params.get("decay", 0.03),
        cutout=params.get("cutout", 0),
        long_threshold=params.get("long_threshold", 0.1),
        short_threshold=params.get("short_threshold", -0.1),
        calculation_source=params.get("calculation_source", "close"),
        strategy_mode=params.get("strategy_mode", False),
        batch_size=params.get("batch_size", 100),
        precision=params.get("precision", "float64"),
        parallel_l1=params.get("parallel_l1", True),
        parallel_l2=params.get("parallel_l2", True),
        use_rust_backend=params.get("use_rust_backend", params.get("prefer_gpu", True)),  # Backward compat
        use_compression=params.get("use_compression", False),
        compression_level=params.get("compression_level", 5),
        compression_algorithm=params.get("compression_algorithm", "blosclz"),
        use_memory_mapped=params.get("use_memory_mapped", False),
    )
