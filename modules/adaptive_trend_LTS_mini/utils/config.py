from dataclasses import dataclass
from typing import Any, Dict

"""Configuration for Adaptive Trend Classification (ATC) analysis."""


@dataclass
class ATCConfig:
    """Configuration for Adaptive Trend Classification (ATC) analysis.

    This class holds both unscaled and scaled parameter values for ATC calculations.

    Important Parameter Notes:
        - lambda_param (unscaled): Use same value as compute_atc_signals(lambda_param=...)
          The scaling (divide by 1000) is applied internally by lambda_scaled property
          or by compute_atc_signals function.
        - decay (unscaled): Use same value as compute_atc_signals(decay_rate=...)
          The scaling (divide by 100) is applied internally by decay_scaled property
          or by compute_atc_signals function.

    Example:
        >>> config = ATCConfig(lambda_param=0.02, decay=0.03)
        >>> # Use unscaled values directly with compute_atc_signals
        >>> result = compute_atc_signals(prices, lambda_param=config.lambda_param, decay_rate=config.decay)
        >>> # Or use scaled values for manual calculations
        >>> scaled_lambda = config.lambda_scaled  # 0.00002
        >>> scaled_decay = config.decay_scaled    # 0.0003
    """

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
    lambda_param: float = 0.02  # UNSCALED lambda value (will be divided by 1000 internally)
    decay: float = 0.03  # UNSCALED decay value (will be divided by 100 internally)
    cutout: int = 0
    strategy_mode: bool = False  # Set to True for shifted, non-repainting signals (Strategy View)

    @property
    def lambda_scaled(self) -> float:
        """Lambda scaled for calculations (divided by 1000 to match PineScript).

        Returns:
            Scaled lambda value: lambda_param / 1000
        """
        return self.lambda_param / 1000

    @property
    def decay_scaled(self) -> float:
        """Decay scaled for calculations (divided by 100 to match PineScript).

        Returns:
            Scaled decay value: decay / 100
        """
        return self.decay / 100

    # Signal threshold parameters
    long_threshold: float = 0.1
    short_threshold: float = -0.1
    equity_floor: float = 0.25  # Minimum equity value to prevent numerical instability

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

    # Approximate calculation parameters
    use_approximate: bool = False  # Use approximate calculations for faster scanning
    approximate_threshold: float = 0.05  # Threshold for approximate calculations
    use_adaptive_approximate: bool = False  # Enable adaptive approximation based on volatility
    approximate_volatility_window: int = 20  # Window size for volatility calculation
    approximate_volatility_factor: float = 1.0  # Scaling factor for volatility-based approximation

    # Cache compression parameters
    use_compression: bool = False  # Enable blosc compression for disk cache
    compression_level: int = 5  # Compression level (0-9, higher = more compression)
    compression_algorithm: str = "blosclz"  # Compression algorithm name

    # Memory optimization parameters
    use_memory_mapped: bool = False  # Enable memory-mapped arrays for large datasets

    # Code generation / JIT specialization parameters
    use_codegen_specialization: bool = False  # Enable JIT specialization for known hot path configs

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "ema_len": self.ema_len,
            "hma_len": self.hma_len,
            "wma_len": self.wma_len,
            "dema_len": self.dema_len,
            "lsma_len": self.lsma_len,
            "kama_len": self.kama_len,
            "ema_w": self.ema_w,
            "hma_w": self.hma_w,
            "wma_w": self.wma_w,
            "dema_w": self.dema_w,
            "lsma_w": self.lsma_w,
            "kama_w": self.kama_w,
            "robustness": self.robustness,
            "lambda_param": self.lambda_param,
            "decay": self.decay,
            "cutout": self.cutout,
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "equity_floor": self.equity_floor,
            "calculation_source": self.calculation_source,
            "limit": self.limit,
            "timeframe": self.timeframe,
            "batch_size": self.batch_size,
            "precision": self.precision,
            "parallel_l1": self.parallel_l1,
            "parallel_l2": self.parallel_l2,
            "use_rust_backend": self.use_rust_backend,
            "use_approximate": self.use_approximate,
            "approximate_threshold": self.approximate_threshold,
            "use_adaptive_approximate": self.use_adaptive_approximate,
            "approximate_volatility_window": self.approximate_volatility_window,
            "approximate_volatility_factor": self.approximate_volatility_factor,
            "use_compression": self.use_compression,
            "compression_level": self.compression_level,
            "compression_algorithm": self.compression_algorithm,
            "use_memory_mapped": self.use_memory_mapped,
            "use_codegen_specialization": self.use_codegen_specialization,
            "strategy_mode": self.strategy_mode,
        }


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
        use_approximate=params.get("use_approximate", False),
        approximate_threshold=params.get("approximate_threshold", 0.05),
        use_adaptive_approximate=params.get("use_adaptive_approximate", False),
        approximate_volatility_window=params.get("approximate_volatility_window", 20),
        approximate_volatility_factor=params.get("approximate_volatility_factor", 1.0),
        use_compression=params.get("use_compression", False),
        compression_level=params.get("compression_level", 5),
        compression_algorithm=params.get("compression_algorithm", "blosclz"),
        use_memory_mapped=params.get("use_memory_mapped", False),
        use_codegen_specialization=params.get("use_codegen_specialization", False),
    )
