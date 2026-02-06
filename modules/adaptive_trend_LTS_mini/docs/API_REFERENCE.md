# Adaptive Trend Classification LTS - API Reference

**Version:** 1.0 (CPU-only mini version)
**Last Updated:** 2026-02-06

This document provides a comprehensive reference for the public API of the `adaptive_trend_LTS_mini` module.

---

## Table of Contents

1. [Core Functions](#core-functions)
   - [compute_atc_signals](#compute_atc_signals)
   - [calculate_layer2_equities](#calculate_layer2_equities)
2. [Classes](#classes)
   - [ATCAnalyzer](#atcanalyzer)
   - [IncrementalATC](#incrementalatc)
   - [MultiTimeframeIncrementalATC](#multitimeframeincrementalatc)
   - [ATCConfig](#atcconfig)
3. [Configuration](#configuration)
   - [ATCConfig Dataclass](#atcconfig-dataclass)
   - [create_atc_config_from_dict](#create_atc_config_from_dict)
4. [Batch Processing](#batch-processing)
   - [process_symbols_batch_dask](#process_symbols_batch_dask)
5. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Custom Configuration](#custom-configuration)
   - [Incremental Updates](#incremental-updates)
   - [Batch Processing](#batch-processing-example)

---

## Core Functions

### compute_atc_signals

The main entry point for computing Adaptive Trend Classification signals.

```python
def compute_atc_signals(
    prices: pd.Series,
    src: Optional[pd.Series] = None,
    *,
    # Moving Average lengths
    ema_len: int = 28,
    hma_len: int = 28,
    wma_len: int = 28,
    dema_len: int = 28,
    lsma_len: int = 28,
    kama_len: int = 28,
    # Moving Average weights
    ema_w: float = 1.0,
    hma_w: float = 1.0,
    wma_w: float = 1.0,
    dema_w: float = 1.0,
    lsma_w: float = 1.0,
    kama_w: float = 1.0,
    # ATC parameters
    robustness: str = "Medium",
    lambda_param: float = 0.02,
    decay_rate: float = 0.03,
    cutout: int = 0,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
    strategy_mode: bool = False,
    # Performance options
    parallel_l1: Optional[bool] = None,
    parallel_l2: Optional[bool] = True,
    precision: str = "float64",
    use_rust_backend: bool = True,
    use_cache: bool = True,
    fast_mode: bool = True,
    # Approximation options
    use_approximate: bool = False,
    approximate_threshold: float = 0.05,
    use_adaptive_approximate: bool = False,
    approximate_volatility_window: int = 20,
    approximate_volatility_factor: float = 1.0,
    # Advanced options
    equity_floor: float = 0.25,
) -> dict[str, pd.Series]:
    """
    Compute Adaptive Trend Classification (ATC) signals.

    This is the main orchestration function that:
    1. Validates and scales parameters
    2. Computes moving averages (6 types)
    3. Calculates Layer 1 signals for each MA type
    4. Calculates Layer 2 equity weights
    5. Combines signals into final Average_Signal

    Args:
        prices: Price series (typically close prices)
        src: Optional alternative source series (if None, uses prices)

        ema_len: Length for EMA calculation (default: 28)
        hma_len: Length for HMA calculation (default: 28)
        wma_len: Length for WMA calculation (default: 28)
        dema_len: Length for DEMA calculation (default: 28)
        lsma_len: Length for LSMA calculation (default: 28)
        kama_len: Length for KAMA calculation (default: 28)

        ema_w: Initial weight for EMA (default: 1.0)
        hma_w: Initial weight for HMA (default: 1.0)
        wma_w: Initial weight for WMA (default: 1.0)
        dema_w: Initial weight for DEMA (default: 1.0)
        lsma_w: Initial weight for LSMA (default: 1.0)
        kama_w: Initial weight for KAMA (default: 1.0)

        robustness: Sensitivity mode - "Narrow", "Medium", or "Wide" (default: "Medium")
                   Controls the offset used for additional MA calculations
        lambda_param: UNSCALED lambda growth rate (default: 0.02)
                     Automatically scaled by dividing by 1000
        decay_rate: UNSCALED decay rate (default: 0.03)
                   Automatically scaled by dividing by 100
        cutout: Number of initial bars to skip (default: 0)
        long_threshold: Threshold for long signal (default: 0.1)
        short_threshold: Threshold for short signal (default: -0.1)
        strategy_mode: If True, shifts signals for non-repainting strategy view (default: False)

        parallel_l1: Enable parallel processing for Layer 1 (default: auto-detect)
        parallel_l2: Enable parallel processing for Layer 2 (default: True)
        precision: Floating point precision - "float64" or "float32" (default: "float64")
        use_rust_backend: Use Rust-accelerated backend (default: True)
        use_cache: Enable MA result caching (default: True)
        fast_mode: Skip memory tracking overhead (default: True)

        use_approximate: Use approximate MA calculations for speed (default: False)
        approximate_threshold: Tolerance for approximate calculations (default: 0.05)
        use_adaptive_approximate: Use volatility-adjusted approximations (default: False)
        approximate_volatility_window: Window for volatility calculation (default: 20)
        approximate_volatility_factor: Scaling factor for volatility (default: 1.0)

        equity_floor: Minimum equity value to prevent numerical instability (default: 0.25)

    Returns:
        Dictionary containing:
            - "EMA_Signal": Layer 1 signal for EMA
            - "HMA_Signal": Layer 1 signal for HMA
            - "WMA_Signal": Layer 1 signal for WMA
            - "DEMA_Signal": Layer 1 signal for DEMA
            - "LSMA_Signal": Layer 1 signal for LSMA
            - "KAMA_Signal": Layer 1 signal for KAMA
            - "EMA_S": Layer 2 equity weight for EMA
            - "HMA_S": Layer 2 equity weight for HMA
            - "WMA_S": Layer 2 equity weight for WMA
            - "DEMA_S": Layer 2 equity weight for DEMA
            - "LSMA_S": Layer 2 equity weight for LSMA
            - "KAMA_S": Layer 2 equity weight for KAMA
            - "Average_Signal": Final combined signal

        All values are pd.Series with same index as input prices.

    Raises:
        ValueError: If inputs are invalid (empty prices, invalid robustness mode, etc.)

    Example:
        >>> import pandas as pd
        >>> from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
        >>>
        >>> # Prepare price data
        >>> prices = pd.Series([100, 101, 102, 103, 104])
        >>>
        >>> # Compute ATC signals
        >>> results = compute_atc_signals(prices)
        >>>
        >>> # Access results
        >>> average_signal = results["Average_Signal"]
        >>> ema_signal = results["EMA_Signal"]
        >>> ema_equity = results["EMA_S"]
    """
```

**Parameter Scaling Notes:**

The `lambda_param` and `decay_rate` parameters are automatically scaled internally:
- `lambda_param` is divided by 1000 (e.g., 0.02 → 0.00002)
- `decay_rate` is divided by 100 (e.g., 0.03 → 0.0003)

Always provide the UNSCALED values to match PineScript behavior.

**Robustness Modes:**

The `robustness` parameter controls the offset used for calculating additional moving averages:
- `"Narrow"`: Small offsets (faster, more responsive)
- `"Medium"`: Balanced offsets (default)
- `"Wide"`: Large offsets (slower, more stable)

**Performance Tips:**

1. Enable `use_rust_backend=True` for 2-10x speedup
2. Use `fast_mode=True` to skip memory tracking overhead
3. Enable `use_approximate=True` for exploratory scanning (trades accuracy for speed)
4. Set `parallel_l1=True` for datasets > 5000 bars
5. Use `precision="float32"` for memory-constrained environments

---

### calculate_layer2_equities

Calculate Layer 2 equity curves based on Layer 1 signal performance.

```python
def calculate_layer2_equities(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: list,
    rate_of_change_series: pd.Series,
    lambda_val: float,
    decay_val: float,
    cutout: int = 0,
    parallel: bool = True,
    precision: str = "float64",
    use_rust_backend: bool = True,
    floor_val: Optional[float] = None,
) -> Dict[str, pd.Series]:
    """
    Calculate Layer 2 equity curves based on Layer 1 signal performance.

    This function calculates the equity curve for each MA type based on how well
    its Layer 1 signal performed. The equity curve serves as a dynamic weight
    in the final signal aggregation.

    Args:
        layer1_signals: Dictionary of Layer 1 signals keyed by MA type (e.g., "EMA", "HMA")
        ma_configs: List of (ma_type, length, initial_weight) tuples
        rate_of_change_series: Rate of change series (calculated once and reused)
        lambda_val: SCALED lambda (growth rate) for exponential growth factor
        decay_val: SCALED decay factor for equity calculations
        cutout: Number of bars to skip at beginning (default: 0)
        parallel: If True, calculate equities in parallel (default: True)
        precision: Floating point precision - "float64" or "float32" (default: "float64")
        use_rust_backend: Use Rust-accelerated backend (default: True)
        floor_val: Minimum equity value (default: None, uses 0.25)

    Returns:
        Dictionary of Layer 2 equity curves keyed by MA type.
        Each equity curve is a pd.Series representing the cumulative performance
        of that MA type's signals.

    Raises:
        ValueError: If ma_configs contains invalid entries

    Note:
        Unlike compute_atc_signals(), this function expects SCALED lambda and decay values.
        If calling directly, divide lambda_param by 1000 and decay_rate by 100.

    Example:
        >>> from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import calculate_layer2_equities
        >>>
        >>> # Assume we have layer1_signals from previous calculation
        >>> ma_configs = [
        ...     ("EMA", 28, 1.0),
        ...     ("HMA", 28, 1.0),
        ... ]
        >>> lambda_scaled = 0.02 / 1000  # Scale lambda
        >>> decay_scaled = 0.03 / 100    # Scale decay
        >>>
        >>> equities = calculate_layer2_equities(
        ...     layer1_signals=layer1_signals,
        ...     ma_configs=ma_configs,
        ...     rate_of_change_series=rate_of_change,
        ...     lambda_val=lambda_scaled,
        ...     decay_val=decay_scaled,
        ... )
    """
```

---

## Classes

### ATCAnalyzer

Analyzer class for processing individual symbols with ATC signals.

```python
def analyze_symbol(
    symbol: str,
    data_fetcher: "DataFetcher",
    config: ATCConfig,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single symbol using ATC.

    This function computes ATC signals and returns the results. It does not
    handle display - that should be done by the calling code.

    Args:
        symbol: Symbol to analyze (e.g., "BTC/USDT")
        data_fetcher: DataFetcher instance for fetching OHLCV data
        config: ATCConfig containing all ATC parameters

    Returns:
        Dictionary containing analysis results with keys:
            - symbol: Symbol name
            - df: OHLCV DataFrame
            - atc_results: ATC signals dictionary (from compute_atc_signals)
            - current_price: Current price
            - exchange_label: Exchange identifier

        Returns None if analysis failed.

    Example:
        >>> from modules.adaptive_trend_LTS_mini.core.analyzer import analyze_symbol
        >>> from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
        >>> from modules.common.core.data_fetcher import DataFetcher
        >>>
        >>> # Initialize
        >>> config = ATCConfig(timeframe="15m", limit=1500)
        >>> data_fetcher = DataFetcher()
        >>>
        >>> # Analyze symbol
        >>> result = analyze_symbol("BTC/USDT", data_fetcher, config)
        >>>
        >>> if result:
        ...     print(f"Symbol: {result['symbol']}")
        ...     print(f"Current price: {result['current_price']}")
        ...     print(f"Average signal: {result['atc_results']['Average_Signal'].iloc[-1]}")
    """
```

---

### IncrementalATC

Incremental ATC calculator that maintains state between updates for live trading.

```python
class IncrementalATC:
    """
    Incremental ATC calculator that maintains state between updates.

    This class enables O(1) updates for new price bars without recalculating
    the entire history. Ideal for live trading applications.

    Attributes:
        config: Configuration dictionary
        state: Current state dictionary (MAs, signals, equities)
        state_manager: Internal state management
        o1_mas: O(1) MA objects (WMA, HMA, LSMA, KAMA)

    Example:
        >>> from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC
        >>> import pandas as pd
        >>>
        >>> # Configuration
        >>> config = {
        ...     "ema_len": 28,
        ...     "hma_len": 28,
        ...     "wma_len": 28,
        ...     "dema_len": 28,
        ...     "lsma_len": 28,
        ...     "kama_len": 28,
        ...     "robustness": "Medium",
        ...     "lambda_param": 0.02,
        ...     "decay": 0.03,
        ...     "use_o1_mas": True,
        ... }
        >>>
        >>> # Initialize with historical data
        >>> atc = IncrementalATC(config)
        >>> historical_prices = pd.Series([100, 101, 102, 103, 104])
        >>> initial_results = atc.initialize(historical_prices)
        >>>
        >>> # Update with new price (O(1) operation)
        >>> new_price = 105
        >>> updated_signal = atc.update(new_price)
        >>> print(f"New Average_Signal: {updated_signal}")
        >>>
        >>> # Save state for persistence
        >>> atc.save_state("atc_state.msgpack")
        >>>
        >>> # Load state later
        >>> restored_atc = IncrementalATC.load_state("atc_state.msgpack")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize incremental ATC with configuration.

        Args:
            config: Configuration dictionary with ATC parameters
                   Must include all MA lengths, robustness, lambda_param, decay, etc.
        """

    def initialize(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Initialize state with full calculation on historical data.

        Args:
            prices: Historical price series for initialization

        Returns:
            Dictionary of ATC signals (same format as compute_atc_signals)
        """

    def update(self, new_price: float) -> float:
        """
        Update state with new price bar and return new Average_Signal.

        This is an O(1) operation that only updates the necessary state
        without recalculating the entire history.

        Args:
            new_price: New price value

        Returns:
            New Average_Signal value
        """

    def save_state(self, path: Union[str, Path]) -> None:
        """
        Save current state to file for persistence.

        Args:
            path: Path to save state (msgpack format)
        """

    @classmethod
    def load_state(cls, path: Union[str, Path]) -> "IncrementalATC":
        """
        Load state from file and create restored IncrementalATC instance.

        Args:
            path: Path to saved state file

        Returns:
            IncrementalATC instance with restored state
        """

    @property
    def state(self) -> Dict[str, Any]:
        """Access state dictionary."""

    @state.setter
    def state(self, value: Dict[str, Any]):
        """Set state dictionary."""
```

**Key Methods:**

- `initialize(prices)`: Full calculation to establish baseline state
- `update(new_price)`: O(1) update for new bar
- `save_state(path)`: Persist state to disk
- `load_state(path)`: Restore from saved state

**State Contents:**

The state dictionary contains:
- Moving average values for all 6 types
- Layer 1 signals
- Layer 2 equities
- O(1) MA internal state (buffers, indices)
- Configuration parameters

---

### MultiTimeframeIncrementalATC

Multi-timeframe incremental ATC for hierarchical signal analysis.

```python
class MultiTimeframeIncrementalATC:
    """
    Multi-timeframe incremental ATC that maintains state for multiple timeframes.

    Automatically aggregates lower timeframe bars to higher timeframes and
    maintains synchronized state across all timeframes.

    Example:
        >>> from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import MultiTimeframeIncrementalATC
        >>>
        >>> # Configuration for multiple timeframes
        >>> config = {
        ...     "ema_len": 28,
        ...     "robustness": "Medium",
        ...     "lambda_param": 0.02,
        ...     "decay": 0.03,
        ... }
        >>>
        >>> # Initialize with timeframes: 1m, 5m, 15m
        >>> mtf_atc = MultiTimeframeIncrementalATC(
        ...     config=config,
        ...     base_timeframe="1m",
        ...     higher_timeframes=["5m", "15m"]
        ... )
        >>>
        >>> # Initialize with historical data
        >>> mtf_atc.initialize(historical_1m_prices)
        >>>
        >>> # Update with new 1m bar (automatically aggregates to 5m, 15m)
        >>> signals = mtf_atc.update(new_1m_price)
        >>> print(f"1m signal: {signals['1m']}")
        >>> print(f"5m signal: {signals['5m']}")
        >>> print(f"15m signal: {signals['15m']}")
    """
```

**Use Cases:**

- Multi-timeframe confirmation strategies
- Hierarchical trend analysis
- Reducing false signals with higher timeframe filters

---

## Configuration

### ATCConfig Dataclass

Comprehensive configuration class for ATC analysis.

```python
@dataclass
class ATCConfig:
    """
    Configuration for Adaptive Trend Classification (ATC) analysis.

    This class holds both unscaled and scaled parameter values for ATC calculations.

    Attributes:
        Moving Average Lengths:
            ema_len: EMA length (default: 28)
            hma_len: HMA length (default: 28)
            wma_len: WMA length (default: 28)
            dema_len: DEMA length (default: 28)
            lsma_len: LSMA length (default: 28)
            kama_len: KAMA length (default: 28)

        Moving Average Weights:
            ema_w: EMA initial weight (default: 1.0)
            hma_w: HMA initial weight (default: 1.0)
            wma_w: WMA initial weight (default: 1.0)
            dema_w: DEMA initial weight (default: 1.0)
            lsma_w: LSMA initial weight (default: 1.0)
            kama_w: KAMA initial weight (default: 1.0)

        ATC Parameters:
            robustness: Sensitivity mode - "Narrow", "Medium", or "Wide" (default: "Medium")
            lambda_param: UNSCALED lambda value (default: 0.02)
            decay: UNSCALED decay value (default: 0.03)
            cutout: Number of initial bars to skip (default: 0)
            strategy_mode: If True, shifts signals for non-repainting (default: False)

        Signal Thresholds:
            long_threshold: Threshold for long signal (default: 0.1)
            short_threshold: Threshold for short signal (default: -0.1)
            equity_floor: Minimum equity value (default: 0.25)

        Data Parameters:
            calculation_source: Price source - "close", "open", "high", "low" (default: "close")
            limit: Number of bars to fetch (default: 1500)
            timeframe: Timeframe string (default: "15m")

        Performance Optimization:
            batch_size: Symbols per batch before GC (default: 100)
            precision: Floating point precision - "float64" or "float32" (default: "float64")
            parallel_l1: Enable Layer 1 parallelism (default: True)
            parallel_l2: Enable Layer 2 parallelism (default: True)
            use_rust_backend: Use Rust backend (default: True)

        Approximation Parameters:
            use_approximate: Use approximate calculations (default: False)
            approximate_threshold: Tolerance for approximations (default: 0.05)
            use_adaptive_approximate: Use volatility-adjusted approximations (default: False)
            approximate_volatility_window: Window for volatility (default: 20)
            approximate_volatility_factor: Scaling factor (default: 1.0)

        Advanced Options:
            use_compression: Enable blosc compression for cache (default: False)
            compression_level: Compression level 0-9 (default: 5)
            compression_algorithm: Algorithm name (default: "blosclz")
            use_memory_mapped: Enable memory-mapped arrays (default: False)
            use_codegen_specialization: Enable JIT specialization (default: False)

    Properties:
        lambda_scaled: Returns lambda_param / 1000
        decay_scaled: Returns decay / 100

    Example:
        >>> from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
        >>>
        >>> # Create with defaults
        >>> config = ATCConfig()
        >>>
        >>> # Create with custom parameters
        >>> config = ATCConfig(
        ...     ema_len=21,
        ...     hma_len=21,
        ...     robustness="Wide",
        ...     lambda_param=0.03,
        ...     decay=0.04,
        ...     timeframe="1h",
        ...     limit=2000,
        ... )
        >>>
        >>> # Access scaled values
        >>> print(f"Lambda scaled: {config.lambda_scaled}")  # 0.00003
        >>> print(f"Decay scaled: {config.decay_scaled}")    # 0.0004
    """
```

**Important Parameter Notes:**

1. **lambda_param and decay**: Always provide UNSCALED values
   - lambda_param: Use same value as Pine Script (e.g., 0.02)
   - decay: Use same value as Pine Script (e.g., 0.03)
   - Scaling is applied automatically via properties or by compute_atc_signals

2. **strategy_mode**:
   - False (default): Indicator view - signals repaint, accurate for current bar
   - True: Strategy view - signals shifted by 1 bar, non-repainting for backtesting

3. **robustness modes**:
   - "Narrow": Quick response, more sensitive to noise
   - "Medium": Balanced (recommended default)
   - "Wide": Slower response, more stable signals

---

### create_atc_config_from_dict

Factory function to create ATCConfig from a dictionary.

```python
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

    Example:
        >>> from modules.adaptive_trend_LTS_mini.utils.config import create_atc_config_from_dict
        >>>
        >>> params = {
        ...     "ema_len": 21,
        ...     "hma_len": 21,
        ...     "robustness": "Wide",
        ...     "lambda_param": 0.03,
        ...     "decay": 0.04,
        ...     "limit": 2000,
        ... }
        >>>
        >>> config = create_atc_config_from_dict(params, timeframe="1h")
    """
```

---

## Batch Processing

### process_symbols_batch_dask

Dask-based batch processing for large datasets (out-of-core processing).

```python
def process_symbols_batch_dask(
    symbols: List[str],
    data_fetcher: "DataFetcher",
    config: ATCConfig,
    n_partitions: int = 4,
) -> pd.DataFrame:
    """
    Process multiple symbols using Dask for out-of-core batch processing.

    This function uses Dask to process large batches of symbols that may not
    fit in memory. Results are computed lazily and can be persisted to disk.

    Args:
        symbols: List of symbols to process
        data_fetcher: DataFetcher instance
        config: ATCConfig configuration
        n_partitions: Number of Dask partitions (default: 4)

    Returns:
        DataFrame with results for all symbols

    Example:
        >>> from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import process_symbols_batch_dask
        >>>
        >>> # Process 1000 symbols with Dask
        >>> symbols = [f"SYMBOL{i}/USDT" for i in range(1000)]
        >>> results = process_symbols_batch_dask(
        ...     symbols=symbols,
        ...     data_fetcher=data_fetcher,
        ...     config=config,
        ...     n_partitions=8,
        ... )
    """
```

**When to Use:**

- Processing 100+ symbols in batch
- Memory-constrained environments
- Distributed computing setups
- Results need to be persisted to disk

---

## Examples

### Basic Usage

Compute ATC signals for a single symbol:

```python
import pandas as pd
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
from modules.common.core.data_fetcher import DataFetcher

# Initialize data fetcher
data_fetcher = DataFetcher()

# Fetch price data
df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTC/USDT",
    timeframe="15m",
    limit=1500,
)

# Extract close prices
prices = df["close"]

# Compute ATC signals with default parameters
results = compute_atc_signals(prices)

# Access results
average_signal = results["Average_Signal"]
ema_signal = results["EMA_Signal"]
ema_equity = results["EMA_S"]

# Get current signal
current_signal = average_signal.iloc[-1]

if current_signal > 0.1:
    print("LONG signal")
elif current_signal < -0.1:
    print("SHORT signal")
else:
    print("NEUTRAL")
```

---

### Custom Configuration

Use custom parameters for specific trading strategies:

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

# Create custom configuration
config = ATCConfig(
    # Fast MAs for scalping
    ema_len=14,
    hma_len=14,
    wma_len=14,
    dema_len=14,
    lsma_len=14,
    kama_len=14,

    # Higher weights for momentum MAs
    hma_w=1.5,
    kama_w=1.5,

    # Narrow robustness for quick response
    robustness="Narrow",

    # Tighter thresholds
    long_threshold=0.05,
    short_threshold=-0.05,

    # Strategy mode for backtesting
    strategy_mode=True,

    # Performance optimization
    use_rust_backend=True,
    parallel_l1=True,
    parallel_l2=True,
)

# Compute signals
results = compute_atc_signals(
    prices,
    ema_len=config.ema_len,
    hma_len=config.hma_len,
    wma_len=config.wma_len,
    dema_len=config.dema_len,
    lsma_len=config.lsma_len,
    kama_len=config.kama_len,
    ema_w=config.ema_w,
    hma_w=config.hma_w,
    wma_w=config.wma_w,
    dema_w=config.dema_w,
    lsma_w=config.lsma_w,
    kama_w=config.kama_w,
    robustness=config.robustness,
    lambda_param=config.lambda_param,
    decay_rate=config.decay,
    long_threshold=config.long_threshold,
    short_threshold=config.short_threshold,
    strategy_mode=config.strategy_mode,
    use_rust_backend=config.use_rust_backend,
    parallel_l1=config.parallel_l1,
    parallel_l2=config.parallel_l2,
)
```

---

### Incremental Updates

Use IncrementalATC for live trading with O(1) updates:

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC
import pandas as pd

# Configuration
config = {
    "ema_len": 28,
    "hma_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "robustness": "Medium",
    "lambda_param": 0.02,
    "decay": 0.03,
    "use_o1_mas": True,
    "use_rust_backend": True,
}

# Initialize with historical data
atc = IncrementalATC(config)
historical_prices = pd.Series([100, 101, 102, 103, 104])
initial_results = atc.initialize(historical_prices)

print(f"Initial Average_Signal: {initial_results['Average_Signal'].iloc[-1]}")

# Live trading loop simulation
import time

for i in range(10):
    # Simulate new price
    new_price = 105 + i * 0.5

    # O(1) update
    start = time.time()
    new_signal = atc.update(new_price)
    elapsed = time.time() - start

    print(f"Price: {new_price:.2f}, Signal: {new_signal:.4f}, Time: {elapsed*1000:.2f}ms")

    # Trading logic
    if new_signal > 0.1:
        print("  -> LONG signal")
    elif new_signal < -0.1:
        print("  -> SHORT signal")

# Save state for next session
atc.save_state("atc_state.msgpack")

# Later: restore state
restored_atc = IncrementalATC.load_state("atc_state.msgpack")
new_signal = restored_atc.update(115.5)
print(f"Restored state, new signal: {new_signal:.4f}")
```

---

### Batch Processing Example

Scan multiple symbols efficiently:

```python
from modules.adaptive_trend_LTS_mini.core.analyzer import analyze_symbol
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
from modules.common.core.data_fetcher import DataFetcher
import pandas as pd

# Configuration
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    robustness="Medium",
    use_rust_backend=True,
    parallel_l1=True,
    parallel_l2=True,
)

# Initialize data fetcher
data_fetcher = DataFetcher()

# Symbols to scan
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]

# Scan symbols
results = []

for symbol in symbols:
    result = analyze_symbol(symbol, data_fetcher, config)

    if result:
        signal = result["atc_results"]["Average_Signal"].iloc[-1]
        price = result["current_price"]

        results.append({
            "symbol": symbol,
            "price": price,
            "signal": signal,
            "trend": "LONG" if signal > 0.1 else "SHORT" if signal < -0.1 else "NEUTRAL",
            "exchange": result["exchange_label"],
        })

# Create summary DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# Filter for strong signals
strong_signals = df_results[abs(df_results["signal"]) > 0.1]
print("\nStrong signals:")
print(strong_signals)
```

---

## Performance Benchmarks

### Single Symbol Performance (1000 bars)

| Configuration | Time (ms) |
|--------------|-----------|
| Rust backend, parallel L1+L2 | 100-200 |
| Rust backend, no parallel | 200-400 |
| Numba backend, parallel | 300-600 |
| Numba backend, no parallel | 500-1000 |
| Approximate mode | 50-100 |

### Batch Processing (100 symbols, 1000 bars each)

| Configuration | Time (s) |
|--------------|----------|
| Rust + Parallel L1+L2 | 10-20 |
| Rust + No parallel | 20-40 |
| Dask batch processing | 15-30 |
| Approximate scanning | 5-10 |

### Incremental Update Performance

| Operation | Time |
|-----------|------|
| initialize() | ~100-500ms (full calculation) |
| update() | <1ms (O(1) update) |
| save_state() | ~10-50ms |
| load_state() | ~10-50ms |

---

## Migration from GPU Version

This is the CPU-only mini version. Key differences from the GPU version:

1. **No CUDA/GPU support**: All processing on CPU via Rust/Rayon
2. **Rust backend default**: `use_rust_backend=True` by default
3. **Parallel CPU**: Uses Rayon for multi-core parallelism
4. **Memory optimizations**: More aggressive memory management
5. **Approximate modes**: Added for faster scanning

For detailed migration guide, see `docs/MIGRATION_FROM_GPU.md`.

---

## Troubleshooting

### Common Issues

**Issue: Slow performance**
- Enable `use_rust_backend=True`
- Set `parallel_l1=True` and `parallel_l2=True`
- Try `use_approximate=True` for exploratory scanning
- Check CPU core count (parallel requires 4+ cores)

**Issue: High memory usage**
- Reduce `limit` parameter (e.g., 1000 instead of 2000)
- Use `precision="float32"` instead of "float64"
- Enable `use_compression=True` for caching
- Process symbols in smaller batches

**Issue: "Cannot compute {MA_TYPE}" error**
- Check price series has enough data points (need > ma_len)
- Verify price series is not empty or all NaN
- Check robustness offsets don't exceed data length

**Issue: Rust backend not available**
- Build Rust extensions: `cd rust_extensions && cargo build --release`
- Falls back to Numba automatically (slower but functional)

---

## Further Reading

- **Setting Guides**: `docs/setting_guides.md` - Detailed parameter tuning
- **Speed Optimization**: `docs/setting_guides_speed_optimization.md` - Performance tips
- **Dask Usage**: `docs/guide_dask_usage.md` - Batch processing with Dask
- **Memory Optimizations**: `docs/guide_memory_optimizations_usage.md` - Memory management
- **Profiling**: `docs/guide_profilling.md` - Performance profiling

---

## API Stability

This API is considered **STABLE** for the CPU-only mini version. Breaking changes will be avoided, and deprecation warnings will be provided for any future changes.

**Version History:**
- v1.0 (2026-02-06): Initial API reference for CPU-only mini version
