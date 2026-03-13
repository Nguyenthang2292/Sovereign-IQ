# JIT Specialization Usage Guide

## Overview

Phase 8.2 implements JIT specialization for ATC computations to improve performance for frequently used configurations. The primary specialization implemented is **EMA-only** mode, which provides 10-20% performance improvement for repeated calls after JIT warm-up.

## Quick Start

### Basic Usage

```python
import pandas as pd
from modules.adaptive_trend_LTS.core.codegen.specialization import (
    compute_atc_specialized,
)
from modules.adaptive_trend_LTS.utils.config import ATCConfig

# Create config with specialization enabled
config = ATCConfig(
    ema_len=28,
    robustness="Medium",
    use_codegen_specialization=True,  # Enable JIT specialization
)

# Compute with specialized path (EMA-only)
result = compute_atc_specialized(
    prices,
    config,
    mode="ema_only",
    use_codegen_specialization=True,
    fallback_to_generic=True,  # Safe fallback if specialization fails
)

# Result contains EMA_Signal, EMA_S, Average_Signal
print(result["EMA_Signal"])
print(result["EMA_S"])
```

### Enable/Disable Specialization

```python
# Enable via config
config = ATCConfig(
    ema_len=28,
    robustness="Medium",
    use_codegen_specialization=True,  # Enable
)

# Or control per-call
result = compute_atc_specialized(
    prices,
    config,
    mode="ema_only",
    use_codegen_specialization=True,  # Enable for this call
    fallback_to_generic=True,  # Fallback if fails
)

# Disable specialization (use generic path)
result = compute_atc_specialized(
    prices,
    config,
    mode="ema_only",
    use_codegen_specialization=False,  # Disable
    fallback_to_generic=True,
)
```

## API Reference

### compute_atc_specialized()

Main entrypoint for specialized ATC computation.

```python
def compute_atc_specialized(
    prices: pd.Series,
    config: ATCConfig,
    mode: str = "default",
    use_codegen_specialization: bool = True,
    fallback_to_generic: bool = True,
    **kwargs: Any,
) -> dict[str, pd.Series]
```

**Parameters**:
- `prices`: Price series
- `config`: ATC configuration
- `mode`: Specialization mode ("ema_only", "default", etc.)
- `use_codegen_specialization`: Enable JIT specialization
- `fallback_to_generic`: Fall back to generic path if specialization fails
- `**kwargs`: Additional parameters for generic path (if fallback used)

**Returns**:
- Dictionary with ATC signals and equities

**Example**:
```python
result = compute_atc_specialized(
    prices,
    config,
    mode="ema_only",
    use_codegen_specialization=True,
    fallback_to_generic=True,
)

# Access results
ema_signal = result["EMA_Signal"]
ema_equity = result["EMA_S"]
avg_signal = result["Average_Signal"]
```

### get_specialized_compute_fn()

Get a specialized compute function for a configuration.

```python
from modules.adaptive_trend_LTS.core.codegen.specialization import (
    get_specialized_compute_fn,
)

# Get specialized function
compute_fn = get_specialized_compute_fn(
    config,
    mode="ema_only",
    use_codegen=True,
)

# Use the function (if available)
if compute_fn:
    result = compute_fn(prices)
```

### is_config_specializable()

Check if a configuration can be specialized.

```python
from modules.adaptive_trend_LTS.core.codegen.specialization import (
    is_config_specializable,
)

# Check if config is specializable
if is_config_specializable(config, mode="ema_only"):
    print("Config can be specialized!")
else:
    print("Using generic path")
```

## Specialization Modes

### EMA-Only (Production-Ready)

**Mode**: `ema_only`

**Scope**: Single MA (EMA) with any length

**Use Cases**:
- Fast scanning and filtering
- Real-time single MA tracking
- Pre-screening before full ATC

**Supported**: ✅ Production-ready, all lengths

```python
result = compute_atc_specialized(
    prices,
    config,
    mode="ema_only",
    use_codegen_specialization=True,
)
```

### Default (Not Prioritized)

**Mode**: `default`

**Scope**: All 6 MAs with standard config (length 28, Medium robustness)

**Use Cases**:
- Full ATC computation
- Backtesting with standard settings

**Supported**: ❌ Not implemented (use generic path instead)

```python
# Use generic path for full ATC
from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import (
    compute_atc_signals,
)

result = compute_atc_signals(prices, ema_len=28, robustness="Medium")
```

## Configuration

### ATCConfig Flag

```python
from modules.adaptive_trend_LTS.utils.config import ATCConfig

# Enable specialization globally
config = ATCConfig(
    ema_len=28,
    robustness="Medium",
    use_codegen_specialization=True,  # Enable
)

# Disable specialization globally
config = ATCConfig(
    ema_len=28,
    robustness="Medium",
    use_codegen_specialization=False,  # Disable
)
```

## Fallback Behavior

When specialization fails or is unavailable, the system falls back to the generic path:

```python
result = compute_atc_specialized(
    prices,
    config,
    mode="ema_only",
    use_codegen_specialization=True,
    fallback_to_generic=True,  # Safe fallback
)

# If specialization fails:
# - Logs warning
# - Falls back to compute_atc_signals()
# - Returns same result format
```

### Strict Mode (No Fallback)

For strict mode, set `fallback_to_generic=False`:

```python
try:
    result = compute_atc_specialized(
        prices,
        config,
        mode="ema_only",
        use_codegen_specialization=True,
        fallback_to_generic=False,  # Strict mode
    )
except ValueError as e:
    # Specialization failed
    print(f"Error: {e}")
```

## Performance

### Expected Gains

- **EMA-only**: 10-20% improvement on repeated calls (after JIT warm-up)
- **Generic path**: No improvement (already optimized with Rust + CUDA)

### Benchmarking

Run benchmarks to measure performance:

```bash
python -m modules.adaptive_trend_LTS.benchmarks.benchmark_specialization
```

Output:
```
Benchmark: Default Length (28)
Data size: 1000, Runs: 100, Warmup: 10
Generic path: 15.234 ms (±2.123)
Specialized path: 12.876 ms (±1.456)
Speedup: 1.18x (15.5% improvement)
✅ Meets target: >=10% improvement (15.5%)
```

## Testing

Run tests:

```bash
pytest modules/adaptive_trend_LTS/tests/test_specialization.py -v
```

Tests verify:
- Configs are correctly identified as specializable
- Specialized path produces same results as generic path
- Fallback mechanism works correctly
- Flag controls behavior as expected

## Best Practices

1. **Use EMA-only for scanning**: Fast single MA for initial filtering
2. **Enable fallback**: Always use `fallback_to_generic=True` in production
3. **Warm up JIT**: First call is slower (JIT compilation), subsequent calls are fast
4. **Benchmark first**: Measure actual performance gains for your use case
5. **Use generic for full ATC**: For all 6 MAs, use `compute_atc_signals()`

## Troubleshooting

### Specialization not working

**Issue**: Specialization not being used

**Solution**:
1. Check Numba is installed: `pip install numba`
2. Verify config has `use_codegen_specialization=True`
3. Check mode is supported (`ema_only` only)
4. Look for warnings in logs

### Slower than generic

**Issue**: First call is slow

**Reason**: JIT compilation overhead

**Solution**: Warm up with a few calls before measuring performance

### Numba errors

**Issue**: Numba compilation fails

**Solution**:
- Enable fallback: `fallback_to_generic=True`
- Check Numba version: `pip install --upgrade numba`
- Report issue if persists

## Documentation

- `core/codegen/specialization.py`: API documentation
- `core/codegen/numba_specialized.py`: JIT implementations
- `benchmarks/benchmark_specialization.py`: Performance benchmarks
- `docs/phase8_2_scope_decisions.md`: Strategic decisions and scope
- `docs/phase8_2_hot_path_configs.md`: Hot path configurations
- `tests/test_specialization.py`: Test coverage

## Next Steps

For developers:
- Extend to other single MA types (KAMA-only, etc.)
- Consider short-length multi-MA if business need arises
- Continue optimizing generic paths (Rust, CUDA, Dask)

For users:
- Use EMA-only for scanning and filtering
- Use generic path for full ATC with all MAs
- Benchmark to validate performance gains for your use case
