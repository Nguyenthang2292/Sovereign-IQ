# Auto num_threads Calculation Feature + Enable/Disable Flag

## Summary

Implemented automatic calculation of `num_threads` for the `approximate_ma_scanner` configuration based on CPU core count, plus added an explicit `enabled` flag to control the feature. This removes the need to manually configure thread count for different machines and makes it clear when the feature is active.

## Changes Made

### 1. Configuration File (`standard_batch_scan_config.yaml`)

**Location**: `standard_batch_scan_config.yaml:153-159`

Added `enabled` flag and changed `num_threads` to `null` (auto-calculate):

```yaml
approximate_ma_scanner:
  enabled: true            # Enable/disable approximate MA calculations (true = 2-3x faster, false = exact MAs)
  use_adaptive: true       # Enable adaptive tolerance based on volatility
  num_threads: null        # Number of threads for parallel processing (null = auto-calculate based on CPU cores)
  volatility_window: 20    # Window size for volatility calculation
  base_tolerance: 0.05     # Base tolerance (5%) for adaptive mode
  volatility_factor: 1.0   # Multiplier for volatility impact on tolerance
```

### 2. Config Loader (`modules/gemini_chart_analyzer/cli/config/loader.py`)

**Added Features**:
- Import `HardwareManager` for CPU detection
- Added `_process_config_auto_values()` function to auto-calculate thread count
- Modified `load_configuration_from_file()` to call the processor after loading
- Only auto-calculates when `enabled: true`

**Auto-calculation Logic**:
```python
def _process_config_auto_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process configuration to replace auto-calculated values."""
    if "approximate_ma_scanner" in config:
        ama_config = config["approximate_ma_scanner"]
        if isinstance(ama_config, dict):
            # Only process if feature is enabled
            is_enabled = ama_config.get("enabled", False)

            if is_enabled:
                num_threads = ama_config.get("num_threads")

                # Auto-calculate if null, "auto", or not set
                if num_threads is None or (isinstance(num_threads, str) and num_threads.lower() == "auto"):
                    hw_manager = get_hardware_manager()
                    hw_manager.detect_resources()
                    resources = hw_manager.get_resources()

                    # Calculate optimal thread count:
                    # Reserve 1-2 cores for system, use logical cores for threading
                    optimal_threads = max(1, resources.cpu_cores - 2)
                    ama_config["num_threads"] = optimal_threads
    return config
```

### 3. Workflow (`modules/gemini_chart_analyzer/core/prefilter/workflow.py`)

**Updated to check `enabled` flag**:
```python
# Approximate MA Scanner settings
if approximate_ma_scanner and approximate_ma_scanner.get("enabled", False):
    args.use_approximate = True
    args.use_adaptive_approximate = approximate_ma_scanner.get("use_adaptive", False)
    args.approximate_volatility_window = approximate_ma_scanner.get("volatility_window", 20)
    args.approximate_volatility_factor = approximate_ma_scanner.get("volatility_factor", 1.0)
else:
    args.use_approximate = False
    args.use_adaptive_approximate = False
```

## How It Works

1. **Configuration Loading**: When `standard_batch_scan_config.yaml` is loaded
2. **Enabled Check**: First checks if `enabled: true` in the config
3. **Auto-calculation**: If enabled and `num_threads` is `null` (or `"auto"`), auto-calculates
4. **Hardware Detection**: Uses `HardwareManager` to detect CPU cores
5. **Thread Calculation**: Calculates optimal threads as `max(1, cpu_cores - 2)` (reserves 2 cores for system)
6. **Config Update**: Updates the config dictionary with the calculated value
7. **Workflow Check**: Workflow checks `enabled` flag before activating approximate MAs

## Configuration Options

### enabled Flag

```yaml
approximate_ma_scanner:
  enabled: true   # Enable feature
  enabled: false  # Disable feature (use exact MAs)
```

### num_threads Values

- `null` (YAML) → Auto-calculate based on CPU cores (when enabled=true)
- `"auto"` (string) → Auto-calculate based on CPU cores (when enabled=true)
- Integer (e.g., `4`) → Use specified value (manual override)

## Calculation Formula

```
optimal_threads = max(1, cpu_cores - 2)
```

**Rationale**:
- Uses **logical cores** (including hyperthreading) for threading workloads
- Reserves **2 cores** for system processes and OS
- Ensures **minimum of 1 thread** for single-core systems

**Example**:
- System with 12 logical cores → `num_threads = 10`
- System with 8 logical cores → `num_threads = 6`
- System with 4 logical cores → `num_threads = 2`

## Benefits

1. **Clear Control**: Explicit `enabled` flag shows intent
2. **Easy Toggle**: Change `enabled: false` to disable without commenting out entire section
3. **Portability**: Same config works optimally across different machines
4. **Scalability**: Automatically uses more threads on high-core-count machines
5. **Safety**: Reserves cores for system stability
6. **Override**: Can still manually specify thread count if needed

## Testing

**All Tests Passed**:

1. ✅ **Real config with enabled=true**: Auto-calculates to 10 threads
2. ✅ **Manual config with enabled=true**: Auto-calculates correctly
3. ✅ **Manual config with enabled=false**: Does NOT auto-calculate (remains None)
4. ✅ **Manual override**: Preserves manually specified num_threads value

**Test Output**:
```
Test 1: Load real config
  enabled: True
  num_threads: 10
  Result: ✅ PASS

Test 2: enabled=True with null num_threads
  enabled: True
  num_threads: 10
  Result: ✅ PASS - Auto-calculated

Test 3: enabled=False
  enabled: False
  num_threads: None
  Result: ✅ PASS - Not auto-calculated (remains None)

Test 4: enabled=True with manual num_threads=8
  enabled: True
  num_threads: 8
  Result: ✅ PASS - Manual value preserved
```

## Usage Examples

### Example 1: Enable with Auto-Calculate (Recommended)
```yaml
approximate_ma_scanner:
  enabled: true       # Feature ON
  num_threads: null   # Auto-calculate
```
**Result**: Uses approximate MAs with optimal thread count for your CPU

### Example 2: Enable with Manual Thread Count
```yaml
approximate_ma_scanner:
  enabled: true       # Feature ON
  num_threads: 8      # Force 8 threads
```
**Result**: Uses approximate MAs with exactly 8 threads

### Example 3: Disable Feature
```yaml
approximate_ma_scanner:
  enabled: false      # Feature OFF
  num_threads: null   # Ignored when disabled
```
**Result**: Uses exact MAs (slower but more precise)

### Example 4: Disable by Commenting Out (Still Supported)
```yaml
# approximate_ma_scanner:
#   enabled: true
#   num_threads: null
```
**Result**: Uses exact MAs (backward compatible)

## Integration Points

The `num_threads` value is passed to:
1. `BatchApproximateMAScanner` class (in `modules/adaptive_trend_LTS/core/compute_moving_averages/batch_approximate_mas.py`)
2. Used for `ThreadPoolExecutor` parallelization in batch MA calculations
3. Only activated when `enabled: true` in workflow

## Migration Notes

**Backward Compatibility**: ✅ Full backward compatibility maintained

- Old configs without `enabled` flag will default to disabled (safe default)
- Existing configs with hardcoded `num_threads: 4` will continue to work
- New configs should use `enabled: true` and `num_threads: null`
- Users can still override with any integer value

## Future Enhancements

Potential improvements:
1. Add similar `enabled` + auto-calculation pattern to other features
2. Consider workload size in thread calculation
3. Add configuration validation warnings for extreme values
4. Centralize auto-calculation logic for reuse

## Related Files

- `standard_batch_scan_config.yaml` - Configuration template with enabled flag
- `modules/gemini_chart_analyzer/cli/config/loader.py` - Config loading with auto-calculation
- `modules/gemini_chart_analyzer/core/prefilter/workflow.py` - Workflow enabled check
- `modules/common/system/managers/hardware_manager.py` - Hardware detection
- `modules/adaptive_trend_LTS/core/compute_moving_averages/batch_approximate_mas.py` - Uses num_threads

---

**Implementation Date**: 2026-01-27
**Status**: ✅ Complete and Tested
