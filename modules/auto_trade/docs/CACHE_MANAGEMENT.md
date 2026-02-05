# Python Cache Management Guide

This guide explains how to clear Python bytecode cache to ensure fresh imports after code changes.

## Why Clear Cache?

Python caches compiled bytecode in `__pycache__` directories. After updating:
- Dataclass definitions (like `ATCConfig`)
- Module code
- Rust extensions
- Import paths

You need to clear the cache to ensure the changes are picked up.

## Usage

### 1. Automatic Cache Clearing (Recommended)

The Rust builder now automatically clears cache after building extensions:

```bash
# Build Rust extensions with automatic cache clear (default)
python modules/auto_trade/utils/rust_builder.py

# Skip cache clearing if needed
python modules/auto_trade/utils/rust_builder.py --no-cache-clear
```

### 2. GUI Startup with Cache Clearing

When launching the GUI, you can clear cache first:

```bash
# Clear cache and start GUI
python modules/auto_trade/run_gui.py --clear-cache

# Skip Rust build and just clear cache
python modules/auto_trade/run_gui.py --clear-cache --no-rust-build
```

### 3. Manual Cache Clearing

Use the standalone cache cleaner utility:

```bash
# Clear cache for default modules (ATC, XGBoost, auto_trade, common)
python modules/auto_trade/utils/cache_cleaner.py

# Clear cache for specific modules
python modules/auto_trade/utils/cache_cleaner.py -m adaptive_trend_LTS_mini xgboost_LTS

# Clear ALL project cache
python modules/auto_trade/utils/cache_cleaner.py --all

# Verbose output
python modules/auto_trade/utils/cache_cleaner.py -v
```

### 4. From Python Code

```python
from modules.auto_trade.utils.cache_cleaner import clear_module_cache

# Clear specific modules
clear_module_cache(module_names=["adaptive_trend_LTS_mini", "xgboost_LTS"])

# Clear all auto_trade related modules
clear_module_cache()  # Uses defaults
```

## Common Scenarios

### After Modifying ATCConfig
```bash
# Clear cache for ATC modules
python modules/auto_trade/utils/cache_cleaner.py -m adaptive_trend_LTS_mini
```

### After Building Rust Extensions
```bash
# Build with automatic cache clear (default behavior)
python modules/auto_trade/utils/rust_builder.py
```

### Before Starting GUI After Code Changes
```bash
# Clear cache and launch GUI
python modules/auto_trade/run_gui.py --clear-cache
```

### After Updating Multiple Modules
```bash
# Clear entire project cache
python modules/auto_trade/utils/cache_cleaner.py --all
```

## Troubleshooting

### AttributeError After Code Changes

**Symptom**: Getting `AttributeError` for newly added attributes

**Solution**:
```bash
# Clear cache and restart
python modules/auto_trade/utils/cache_cleaner.py
# Then restart your Python process/GUI
```

### Import Errors After Rust Build

**Symptom**: Rust extensions not being recognized after build

**Solution**:
```bash
# Rebuild with cache clear
python modules/auto_trade/utils/rust_builder.py
# Or manually clear and restart
python modules/auto_trade/utils/cache_cleaner.py --all
```

### Old Code Still Running

**Symptom**: Changes not taking effect despite modifying files

**Solution**:
1. Clear cache: `python modules/auto_trade/utils/cache_cleaner.py --all`
2. **Restart your Python process** (cache clearing doesn't affect running processes)

## Important Notes

1. **Cache clearing doesn't affect running processes** - you must restart Python after clearing
2. **Automatic clearing** happens after successful Rust builds by default
3. **Safe operation** - cache directories are automatically regenerated on next import
4. **Permission errors** - if you see permission errors, close any processes using the modules first

## Best Practices

1. **After Rust builds**: Cache is automatically cleared, just restart your app
2. **After code changes**: Use `--clear-cache` flag when starting GUI
3. **During development**: Clear cache when you encounter unexpected behavior
4. **Before deployment**: Clear all cache to ensure clean state

## Command Quick Reference

```bash
# Build Rust + clear cache (automatic)
python modules/auto_trade/utils/rust_builder.py

# GUI with cache clear
python modules/auto_trade/run_gui.py --clear-cache

# Manual clear (default modules)
python modules/auto_trade/utils/cache_cleaner.py

# Manual clear (all project)
python modules/auto_trade/utils/cache_cleaner.py --all

# Manual clear (specific)
python modules/auto_trade/utils/cache_cleaner.py -m module1 module2
```
