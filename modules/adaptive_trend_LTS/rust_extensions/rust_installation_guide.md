# Rust Installation Guide for Adaptive Trend Enhance V2

## Issue: Rust not recognized in venv

If you encounter the error `rustc is not installed or not in PATH` even though Rust is installed, here's how to fix it.

## Quick Solution

### Method 1: Use the updated script (Recommended)

The `build_rust.bat` script has been updated to automatically add Rust to PATH:

```powershell
.\build_rust.bat
```

### Method 2: Add Rust to PATH manually

**In PowerShell (for current session):**

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
rustc --version  # Verify
.\build_rust.bat  # Build
```

**To make it permanent:**

1. Open **System Properties** → **Environment Variables**
2. Find the `Path` variable in **User variables**
3. Add: `%USERPROFILE%\.cargo\bin`
4. Restart terminal

### Method 3: Use the check script

```powershell
.\check_rust.ps1
```

This script will:

- Check if Rust is installed
- Automatically add to PATH if needed
- Check and install Maturin if missing

## Installing Rust (if not already installed)

### Step 1: Download and install Rust

1. Visit: <https://rustup.rs/>
2. Download `rustup-init.exe` (or <https://win.rustup.rs/x86_64>)
3. Run the installer and select option `1` (default)
4. Wait for installation to complete

### Step 2: Restart terminal

**Important:** Close and reopen PowerShell/CMD so PATH is updated.

### Step 3: Verify installation

```powershell
rustc --version
cargo --version
```

Expected output:

```code
rustc 1.93.0 (or newer version)
cargo 1.93.0 (or newer version)
```

## Installing Maturin

Maturin is the build tool for Rust extensions in Python:

```powershell
pip install maturin
```

## Building Rust Extensions

After Rust is in PATH:

```powershell
.\build_rust.bat
```

Or manually:

```powershell
cd modules\adaptive_trend_LTS\rust_extensions
maturin develop --release
```

## Verifying successful installation

### Check in Python

```python
try:
    from atc_rust import (
        calculate_equity_rust,
        calculate_kama_rust,
        calculate_ema_rust,
    )
    print("✅ Rust extensions installed successfully!")
except ImportError as e:
    print(f"❌ Rust extensions not installed: {e}")
```

### Run tests

```powershell
cd modules\adaptive_trend_LTS\rust_extensions
cargo test
```

## Troubleshooting

### Error: "rustc is not recognized"

**Cause:** Rust is not in the current terminal's PATH.

**Solution:**

1. Restart terminal (after installing Rust)
2. Or run: `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`
3. Or use the updated `build_rust.bat` (automatically adds PATH)

### Error: "linker not found" or "link.exe not found"

**Cause:** Missing Visual Studio Build Tools.

**Solution:**

1. Install **Visual Studio Build Tools** or **Visual Studio** with C++ workload
2. Or install **Windows SDK**

### Error: "Python version mismatch"

**Cause:** Maturin built in wrong Python environment.

**Solution:**

1. Activate virtual environment before building:

   ```powershell
   .\venv\Scripts\Activate.ps1
   .\build_rust.bat
   ```

### Slow first build

**Normal:** First compilation may take 5-10 minutes. Subsequent builds will be faster thanks to cache.

## Performance

Rust backend provides **2-3x speedup** compared to Numba JIT:

- **Equity Calculation**: ~31µs (10,000 bars)
- **KAMA Calculation**: ~148µs (10,000 bars)  
- **Signal Persistence**: ~5µs (10,000 bars)

## Rust Extensions Features

The Rust module provides optimized functions:

- `calculate_equity_rust`: Calculate equity curves
- `calculate_kama_rust`: Kaufman Adaptive Moving Average
- `calculate_ema_rust`: Exponential Moving Average
- `calculate_wma_rust`: Weighted Moving Average
- `calculate_dema_rust`: Double Exponential Moving Average
- `calculate_lsma_rust`: Least Squares Moving Average
- `calculate_hma_rust`: Hull Moving Average
- `process_signal_persistence_rust`: Process signal persistence

## Usage in Python

The module automatically uses Rust backend if available:

```python
from modules.adaptive_trend_LTS.core.rust_backend import (
    calculate_equity,
    calculate_kama,
    calculate_ema,
)

# Rust will be used automatically if installed
equity = calculate_equity(r_values, sig_prev, starting_equity, decay_multiplier, cutout)
```

## References

- Rust Installation: <https://rustup.rs/>
- Maturin Documentation: <https://www.maturin.rs/>
- PyO3 Documentation: <https://pyo3.rs/>
