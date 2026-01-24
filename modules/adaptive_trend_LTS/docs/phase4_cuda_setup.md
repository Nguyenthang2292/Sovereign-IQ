# Phase 4: CUDA Development Environment Setup

This document covers **Task 2.1.1** (Setup CUDA Development Environment) for Phase 4 GPU optimizations.

## 1. Install CUDA Toolkit

1. Download **CUDA Toolkit 12.x** from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads).
2. Run the installer. On Windows, use the exe; on Linux, follow distro-specific steps.
3. **Verify**:
   ```bash
   nvcc --version
   ```
4. **Environment variables** (set by installer on many systems; adjust if needed):
   - **Windows**: `CUDA_PATH` = e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
   - **Linux**: `CUDA_HOME` or `LD_LIBRARY_PATH` include CUDA lib; `PATH` includes `$CUDA_HOME/bin`

## 2. Install Development Tools

### PyCUDA

```bash
pip install pycuda
```

### CuPy

Match your CUDA version:

- CUDA 12.x: `pip install cupy-cuda12x`
- CUDA 11.x: `pip install cupy-cuda11x`

Verify:

```bash
python -c "import cupy; print(cupy.__version__)"
```

### Optional: CUDA samples

Use for reference only. Install from [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) or via Toolkit installer.

## 3. Verify GPU Compatibility

1. **Check GPU and driver**:
   ```bash
   nvidia-smi
   ```
2. **Compute capability**: Phase 4 recommends **>= 6.0** (e.g. GTX 1060+). Check `nvidia-smi` or vendor specs.
3. **Test compilation**: Run the project verification script from **project root**:
   ```bash
   python scripts/verify_cuda_env.py
   ```

## 4. Rust CUDA Setup (New)

Task 2.1.2 uses **Rust** for CUDA orchestration via the `cudarc` crate.

### Installation

No extra pip install needed besides `maturin`. The dependencies are handled by `Cargo.toml`.

### Windows Linker Fix (Critical)

On Windows, the Rust linker may fail to find `cuda.lib` even if `CUDA_PATH` is set. Use `RUSTFLAGS` to point to the library directory:

```powershell
# In PowerShell before building
$env:RUSTFLAGS="-L 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64'"

# Then build
cd modules/adaptive_trend_LTS/rust_extensions
maturin develop --release
```

## 5. Quick Verification Checklist

| Check           | Command / Action                                         |
| --------------- | -------------------------------------------------------- |
| NVIDIA driver   | `nvidia-smi`                                             |
| CUDA Toolkit    | `nvcc --version`                                         |
| CuPy            | `python -c "import cupy; print(cupy.__version__)"`       |
| PyCUDA          | `python -c "import pycuda.driver; pycuda.driver.init()"` |
| **Rust CUDA**   | `cargo check` in `rust_extensions/`                      |
| Full env verify | `python scripts/verify_cuda_env.py`                      |

## 6. Troubleshooting

- **nvcc not found**: Add `CUDA_PATH/bin` (Windows) or `$CUDA_HOME/bin` (Linux) to `PATH`.
- **Rust cudarc build fail**: Ensure `RUSTFLAGS` is set correctly for Windows as shown above.
- **PyCUDA build fails**: Ensure CUDA Toolkit and `nvcc` are installed and in `PATH`.
- **CuPy import error**: Install the matching `cupy-cuda12x` variant.

## 7. References

- [cudarc Docs](https://docs.rs/cudarc/latest/cudarc/)
- [PyCUDA](https://documen.tician.de/pycuda/)
- [CuPy Install Guide](https://docs.cupy.dev/en/stable/install.html)
