use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::sync::{Arc, OnceLock};

// Include gpu_common.h content for NVRTC compilation
const GPU_COMMON_H: &str = include_str!("../../core/gpu_backend/gpu_common.h");

// Helper function to inline gpu_common.h into CUDA source
// For NVRTC, we need to strip out the sections guarded by #ifndef __NVRTC__
fn inline_gpu_common(source: &str) -> String {
    // Strip out the #ifndef __NVRTC__ sections from gpu_common.h
    let nvrtc_safe_header = strip_nvrtc_guards(GPU_COMMON_H);
    source.replace("#include \"gpu_common.h\"", &nvrtc_safe_header)
}

// Remove content between #ifndef __NVRTC__ and #endif
fn strip_nvrtc_guards(content: &str) -> String {
    let mut result = String::new();
    let mut skip_depth = 0;
    let mut in_nvrtc_guard = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Check if this is an #ifndef __NVRTC__ line
        if trimmed.starts_with("#ifndef") && trimmed.contains("__NVRTC__") {
            in_nvrtc_guard = true;
            skip_depth += 1;
            continue;
        }

        // Check for #endif when we're in an NVRTC guard
        if in_nvrtc_guard && trimmed.starts_with("#endif") {
            skip_depth -= 1;
            if skip_depth == 0 {
                in_nvrtc_guard = false;
            }
            continue;
        }

        // If we're not skipping, add the line
        if skip_depth == 0 {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

/// CUDA kernel source embedded at compile time
const MA_KERNELS_SRC: &str = include_str!("../../core/gpu_backend/ma_kernels.cu");

/// Global cache for compiled CUDA module (compiled once, reused forever)
struct CudaCache {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

static CUDA_CACHE: OnceLock<Result<CudaCache, String>> = OnceLock::new();

fn get_cuda_cache() -> Result<&'static CudaCache, PyErr> {
    let cache = CUDA_CACHE.get_or_init(|| {
        // Initialize CUDA context
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {:?}", e))?;

        // Inline gpu_common.h into source before compilation (NVRTC needs this)
        let source = inline_gpu_common(MA_KERNELS_SRC);
        // Compile PTX once
        let ptx =
            compile_ptx(&source).map_err(|e| format!("PTX compile failed: {:?}", e))?;

        // Load module once
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module load failed: {:?}", e))?;

        Ok(CudaCache { ctx, module })
    });

    cache
        .as_ref()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.clone()))
}

/// Calculate EMA on GPU using cached CUDA kernel.
#[pyfunction]
pub fn calculate_ema_cuda<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = prices.as_array().len();
    let cache = get_cuda_cache()?;
    let stream = cache.ctx.default_stream();

    let f = cache.module.load_function("ema_kernel").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load kernel: {:?}", e))
    })?;

    let prices_data = prices.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("prices not contiguous: {:?}", e))
    })?;
    let prices_dev = stream.clone_htod(prices_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("H2D Transfer Error: {:?}", e))
    })?;
    let mut ema_dev = stream.alloc_zeros::<f64>(n).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Device Allocation Error: {:?}",
            e
        ))
    })?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&prices_dev)
            .arg(&mut ema_dev)
            .arg(&(length as i32))
            .arg(&(n as i32))
            .launch(cfg)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Kernel Launch Error: {:?}",
                    e
                ))
            })?;
    }

    let mut ema_host = vec![0.0; n];
    stream.memcpy_dtoh(&ema_dev, &mut ema_host).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("D2H Transfer Error: {:?}", e))
    })?;
    Ok(PyArray1::from_vec(_py, ema_host))
}

/// Calculate KAMA on GPU using dual-pass CUDA kernels.
#[pyfunction]
pub fn calculate_kama_cuda<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = prices.as_array().len();
    let cache = get_cuda_cache()?;
    let stream = cache.ctx.default_stream();

    let noise_kernel = cache
        .module
        .load_function("kama_noise_kernel")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load noise kernel: {:?}",
                e
            ))
        })?;
    let smooth_kernel = cache
        .module
        .load_function("kama_smooth_kernel")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load smooth kernel: {:?}",
                e
            ))
        })?;

    let prices_data = prices.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("prices not contiguous: {:?}", e))
    })?;
    let prices_dev = stream.clone_htod(prices_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("H2D Transfer Error: {:?}", e))
    })?;
    let mut noise_dev = stream.alloc_zeros::<f64>(n).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Device Allocation Error: {:?}",
            e
        ))
    })?;
    let mut kama_dev = stream.alloc_zeros::<f64>(n).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Device Allocation Error: {:?}",
            e
        ))
    })?;

    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    let cfg_parallel = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&noise_kernel)
            .arg(&prices_dev)
            .arg(&mut noise_dev)
            .arg(&(length as i32))
            .arg(&(n as i32))
            .launch(cfg_parallel)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Noise Kernel Launch Error: {:?}",
                    e
                ))
            })?;
    }

    let cfg_seq = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&smooth_kernel)
            .arg(&prices_dev)
            .arg(&noise_dev)
            .arg(&mut kama_dev)
            .arg(&(length as i32))
            .arg(&(n as i32))
            .launch(cfg_seq)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Smooth Kernel Launch Error: {:?}",
                    e
                ))
            })?;
    }

    let mut kama_host = vec![0.0; n];
    stream.memcpy_dtoh(&kama_dev, &mut kama_host).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("D2H Transfer Error: {:?}", e))
    })?;
    Ok(PyArray1::from_vec(_py, kama_host))
}

/// Internal helper to launch WMA kernel on device data
fn launch_wma_kernel(
    cache: &CudaCache,
    prices_dev: &cudarc::driver::CudaSlice<f64>,
    length: usize,
) -> Result<cudarc::driver::CudaSlice<f64>, String> {
    let n = prices_dev.len();
    let stream = cache.ctx.default_stream();

    let f = cache
        .module
        .load_function("wma_kernel")
        .map_err(|e| format!("{:?}", e))?;
    let mut wma_dev = stream
        .alloc_zeros::<f64>(n)
        .map_err(|e| format!("{:?}", e))?;

    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&f)
            .arg(prices_dev)
            .arg(&mut wma_dev)
            .arg(&(length as i32))
            .arg(&(n as i32))
            .launch(cfg)
            .map_err(|e| format!("{:?}", e))?;
    }

    Ok(wma_dev)
}

/// Calculate WMA on GPU using convolution-based CUDA kernel.
#[pyfunction]
pub fn calculate_wma_cuda<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = prices.as_array().len();
    let cache = get_cuda_cache()?;
    let stream = cache.ctx.default_stream();

    let prices_data = prices.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("prices not contiguous: {:?}", e))
    })?;
    let prices_dev = stream.clone_htod(prices_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("H2D Transfer Error: {:?}", e))
    })?;

    // Use the helper
    let wma_dev = launch_wma_kernel(cache, &prices_dev, length)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    let mut wma_host = vec![0.0; n];
    stream.memcpy_dtoh(&wma_dev, &mut wma_host).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("D2H Transfer Error: {:?}", e))
    })?;
    Ok(PyArray1::from_vec(_py, wma_host))
}

/// Calculate HMA on GPU by orchestrating WMA calls entirely on the device.
/// HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
#[pyfunction]
pub fn calculate_hma_cuda<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = prices.as_array().len();
    if n < length {
        return Ok(PyArray1::from_vec(_py, vec![f64::NAN; n]));
    }

    let cache = get_cuda_cache()?;
    let stream = cache.ctx.default_stream();

    // 1. Upload prices once
    let prices_data = prices.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("prices not contiguous: {:?}", e))
    })?;
    let prices_dev = stream.clone_htod(prices_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("H2D Transfer Error: {:?}", e))
    })?;

    // 2. Compute WMA(half) and WMA(full) on device
    let half_len = std::cmp::max(length / 2, 1);
    let sqrt_len = std::cmp::max((length as f64).sqrt() as usize, 1);

    let wma_half_dev = launch_wma_kernel(cache, &prices_dev, half_len)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    let wma_full_dev = launch_wma_kernel(cache, &prices_dev, length)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    // 3. Compute Diff on device: diff = 2 * half - full
    let diff_kernel = cache.module.load_function("hma_diff_kernel").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load diff kernel: {:?}",
            e
        ))
    })?;
    let mut diff_dev = stream.alloc_zeros::<f64>(n).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Device Alloc Error: {:?}", e))
    })?;

    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&diff_kernel)
            .arg(&wma_half_dev)
            .arg(&wma_full_dev)
            .arg(&mut diff_dev)
            .arg(&(n as i32))
            .launch(cfg)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Diff Kernel Error: {:?}",
                    e
                ))
            })?;
    }

    // 4. Compute Final WMA(sqrt) on device
    let final_hma_dev = launch_wma_kernel(cache, &diff_dev, sqrt_len)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    // 5. Download result
    let mut hma_host = vec![0.0; n];
    stream
        .memcpy_dtoh(&final_hma_dev, &mut hma_host)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "D2H Transfer Error: {:?}",
                e
            ))
        })?;

    Ok(PyArray1::from_vec(_py, hma_host))
}
