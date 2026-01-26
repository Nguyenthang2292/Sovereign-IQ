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

/// CUDA kernel source embedded at compile time for performance.
const EQUITY_KERNEL_SRC: &str = include_str!("../../core/gpu_backend/equity_kernel.cu");

/// Global cache for compiled CUDA module
struct EquityCudaCache {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

static EQUITY_CUDA_CACHE: OnceLock<Result<EquityCudaCache, String>> = OnceLock::new();

fn get_equity_cache() -> Result<&'static EquityCudaCache, PyErr> {
    let cache = EQUITY_CUDA_CACHE.get_or_init(|| {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {:?}", e))?;
        // Inline gpu_common.h into source before compilation (NVRTC needs this)
        let source = inline_gpu_common(EQUITY_KERNEL_SRC);
        let ptx =
            compile_ptx(&source).map_err(|e| format!("PTX compile failed: {:?}", e))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module load failed: {:?}", e))?;
        Ok(EquityCudaCache { ctx, module })
    });

    cache
        .as_ref()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.clone()))
}

/// Calculate equity values on GPU using a cached CUDA kernel.
#[pyfunction]
pub fn calculate_equity_cuda<'py>(
    _py: Python<'py>,
    r_values: PyReadonlyArray1<'py, f64>,
    sig_prev_values: PyReadonlyArray1<'py, f64>,
    starting_equity: f64,
    decay_multiplier: f64,
    cutout: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = r_values.as_array().len();
    if n != sig_prev_values.as_array().len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "r_values and sig_prev_values must have same length",
        ));
    }

    let cache = get_equity_cache()?;
    let stream = cache.ctx.default_stream();

    let f = cache.module.load_function("equity_kernel").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load kernel: {:?}", e))
    })?;

    let r_data = r_values.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("r_values not contiguous: {:?}", e))
    })?;
    let sig_data = sig_prev_values.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "sig_prev_values not contiguous: {:?}",
            e
        ))
    })?;

    let r_dev = stream.clone_htod(r_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("H2D r_values failed: {:?}", e))
    })?;
    let sig_dev = stream.clone_htod(sig_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "H2D sig_prev_values failed: {:?}",
            e
        ))
    })?;
    let mut eq_dev = stream.alloc_zeros::<f64>(n).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Device alloc failed: {:?}", e))
    })?;

    // For single series, offsets = [0]
    let offsets = vec![0i32];
    let offsets_dev = stream.clone_htod(&offsets).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("H2D offsets failed: {:?}", e))
    })?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&f)
            .arg(&r_dev)
            .arg(&sig_dev)
            .arg(&mut eq_dev)
            .arg(&starting_equity)
            .arg(&decay_multiplier)
            .arg(&(cutout as i32))
            .arg(&(n as i32))
            .arg(&offsets_dev)
            .arg(&1i32)  // num_symbols = 1 for single series
            .launch(cfg)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Kernel launch failed: {:?}",
                    e
                ))
            })?;
    }

    let mut eq_host = vec![0.0; n];
    stream.memcpy_dtoh(&eq_dev, &mut eq_host).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("D2H copy failed: {:?}", e))
    })?;

    Ok(PyArray1::from_vec(_py, eq_host))
}
