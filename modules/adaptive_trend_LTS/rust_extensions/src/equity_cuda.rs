use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::sync::{Arc, OnceLock};

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
        let ptx =
            compile_ptx(EQUITY_KERNEL_SRC).map_err(|e| format!("PTX compile failed: {:?}", e))?;
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
