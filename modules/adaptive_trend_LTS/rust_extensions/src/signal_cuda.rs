use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::sync::{Arc, OnceLock};

/// CUDA kernel source embedded at compile time
const SIGNAL_KERNELS_SRC: &str = include_str!("../../core/gpu_backend/signal_kernels.cu");

/// Global cache for compiled CUDA module
struct SignalCudaCache {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
}

static SIGNAL_CUDA_CACHE: OnceLock<Result<SignalCudaCache, String>> = OnceLock::new();

fn get_signal_cache() -> Result<&'static SignalCudaCache, PyErr> {
    let cache = SIGNAL_CUDA_CACHE.get_or_init(|| {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {:?}", e))?;
        let ptx =
            compile_ptx(SIGNAL_KERNELS_SRC).map_err(|e| format!("PTX compile failed: {:?}", e))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module load failed: {:?}", e))?;
        Ok(SignalCudaCache { ctx, module })
    });

    cache
        .as_ref()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.clone()))
}

/// Calculate weighted average signal on GPU.
#[pyfunction]
pub fn calculate_average_signal_cuda<'py>(
    _py: Python<'py>,
    signals: PyReadonlyArray2<'py, f64>,
    equities: PyReadonlyArray2<'py, f64>,
    long_threshold: f64,
    short_threshold: f64,
    cutout: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signals_shape = signals.shape();
    let equities_shape = equities.shape();
    if signals_shape != equities_shape {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "signals and equities must have same shape",
        ));
    }
    let n_mas = signals_shape[0];
    let n_bars = signals_shape[1];

    let cache = get_signal_cache()?;
    let stream = cache.ctx.default_stream();

    let f = cache
        .module
        .load_function("weighted_average_signal_kernel")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load kernel: {:?}",
                e
            ))
        })?;

    let signals_data = signals.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signals not contiguous: {:?}", e))
    })?;
    let equities_data = equities.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("equities not contiguous: {:?}", e))
    })?;

    let signals_dev = stream.clone_htod(signals_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "H2D Transfer Error (signals): {:?}",
            e
        ))
    })?;
    let equities_dev = stream.clone_htod(equities_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "H2D Transfer Error (equities): {:?}",
            e
        ))
    })?;
    let mut avg_signal_dev = stream.alloc_zeros::<f64>(n_bars).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Device Allocation Error: {:?}",
            e
        ))
    })?;

    let block_size = 256u32;
    let grid_size = ((n_bars as u32) + block_size - 1) / block_size;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&signals_dev)
            .arg(&equities_dev)
            .arg(&mut avg_signal_dev)
            .arg(&(n_mas as i32))
            .arg(&(n_bars as i32))
            .arg(&cutout)
            .arg(&long_threshold)
            .arg(&short_threshold)
            .launch(cfg)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Kernel Launch Error: {:?}",
                    e
                ))
            })?;
    }

    let mut avg_signal_host = vec![0.0; n_bars];
    stream
        .memcpy_dtoh(&avg_signal_dev, &mut avg_signal_host)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "D2H Transfer Error: {:?}",
                e
            ))
        })?;
    Ok(PyArray1::from_vec(_py, avg_signal_host))
}

/// Classify trend based on signal thresholds on GPU.
#[pyfunction]
pub fn classify_trend_cuda<'py>(
    _py: Python<'py>,
    signals: PyReadonlyArray1<'py, f64>,
    long_threshold: f64,
    short_threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let n = signals.as_array().len();

    let cache = get_signal_cache()?;
    let stream = cache.ctx.default_stream();

    let f = cache
        .module
        .load_function("classify_trend_kernel")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load kernel: {:?}",
                e
            ))
        })?;

    let signals_data = signals.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signals not contiguous: {:?}", e))
    })?;
    let signals_dev = stream.clone_htod(signals_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("H2D Transfer Error: {:?}", e))
    })?;
    let mut trends_dev = stream.alloc_zeros::<i32>(n).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Device Allocation Error: {:?}",
            e
        ))
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
            .launch_builder(&f)
            .arg(&signals_dev)
            .arg(&mut trends_dev)
            .arg(&(n as i32))
            .arg(&long_threshold)
            .arg(&short_threshold)
            .launch(cfg)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Kernel Launch Error: {:?}",
                    e
                ))
            })?;
    }

    let mut trends_host = vec![0i32; n];
    stream
        .memcpy_dtoh(&trends_dev, &mut trends_host)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "D2H Transfer Error: {:?}",
                e
            ))
        })?;
    Ok(PyArray1::from_vec(_py, trends_host))
}

/// Fused weighted average + classification on GPU.
#[pyfunction]
pub fn calculate_and_classify_cuda<'py>(
    _py: Python<'py>,
    signals: PyReadonlyArray2<'py, f64>,
    equities: PyReadonlyArray2<'py, f64>,
    long_threshold: f64,
    short_threshold: f64,
    cutout: i32,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>)> {
    let signals_shape = signals.shape();
    let equities_shape = equities.shape();
    if signals_shape != equities_shape {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "signals and equities must have same shape",
        ));
    }
    let n_mas = signals_shape[0];
    let n_bars = signals_shape[1];

    let cache = get_signal_cache()?;
    let stream = cache.ctx.default_stream();

    let f = cache
        .module
        .load_function("weighted_average_and_classify_kernel")
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load kernel: {:?}",
                e
            ))
        })?;

    let signals_data = signals.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signals not contiguous: {:?}", e))
    })?;
    let equities_data = equities.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("equities not contiguous: {:?}", e))
    })?;

    let signals_dev = stream.clone_htod(signals_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "H2D Transfer Error (signals): {:?}",
            e
        ))
    })?;
    let equities_dev = stream.clone_htod(equities_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "H2D Transfer Error (equities): {:?}",
            e
        ))
    })?;
    let mut avg_signal_dev = stream.alloc_zeros::<f64>(n_bars).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Device Allocation Error: {:?}",
            e
        ))
    })?;
    let mut trends_dev = stream.alloc_zeros::<i32>(n_bars).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Device Allocation Error: {:?}",
            e
        ))
    })?;

    let block_size = 256u32;
    let grid_size = ((n_bars as u32) + block_size - 1) / block_size;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&signals_dev)
            .arg(&equities_dev)
            .arg(&mut avg_signal_dev)
            .arg(&mut trends_dev)
            .arg(&(n_mas as i32))
            .arg(&(n_bars as i32))
            .arg(&cutout)
            .arg(&long_threshold)
            .arg(&short_threshold)
            .launch(cfg)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Kernel Launch Error: {:?}",
                    e
                ))
            })?;
    }

    let mut avg_signal_host = vec![0.0; n_bars];
    let mut trends_host = vec![0i32; n_bars];
    stream
        .memcpy_dtoh(&avg_signal_dev, &mut avg_signal_host)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "D2H Transfer Error (avg_signal): {:?}",
                e
            ))
        })?;
    stream
        .memcpy_dtoh(&trends_dev, &mut trends_host)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "D2H Transfer Error (trends): {:?}",
                e
            ))
        })?;

    Ok((
        PyArray1::from_vec(_py, avg_signal_host),
        PyArray1::from_vec(_py, trends_host),
    ))
}
