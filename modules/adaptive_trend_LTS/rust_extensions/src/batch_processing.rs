//! Phase 4 - True Batch CUDA Processing: Rust-CUDA Bridge
//!
//! This module provides the full ATC signal computation pipeline in batch mode.

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, OnceLock};

const BATCH_MA_KERNELS_SRC: &str = include_str!("../../core/gpu_backend/batch_ma_kernels.cu");
const BATCH_SIGNAL_KERNELS_SRC: &str =
    include_str!("../../core/gpu_backend/batch_signal_kernels.cu");

struct BatchCudaCache {
    ctx: Arc<CudaContext>,
    ma_module: Arc<CudaModule>,
    signal_module: Arc<CudaModule>,
}

static BATCH_CUDA_CACHE: OnceLock<Result<BatchCudaCache, String>> = OnceLock::new();

fn get_batch_cuda_cache() -> Result<&'static BatchCudaCache, PyErr> {
    let result = BATCH_CUDA_CACHE.get_or_init(|| {
        let ctx = CudaContext::new(0).map_err(|e| format!("{:?}", e))?;

        // Enable debug symbols for Nsight debugging

        let ma_ptx = compile_ptx(BATCH_MA_KERNELS_SRC).map_err(|e| format!("{:?}", e))?;
        let ma_module = ctx.load_module(ma_ptx).map_err(|e| format!("{:?}", e))?;

        let signal_ptx = compile_ptx(BATCH_SIGNAL_KERNELS_SRC).map_err(|e| format!("{:?}", e))?;
        let signal_module = ctx
            .load_module(signal_ptx)
            .map_err(|e| format!("{:?}", e))?;
        Ok(BatchCudaCache {
            ctx,
            ma_module,
            signal_module,
        })
    });
    match result {
        Ok(cache) => Ok(cache),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.clone())),
    }
}

fn get_diflen(length: usize, robustness: &str) -> Vec<usize> {
    let (l1, l2, l3, l4, l_1, l_2, l_3, l_4) = match robustness {
        "Narrow" => (
            length + 1,
            length + 2,
            length + 3,
            length + 4,
            length.saturating_sub(1),
            length.saturating_sub(2),
            length.saturating_sub(3),
            length.saturating_sub(4),
        ),
        "Wide" => (
            length + 1,
            length + 3,
            length + 5,
            length + 7,
            length.saturating_sub(1),
            length.saturating_sub(3),
            length.saturating_sub(5),
            length.saturating_sub(7),
        ),
        _ => (
            length + 1,
            length + 2,
            length + 4,
            length + 6,
            length.saturating_sub(1),
            length.saturating_sub(2),
            length.saturating_sub(4),
            length.saturating_sub(6),
        ),
    };
    vec![
        length,
        l1,
        l2,
        l3,
        l4,
        l_1.max(1),
        l_2.max(1),
        l_3.max(1),
        l_4.max(1),
    ]
}

pub struct BatchSymbolData {
    pub all_prices: Vec<f64>,
    pub offsets: Vec<i32>,
    pub lengths: Vec<i32>,
    pub num_symbols: usize,
}

impl BatchSymbolData {
    pub fn from_python_dict(symbols_data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let num_symbols = symbols_data.len();
        let mut all_prices = Vec::new();
        let mut offsets = Vec::with_capacity(num_symbols);
        let mut lengths = Vec::with_capacity(num_symbols);
        for (_, value) in symbols_data.iter() {
            let prices: PyReadonlyArray1<f64> = value.extract()?;
            let slice = prices.as_slice()?;
            offsets.push(all_prices.len() as i32);
            lengths.push(slice.len() as i32);
            all_prices.extend_from_slice(slice);
        }
        Ok(BatchSymbolData {
            all_prices,
            offsets,
            lengths,
            num_symbols,
        })
    }
}

fn compute_ma_dev(
    cache: &BatchCudaCache,
    prices_dev: &CudaSlice<f64>,
    offsets_dev: &CudaSlice<i32>,
    lengths_dev: &CudaSlice<i32>,
    ma_type: &str,
    length: usize,
    num_symbols: usize,
    total_bars: usize,
) -> Result<CudaSlice<f64>, String> {
    let stream = cache.ctx.default_stream();
    let mut res = stream
        .alloc_zeros::<f64>(total_bars)
        .map_err(|e| format!("{:?}", e))?;
    match ma_type {
        "EMA" => {
            let cfg = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&cache.ma_module.load_function("batch_ema_kernel").unwrap())
                    .arg(prices_dev)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut res)
                    .arg(&(length as i32))
                    .arg(&(num_symbols as i32))
                    .launch(cfg)
                    .map_err(|e| format!("{:?}", e))?;
            }
        }
        "WMA" => {
            let cfg = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&cache.ma_module.load_function("batch_wma_kernel").unwrap())
                    .arg(prices_dev)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut res)
                    .arg(&(length as i32))
                    .arg(&(num_symbols as i32))
                    .launch(cfg)
                    .map_err(|e| format!("{:?}", e))?;
            }
        }
        "LSMA" => {
            let cfg = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&cache.ma_module.load_function("batch_lsma_kernel").unwrap())
                    .arg(prices_dev)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut res)
                    .arg(&(length as i32))
                    .arg(&(num_symbols as i32))
                    .launch(cfg)
                    .map_err(|e| format!("{:?}", e))?;
            }
        }
        "DEMA" => {
            let e1 = compute_ma_dev(
                cache,
                prices_dev,
                offsets_dev,
                lengths_dev,
                "EMA",
                length,
                num_symbols,
                total_bars,
            )?;
            let mut e2 = stream.alloc_zeros::<f64>(total_bars).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(
                        &cache
                            .ma_module
                            .load_function("batch_ema_simple_kernel")
                            .unwrap(),
                    )
                    .arg(&e1)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut e2)
                    .arg(&(length as i32))
                    .arg(&(num_symbols as i32))
                    .launch(cfg)
                    .unwrap();
            }
            let cfg_comb = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(
                        &cache
                            .ma_module
                            .load_function("batch_linear_combine_kernel")
                            .unwrap(),
                    )
                    .arg(&e1)
                    .arg(&e2)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut res)
                    .arg(&2.0)
                    .arg(&1.0)
                    .arg(&(num_symbols as i32))
                    .launch(cfg_comb)
                    .unwrap();
            }
        }
        "HMA" => {
            let h = std::cmp::max(length / 2, 1);
            let sq = std::cmp::max((length as f64).sqrt() as usize, 1);
            let wh = compute_ma_dev(
                cache,
                prices_dev,
                offsets_dev,
                lengths_dev,
                "WMA",
                h,
                num_symbols,
                total_bars,
            )?;
            let wf = compute_ma_dev(
                cache,
                prices_dev,
                offsets_dev,
                lengths_dev,
                "WMA",
                length,
                num_symbols,
                total_bars,
            )?;
            let mut diff = stream.alloc_zeros::<f64>(total_bars).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(
                        &cache
                            .ma_module
                            .load_function("batch_linear_combine_kernel")
                            .unwrap(),
                    )
                    .arg(&wh)
                    .arg(&wf)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut diff)
                    .arg(&2.0)
                    .arg(&1.0)
                    .arg(&(num_symbols as i32))
                    .launch(cfg)
                    .unwrap();
            }
            res = compute_ma_dev(
                cache,
                &diff,
                offsets_dev,
                lengths_dev,
                "WMA",
                sq,
                num_symbols,
                total_bars,
            )?;
        }
        "KAMA" => {
            let nk = cache
                .ma_module
                .load_function("batch_kama_noise_kernel")
                .unwrap();
            let sk = cache
                .ma_module
                .load_function("batch_kama_smooth_kernel")
                .unwrap();
            let mut noise = stream.alloc_zeros::<f64>(total_bars).unwrap();
            let cp = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&nk)
                    .arg(prices_dev)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut noise)
                    .arg(&(length as i32))
                    .arg(&(num_symbols as i32))
                    .launch(cp)
                    .unwrap();
            }
            let cs = LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&sk)
                    .arg(prices_dev)
                    .arg(&noise)
                    .arg(offsets_dev)
                    .arg(lengths_dev)
                    .arg(&mut res)
                    .arg(&(length as i32))
                    .arg(&(num_symbols as i32))
                    .launch(cs)
                    .unwrap();
            }
        }
        _ => return Err(format!("Bad MA type")),
    }
    Ok(res)
}

fn compute_l1_sig_dev(
    cache: &BatchCudaCache,
    prices_dev: &CudaSlice<f64>,
    roc_dev: &CudaSlice<f64>,
    offsets_dev: &CudaSlice<i32>,
    lengths_dev: &CudaSlice<i32>,
    ma_type: &str,
    length: usize,
    robustness: &str,
    _la: f64,
    de: f64,
    num_symbols: usize,
    total_bars: usize,
) -> Result<CudaSlice<f64>, String> {
    let lens = get_diflen(length, robustness);
    let mut all_s = Vec::new();
    let mut all_e = Vec::new();
    let stream = cache.ctx.default_stream();
    let p_ker = cache
        .signal_module
        .load_function("batch_signal_persistence_kernel")
        .unwrap();
    let e_ker = cache
        .signal_module
        .load_function("batch_equity_kernel")
        .unwrap();
    let s_ker = cache
        .signal_module
        .load_function("batch_shift_kernel")
        .unwrap();
    for &l in &lens {
        let ma = compute_ma_dev(
            cache,
            prices_dev,
            offsets_dev,
            lengths_dev,
            ma_type,
            l,
            num_symbols,
            total_bars,
        )?;
        let mut sig = stream.alloc_zeros::<f64>(total_bars).unwrap();
        unsafe {
            stream
                .launch_builder(&p_ker)
                .arg(prices_dev)
                .arg(&ma)
                .arg(offsets_dev)
                .arg(lengths_dev)
                .arg(&mut sig)
                .arg(&(num_symbols as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_symbols as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        let mut sig_p = stream.alloc_zeros::<f64>(total_bars).unwrap();
        unsafe {
            stream
                .launch_builder(&s_ker)
                .arg(&sig)
                .arg(offsets_dev)
                .arg(lengths_dev)
                .arg(&mut sig_p)
                .arg(&0.0)
                .arg(&(num_symbols as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_symbols as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        let mut eq = stream.alloc_zeros::<f64>(total_bars).unwrap();
        unsafe {
            stream
                .launch_builder(&e_ker)
                .arg(roc_dev)
                .arg(&sig_p)
                .arg(offsets_dev)
                .arg(lengths_dev)
                .arg(&mut eq)
                .arg(&1.0)
                .arg(&(1.0 - de))
                .arg(&0)
                .arg(&(num_symbols as i32))
                .launch(LaunchConfig {
                    grid_dim: (num_symbols as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        all_s.push(sig);
        all_e.push(eq);
    }
    let mut f_s = stream.alloc_zeros::<f64>(total_bars * 9).unwrap();
    let mut f_e = stream.alloc_zeros::<f64>(total_bars * 9).unwrap();
    for i in 0..9 {
        stream
            .memcpy_dtod(
                &all_s[i],
                &mut f_s.slice_mut(i * total_bars..(i + 1) * total_bars),
            )
            .unwrap();
        stream
            .memcpy_dtod(
                &all_e[i],
                &mut f_e.slice_mut(i * total_bars..(i + 1) * total_bars),
            )
            .unwrap();
    }
    let mut w_avg = stream.alloc_zeros::<f64>(total_bars).unwrap();
    let a_ker = cache
        .signal_module
        .load_function("batch_weighted_average_l1_kernel")
        .unwrap();
    unsafe {
        stream
            .launch_builder(&a_ker)
            .arg(&f_s)
            .arg(&f_e)
            .arg(offsets_dev)
            .arg(lengths_dev)
            .arg(&mut w_avg)
            .arg(&(total_bars as i32))
            .arg(&(num_symbols as i32))
            .launch(LaunchConfig {
                grid_dim: (num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    Ok(w_avg)
}

#[pyfunction]
#[pyo3(signature = (symbols_data, ema_len=28, hull_len=28, wma_len=28, dema_len=28, lsma_len=28, kama_len=28, ema_w=1.0, hma_w=1.0, wma_w=1.0, dema_w=1.0, lsma_w=1.0, kama_w=1.0, robustness="Medium", la=0.02, de=0.03, cutout=0, long_threshold=0.1, short_threshold=-0.1, _strategy_mode=false))]
pub fn compute_atc_signals_batch<'py>(
    py: Python<'py>,
    symbols_data: &Bound<'py, PyDict>,
    ema_len: usize,
    hull_len: usize,
    wma_len: usize,
    dema_len: usize,
    lsma_len: usize,
    kama_len: usize,
    ema_w: f64,
    hma_w: f64,
    wma_w: f64,
    dema_w: f64,
    lsma_w: f64,
    kama_w: f64,
    robustness: &str,
    la: f64,
    de: f64,
    cutout: i32,
    long_threshold: f64,
    short_threshold: f64,
    _strategy_mode: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let cache = get_batch_cuda_cache()?;
    let stream = cache.ctx.default_stream();
    let batch = BatchSymbolData::from_python_dict(symbols_data).unwrap();
    let prices_dev = stream.clone_htod(&batch.all_prices).unwrap();
    let off_dev = stream.clone_htod(&batch.offsets).unwrap();
    let len_dev = stream.clone_htod(&batch.lengths).unwrap();
    let total_bars = batch.all_prices.len();

    let mut roc_dev = stream.alloc_zeros::<f64>(total_bars).unwrap();
    unsafe {
        stream
            .launch_builder(
                &cache
                    .signal_module
                    .load_function("batch_roc_with_growth_kernel")
                    .unwrap(),
            )
            .arg(&prices_dev)
            .arg(&off_dev)
            .arg(&len_dev)
            .arg(&mut roc_dev)
            .arg(&la)
            .arg(&(batch.num_symbols as i32))
            .launch(LaunchConfig {
                grid_dim: (batch.num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    let types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"];
    let lens = [ema_len, hull_len, wma_len, dema_len, lsma_len, kama_len];
    let weights = [ema_w, hma_w, wma_w, dema_w, lsma_w, kama_w];

    let mut l1_sigs = Vec::new();
    let mut l2_eqs = Vec::new();
    let e_ker = cache
        .signal_module
        .load_function("batch_equity_kernel")
        .unwrap();
    let s_ker = cache
        .signal_module
        .load_function("batch_shift_kernel")
        .unwrap();

    for i in 0..6 {
        let sig = compute_l1_sig_dev(
            cache,
            &prices_dev,
            &roc_dev,
            &off_dev,
            &len_dev,
            types[i],
            lens[i],
            robustness,
            la,
            de,
            batch.num_symbols,
            total_bars,
        )
        .unwrap();
        let mut sig_p = stream.alloc_zeros::<f64>(total_bars).unwrap();
        unsafe {
            stream
                .launch_builder(&s_ker)
                .arg(&sig)
                .arg(&off_dev)
                .arg(&len_dev)
                .arg(&mut sig_p)
                .arg(&0.0)
                .arg(&(batch.num_symbols as i32))
                .launch(LaunchConfig {
                    grid_dim: (batch.num_symbols as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        let mut eq = stream.alloc_zeros::<f64>(total_bars).unwrap();
        unsafe {
            stream
                .launch_builder(&e_ker)
                .arg(&roc_dev)
                .arg(&sig_p)
                .arg(&off_dev)
                .arg(&len_dev)
                .arg(&mut eq)
                .arg(&weights[i])
                .arg(&(1.0 - de))
                .arg(&0)
                .arg(&(batch.num_symbols as i32))
                .launch(LaunchConfig {
                    grid_dim: (batch.num_symbols as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        l1_sigs.push(sig);
        l2_eqs.push(eq);
    }

    let mut f1 = stream.alloc_zeros::<f64>(total_bars * 6).unwrap();
    let mut f2 = stream.alloc_zeros::<f64>(total_bars * 6).unwrap();
    for i in 0..6 {
        stream
            .memcpy_dtod(
                &l1_sigs[i],
                &mut f1.slice_mut(i * total_bars..(i + 1) * total_bars),
            )
            .unwrap();
        stream
            .memcpy_dtod(
                &l2_eqs[i],
                &mut f2.slice_mut(i * total_bars..(i + 1) * total_bars),
            )
            .unwrap();
    }

    let mut avg = stream.alloc_zeros::<f64>(total_bars).unwrap();
    unsafe {
        stream
            .launch_builder(
                &cache
                    .signal_module
                    .load_function("batch_final_average_signal_kernel")
                    .unwrap(),
            )
            .arg(&f1)
            .arg(&f2)
            .arg(&off_dev)
            .arg(&len_dev)
            .arg(&mut avg)
            .arg(&long_threshold)
            .arg(&short_threshold)
            .arg(&(cutout as i32))
            .arg(&(total_bars as i32))
            .arg(&(batch.num_symbols as i32))
            .launch(LaunchConfig {
                grid_dim: (batch.num_symbols as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }

    let mut host = vec![0.0; total_bars];
    stream.memcpy_dtoh(&avg, &mut host).unwrap();
    let res = PyDict::new(py);
    for (i, (key, _)) in symbols_data.iter().enumerate() {
        let sym: String = key.extract()?;
        let start = batch.offsets[i] as usize;
        let len = batch.lengths[i] as usize;
        res.set_item(
            sym,
            PyArray1::from_vec(py, host[start..start + len].to_vec()),
        )?;
    }
    Ok(res)
}
