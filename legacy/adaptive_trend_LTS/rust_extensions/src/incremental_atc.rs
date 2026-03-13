use pyo3::prelude::*;
use std::collections::VecDeque;

/// State for O(1) Weighted Moving Average
#[derive(Clone)]
struct O1WMAState {
    length: usize,
    denominator: f64,
    price_window: VecDeque<f64>,
    weighted_sum: f64,
    current_value: f64,
    is_initialized: bool,
}

impl O1WMAState {
    fn new(length: usize) -> Self {
        O1WMAState {
            length,
            denominator: (length * (length + 1) / 2) as f64,
            price_window: VecDeque::with_capacity(length),
            weighted_sum: 0.0,
            current_value: 0.0,
            is_initialized: false,
        }
    }

    fn update(&mut self, price: f64) -> f64 {
        if !self.is_initialized {
            self.price_window.push_back(price);
            self.weighted_sum += price * self.price_window.len() as f64;

            if self.price_window.len() == self.length {
                self.current_value = self.weighted_sum / self.denominator;
                self.is_initialized = true;
            } else if !self.price_window.is_empty() {
                let sum: f64 = self.price_window.iter().sum();
                self.current_value = sum / self.price_window.len() as f64;
            }
            return self.current_value;
        }

        // O(1) update: shift weights
        let oldest_price = self.price_window.pop_front().unwrap();
        let sum_y: f64 = self.price_window.iter().sum();

        self.price_window.push_back(price);
        self.weighted_sum -= sum_y + oldest_price;
        self.weighted_sum += price * self.length as f64;

        self.current_value = self.weighted_sum / self.denominator;
        self.current_value
    }
}

/// State for O(1) Hull Moving Average (using nested O1WMA)
#[derive(Clone)]
struct O1HMAState {
    wma_half: O1WMAState,
    wma_full: O1WMAState,
    wma_final: O1WMAState,
    intermediate_series: VecDeque<f64>,
    current_value: f64,
    is_initialized: bool,
}

impl O1HMAState {
    fn new(length: usize) -> Self {
        let half_len = (length / 2).max(1);
        let sqrt_len = (length as f64).sqrt() as usize;

        O1HMAState {
            wma_half: O1WMAState::new(half_len),
            wma_full: O1WMAState::new(length),
            wma_final: O1WMAState::new(sqrt_len),
            intermediate_series: VecDeque::with_capacity(sqrt_len),
            current_value: 0.0,
            is_initialized: false,
        }
    }

    fn update(&mut self, price: f64) -> f64 {
        let half_val = self.wma_half.update(price);
        let full_val = self.wma_full.update(price);

        let intermediate = 2.0 * half_val - full_val;
        self.intermediate_series.push_back(intermediate);

        let final_val = self.wma_final.update(intermediate);

        if self.wma_final.is_initialized {
            self.current_value = final_val;
            self.is_initialized = true;
        } else if !self.intermediate_series.is_empty() {
            let sum: f64 = self.intermediate_series.iter().sum();
            self.current_value = sum / self.intermediate_series.len() as f64;
        }

        self.current_value
    }
}

/// State for O(1) Least Squares Moving Average
#[derive(Clone)]
struct O1LSMAState {
    length: usize,
    x_values: Vec<f64>,
    sum_x: f64,
    sum_x2: f64,
    denom: f64,
    price_window: VecDeque<f64>,
    sum_y: f64,
    sum_xy: f64,
    current_value: f64,
    is_initialized: bool,
}

impl O1LSMAState {
    fn new(length: usize) -> Self {
        let x_values: Vec<f64> = (0..length).map(|i| i as f64).collect();
        let sum_x: f64 = x_values.iter().sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();
        let denom = (length as f64) * sum_x2 - sum_x * sum_x;

        O1LSMAState {
            length,
            x_values,
            sum_x,
            sum_x2,
            denom,
            price_window: VecDeque::with_capacity(length),
            sum_y: 0.0,
            sum_xy: 0.0,
            current_value: 0.0,
            is_initialized: false,
        }
    }

    fn update(&mut self, price: f64) -> f64 {
        if !self.is_initialized && self.price_window.len() < self.length {
            let idx = self.price_window.len();
            self.price_window.push_back(price);
            self.sum_y += price;
            self.sum_xy += price * self.x_values[idx];

            if self.price_window.len() == self.length {
                self.compute_lsma();
                self.is_initialized = true;
            } else {
                self.current_value = price;
            }
            return self.current_value;
        }

        let oldest_price = self.price_window.pop_front().unwrap();

        self.price_window.push_back(price);
        self.sum_y -= oldest_price;
        self.sum_y += price;
        self.sum_xy -= self.sum_y - price;
        self.sum_xy += price * self.x_values[self.length - 1];

        self.compute_lsma();
        self.current_value
    }

    fn compute_lsma(&mut self) {
        if self.denom == 0.0 {
            if !self.price_window.is_empty() {
                self.current_value =
                    self.price_window.iter().sum::<f64>() / self.price_window.len() as f64;
            }
            return;
        }

        let slope = (self.length as f64 * self.sum_xy - self.sum_x * self.sum_y) / self.denom;
        let intercept = (self.sum_y - slope * self.sum_x) / self.length as f64;
        self.current_value = intercept + slope * self.x_values[self.length - 1];
    }
}

/// State for O(1) Kaufman Adaptive Moving Average
#[derive(Clone)]
struct O1KAMAState {
    length: usize,
    fast_sc: f64,
    slow_sc: f64,
    price_window: VecDeque<f64>,
    volatility_sum: f64,
    current_value: f64,
    is_initialized: bool,
}

impl O1KAMAState {
    fn new(length: usize) -> Self {
        O1KAMAState {
            length,
            fast_sc: 2.0 / (2.0 + 1.0),
            slow_sc: 2.0 / (30.0 + 1.0),
            price_window: VecDeque::with_capacity(length + 1),
            volatility_sum: 0.0,
            current_value: 0.0,
            is_initialized: false,
        }
    }

    fn update(&mut self, price: f64) -> f64 {
        if !self.is_initialized {
            if let Some(&prev_price) = self.price_window.back() {
                self.volatility_sum += (price - prev_price).abs();
            }
            self.price_window.push_back(price);

            if self.price_window.len() == self.length + 1 {
                self.is_initialized = true;
                self.update_kama(price);
            } else {
                self.current_value = price;
            }
            return self.current_value;
        }

        let oldest_price = self.price_window.pop_front().unwrap();
        let second_oldest = self.price_window.front().copied().unwrap_or(oldest_price);
        let prev_price = *self.price_window.back().unwrap_or(&price);

        self.volatility_sum -= (second_oldest - oldest_price).abs();
        self.volatility_sum += (price - prev_price).abs();

        self.price_window.push_back(price);
        self.update_kama(price);
        self.current_value
    }

    fn update_kama(&mut self, price: f64) {
        let change = (price - *self.price_window.front().unwrap()).abs();
        let volatility = self.volatility_sum;

        let er = if volatility != 0.0 {
            change / volatility
        } else {
            0.0
        };

        let sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc).powi(2);

        self.current_value = self.current_value + sc * (price - self.current_value);
    }
}

/// State for Incremental ATC updates
#[pyclass]
#[derive(Clone)]
pub struct IncrementalATCState {
    ema_length: usize,
    ema_value: f64,
    ema2_value: f64, // For DEMA
    wma: O1WMAState,
    hma: O1HMAState,
    lsma: O1LSMAState,
    kama: O1KAMAState,
    equity_ema: f64,
    equity_hma: f64,
    equity_wma: f64,
    equity_dema: f64,
    equity_lsma: f64,
    equity_kama: f64,
    decay: f64,
    la: f64,
    long_threshold: f64,
    short_threshold: f64,
    price_window: VecDeque<f64>,
    initialized: bool,
}

#[pymethods]
impl IncrementalATCState {
    #[new]
    #[pyo3(signature = (
        ema_length,
        hma_length,
        wma_length,
        lsma_length,
        kama_length,
        decay,
        la,
        long_threshold,
        short_threshold
    ))]
    fn new(
        ema_length: usize,
        hma_length: usize,
        wma_length: usize,
        lsma_length: usize,
        kama_length: usize,
        decay: f64,
        la: f64,
        long_threshold: f64,
        short_threshold: f64,
    ) -> Self {
        let max_history = [ema_length, hma_length, wma_length, lsma_length, kama_length]
            .iter()
            .max()
            .unwrap()
            + 1;

        IncrementalATCState {
            ema_length,
            ema_value: 0.0,
            ema2_value: 0.0,
            wma: O1WMAState::new(wma_length),
            hma: O1HMAState::new(hma_length),
            lsma: O1LSMAState::new(lsma_length),
            kama: O1KAMAState::new(kama_length),
            equity_ema: 1.0,
            equity_hma: 1.0,
            equity_wma: 1.0,
            equity_dema: 1.0,
            equity_lsma: 1.0,
            equity_kama: 1.0,
            decay,
            la,
            long_threshold,
            short_threshold,
            price_window: VecDeque::with_capacity(max_history),
            initialized: false,
        }
    }

    /// Initialize the state with historical prices
    fn initialize(&mut self, prices: Vec<f64>) {
        self.price_window.clear();
        for price in &prices {
            self.price_window.push_back(*price);
        }

        // Initialize EMA value (simplified - would normally use full calculation)
        if !prices.is_empty() {
            self.ema_value = prices[prices.len() - 1];
        }

        self.initialized = true;
    }

    /// Get current state as a dictionary (for Python serialization)
    fn get_state(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let state = pyo3::types::PyDict::new(py);
            state.set_item("ema_value", self.ema_value)?;
            state.set_item("wma", self.wma.current_value)?;
            state.set_item("hma", self.hma.current_value)?;
            state.set_item("lsma", self.lsma.current_value)?;
            state.set_item("kama", self.kama.current_value)?;
            state.set_item("dema", 2.0 * self.ema_value - self.ema2_value)?;
            state.set_item("equity_ema", self.equity_ema)?;
            state.set_item("equity_hma", self.equity_hma)?;
            state.set_item("equity_wma", self.equity_wma)?;
            state.set_item("equity_dema", self.equity_dema)?;
            state.set_item("equity_lsma", self.equity_lsma)?;
            state.set_item("equity_kama", self.equity_kama)?;
            state.set_item("initialized", self.initialized)?;
            Ok(state.into())
        })
    }

    /// Reset the state
    fn reset(&mut self) {
        let ema_len = self.ema_length;
        let hma_len = self.hma.wma_full.length;
        let wma_len = self.wma.length;
        let lsma_len = self.lsma.length;
        let kama_len = self.kama.length;
        let decay = self.decay;
        let la = self.la;
        let long_threshold = self.long_threshold;
        let short_threshold = self.short_threshold;

        *self = Self::new(
            ema_len,
            hma_len,
            wma_len,
            lsma_len,
            kama_len,
            decay,
            la,
            long_threshold,
            short_threshold,
        );
    }
}

/// Update Incremental ATC with a single price bar (Rust backend)
#[pyfunction]
pub fn update_incremental_atc_rust(
    state_py: PyObject,
    new_price: f64,
    py: Python,
) -> PyResult<(f64, PyObject)> {
    // Extract state from Python object
    let state_dict = state_py.downcast_bound::<pyo3::types::PyDict>(py)?;

    // Reconstruct IncrementalATCState from dict
    let mut state = IncrementalATCState {
        ema_value: state_dict
            .get_item("ema_value")?
            .unwrap()
            .extract::<f64>()?,
        ema2_value: state_dict
            .get_item("ema2_value")?
            .unwrap()
            .extract::<f64>()?,
        wma: O1WMAState {
            length: state_dict
                .get_item("wma_length")?
                .unwrap()
                .extract::<usize>()?,
            denominator: state_dict
                .get_item("wma_denominator")?
                .unwrap()
                .extract::<f64>()?,
            price_window: VecDeque::from(
                state_dict
                    .get_item("wma_window")?
                    .unwrap()
                    .extract::<Vec<f64>>()?,
            ),
            weighted_sum: state_dict
                .get_item("wma_weighted_sum")?
                .unwrap()
                .extract::<f64>()?,
            current_value: state_dict
                .get_item("wma_value")?
                .unwrap()
                .extract::<f64>()?,
            is_initialized: state_dict
                .get_item("wma_initialized")?
                .unwrap()
                .extract::<bool>()?,
        },
        hma: O1HMAState {
            wma_half: O1WMAState {
                length: state_dict
                    .get_item("hma_half_length")?
                    .unwrap()
                    .extract::<usize>()?,
                denominator: state_dict
                    .get_item("hma_half_denominator")?
                    .unwrap()
                    .extract::<f64>()?,
                price_window: VecDeque::from(
                    state_dict
                        .get_item("hma_half_window")?
                        .unwrap()
                        .extract::<Vec<f64>>()?,
                ),
                weighted_sum: state_dict
                    .get_item("hma_half_weighted_sum")?
                    .unwrap()
                    .extract::<f64>()?,
                current_value: state_dict
                    .get_item("hma_half_value")?
                    .unwrap()
                    .extract::<f64>()?,
                is_initialized: state_dict
                    .get_item("hma_half_initialized")?
                    .unwrap()
                    .extract::<bool>()?,
            },
            wma_full: O1WMAState {
                length: state_dict
                    .get_item("hma_full_length")?
                    .unwrap()
                    .extract::<usize>()?,
                denominator: state_dict
                    .get_item("hma_full_denominator")?
                    .unwrap()
                    .extract::<f64>()?,
                price_window: VecDeque::from(
                    state_dict
                        .get_item("hma_full_window")?
                        .unwrap()
                        .extract::<Vec<f64>>()?,
                ),
                weighted_sum: state_dict
                    .get_item("hma_full_weighted_sum")?
                    .unwrap()
                    .extract::<f64>()?,
                current_value: state_dict
                    .get_item("hma_full_value")?
                    .unwrap()
                    .extract::<f64>()?,
                is_initialized: state_dict
                    .get_item("hma_full_initialized")?
                    .unwrap()
                    .extract::<bool>()?,
            },
            wma_final: O1WMAState {
                length: state_dict
                    .get_item("hma_final_length")?
                    .unwrap()
                    .extract::<usize>()?,
                denominator: state_dict
                    .get_item("hma_final_denominator")?
                    .unwrap()
                    .extract::<f64>()?,
                price_window: VecDeque::from(
                    state_dict
                        .get_item("hma_final_window")?
                        .unwrap()
                        .extract::<Vec<f64>>()?,
                ),
                weighted_sum: state_dict
                    .get_item("hma_final_weighted_sum")?
                    .unwrap()
                    .extract::<f64>()?,
                current_value: state_dict
                    .get_item("hma_final_value")?
                    .unwrap()
                    .extract::<f64>()?,
                is_initialized: state_dict
                    .get_item("hma_final_initialized")?
                    .unwrap()
                    .extract::<bool>()?,
            },
            intermediate_series: VecDeque::from(
                state_dict
                    .get_item("hma_intermediate_series")?
                    .unwrap()
                    .extract::<Vec<f64>>()?,
            ),
            current_value: state_dict
                .get_item("hma_value")?
                .unwrap()
                .extract::<f64>()?,
            is_initialized: state_dict
                .get_item("hma_initialized")?
                .unwrap()
                .extract::<bool>()?,
        },
        lsma: O1LSMAState {
            length: state_dict
                .get_item("lsma_length")?
                .unwrap()
                .extract::<usize>()?,
            x_values: state_dict
                .get_item("lsma_x_values")?
                .unwrap()
                .extract::<Vec<f64>>()?,
            sum_x: state_dict
                .get_item("lsma_sum_x")?
                .unwrap()
                .extract::<f64>()?,
            sum_x2: state_dict
                .get_item("lsma_sum_x2")?
                .unwrap()
                .extract::<f64>()?,
            denom: state_dict
                .get_item("lsma_denom")?
                .unwrap()
                .extract::<f64>()?,
            price_window: VecDeque::from(
                state_dict
                    .get_item("lsma_window")?
                    .unwrap()
                    .extract::<Vec<f64>>()?,
            ),
            sum_y: state_dict
                .get_item("lsma_sum_y")?
                .unwrap()
                .extract::<f64>()?,
            sum_xy: state_dict
                .get_item("lsma_sum_xy")?
                .unwrap()
                .extract::<f64>()?,
            current_value: state_dict
                .get_item("lsma_value")?
                .unwrap()
                .extract::<f64>()?,
            is_initialized: state_dict
                .get_item("lsma_initialized")?
                .unwrap()
                .extract::<bool>()?,
        },
        kama: O1KAMAState {
            length: state_dict
                .get_item("kama_length")?
                .unwrap()
                .extract::<usize>()?,
            fast_sc: state_dict
                .get_item("kama_fast_sc")?
                .unwrap()
                .extract::<f64>()?,
            slow_sc: state_dict
                .get_item("kama_slow_sc")?
                .unwrap()
                .extract::<f64>()?,
            price_window: VecDeque::from(
                state_dict
                    .get_item("kama_window")?
                    .unwrap()
                    .extract::<Vec<f64>>()?,
            ),
            volatility_sum: state_dict
                .get_item("kama_volatility_sum")?
                .unwrap()
                .extract::<f64>()?,
            current_value: state_dict
                .get_item("kama_value")?
                .unwrap()
                .extract::<f64>()?,
            is_initialized: state_dict
                .get_item("kama_initialized")?
                .unwrap()
                .extract::<bool>()?,
        },
        equity_ema: state_dict
            .get_item("equity_ema")?
            .unwrap()
            .extract::<f64>()?,
        equity_hma: state_dict
            .get_item("equity_hma")?
            .unwrap()
            .extract::<f64>()?,
        equity_wma: state_dict
            .get_item("equity_wma")?
            .unwrap()
            .extract::<f64>()?,
        equity_dema: state_dict
            .get_item("equity_dema")?
            .unwrap()
            .extract::<f64>()?,
        equity_lsma: state_dict
            .get_item("equity_lsma")?
            .unwrap()
            .extract::<f64>()?,
        equity_kama: state_dict
            .get_item("equity_kama")?
            .unwrap()
            .extract::<f64>()?,
        decay: state_dict.get_item("decay")?.unwrap().extract::<f64>()?,
        la: state_dict.get_item("la")?.unwrap().extract::<f64>()?,
        long_threshold: state_dict
            .get_item("long_threshold")?
            .unwrap()
            .extract::<f64>()?,
        short_threshold: state_dict
            .get_item("short_threshold")?
            .unwrap()
            .extract::<f64>()?,
        ema_length: state_dict
            .get_item("ema_length")?
            .unwrap()
            .extract::<usize>()?,
        price_window: VecDeque::from(
            state_dict
                .get_item("price_window")?
                .unwrap()
                .extract::<Vec<f64>>()?,
        ),
        initialized: state_dict
            .get_item("initialized")?
            .unwrap()
            .extract::<bool>()?,
    };

    // Update MAs
    state.wma.update(new_price);
    state.hma.update(new_price);
    state.lsma.update(new_price);
    state.kama.update(new_price);

    // Update EMA
    let alpha = 2.0 / (state.ema_length as f64 + 1.0);
    state.ema_value = alpha * new_price + (1.0 - alpha) * state.ema_value;

    // Update EMA2 for DEMA
    let prev_ema2 = state.ema2_value;
    state.ema2_value = alpha * state.ema_value + (1.0 - alpha) * prev_ema2;

    // Calculate DEMA
    let dema_value = 2.0 * state.ema_value - state.ema2_value;

    // Calculate Layer 1 signals (simplified - would normally use full layer1 logic)
    let signal_l1_ema = if new_price > state.ema_value {
        1.0
    } else {
        -1.0
    };
    let signal_l1_hma = if new_price > state.hma.current_value {
        1.0
    } else {
        -1.0
    };
    let signal_l1_wma = if new_price > state.wma.current_value {
        1.0
    } else {
        -1.0
    };
    let signal_l1_dema = if new_price > dema_value { 1.0 } else { -1.0 };
    let signal_l1_lsma = if new_price > state.lsma.current_value {
        1.0
    } else {
        -1.0
    };
    let signal_l1_kama = if new_price > state.kama.current_value {
        1.0
    } else {
        -1.0
    };

    // Average Layer 1 signal (computed for potential future use / debugging)
    let _signal_l1 = (signal_l1_ema
        + signal_l1_hma
        + signal_l1_wma
        + signal_l1_dema
        + signal_l1_lsma
        + signal_l1_kama)
        / 6.0;

    // Update equities
    state.equity_ema = state.equity_ema * (1.0 - state.decay) + signal_l1_ema * state.la;
    state.equity_hma = state.equity_hma * (1.0 - state.decay) + signal_l1_hma * state.la;
    state.equity_wma = state.equity_wma * (1.0 - state.decay) + signal_l1_wma * state.la;
    state.equity_dema = state.equity_dema * (1.0 - state.decay) + signal_l1_dema * state.la;
    state.equity_lsma = state.equity_lsma * (1.0 - state.decay) + signal_l1_lsma * state.la;
    state.equity_kama = state.equity_kama * (1.0 - state.decay) + signal_l1_kama * state.la;

    // Calculate discrete signals
    let c_ema = if signal_l1_ema > state.long_threshold {
        1.0
    } else if signal_l1_ema < state.short_threshold {
        -1.0
    } else {
        0.0
    };
    let c_hma = if signal_l1_hma > state.long_threshold {
        1.0
    } else if signal_l1_hma < state.short_threshold {
        -1.0
    } else {
        0.0
    };
    let c_wma = if signal_l1_wma > state.long_threshold {
        1.0
    } else if signal_l1_wma < state.short_threshold {
        -1.0
    } else {
        0.0
    };
    let c_dema = if signal_l1_dema > state.long_threshold {
        1.0
    } else if signal_l1_dema < state.short_threshold {
        -1.0
    } else {
        0.0
    };
    let c_lsma = if signal_l1_lsma > state.long_threshold {
        1.0
    } else if signal_l1_lsma < state.short_threshold {
        -1.0
    } else {
        0.0
    };
    let c_kama = if signal_l1_kama > state.long_threshold {
        1.0
    } else if signal_l1_kama < state.short_threshold {
        -1.0
    } else {
        0.0
    };

    // Calculate final signal (weighted average)
    let nom = c_ema * state.equity_ema
        + c_hma * state.equity_hma
        + c_wma * state.equity_wma
        + c_dema * state.equity_dema
        + c_lsma * state.equity_lsma
        + c_kama * state.equity_kama;
    let den = state.equity_ema
        + state.equity_hma
        + state.equity_wma
        + state.equity_dema
        + state.equity_lsma
        + state.equity_kama;

    let signal = if den != 0.0 { nom / den } else { 0.0 };

    // Serialize updated state back to Python dict
    let updated_state = pyo3::types::PyDict::new(py);
    updated_state.set_item("ema_value", state.ema_value)?;
    updated_state.set_item("ema2_value", state.ema2_value)?;
    updated_state.set_item("wma_length", state.wma.length)?;
    updated_state.set_item("wma_denominator", state.wma.denominator)?;
    updated_state.set_item(
        "wma_window",
        state.wma.price_window.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("wma_weighted_sum", state.wma.weighted_sum)?;
    updated_state.set_item("wma_value", state.wma.current_value)?;
    updated_state.set_item("wma_initialized", state.wma.is_initialized)?;
    updated_state.set_item("hma_half_length", state.hma.wma_half.length)?;
    updated_state.set_item("hma_half_denominator", state.hma.wma_half.denominator)?;
    updated_state.set_item(
        "hma_half_window",
        state.hma.wma_half.price_window.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("hma_half_weighted_sum", state.hma.wma_half.weighted_sum)?;
    updated_state.set_item("hma_half_value", state.hma.wma_half.current_value)?;
    updated_state.set_item("hma_half_initialized", state.hma.wma_half.is_initialized)?;
    updated_state.set_item("hma_full_length", state.hma.wma_full.length)?;
    updated_state.set_item("hma_full_denominator", state.hma.wma_full.denominator)?;
    updated_state.set_item(
        "hma_full_window",
        state.hma.wma_full.price_window.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("hma_full_weighted_sum", state.hma.wma_full.weighted_sum)?;
    updated_state.set_item("hma_full_value", state.hma.wma_full.current_value)?;
    updated_state.set_item("hma_full_initialized", state.hma.wma_full.is_initialized)?;
    updated_state.set_item("hma_final_length", state.hma.wma_final.length)?;
    updated_state.set_item("hma_final_denominator", state.hma.wma_final.denominator)?;
    updated_state.set_item(
        "hma_final_window",
        state.hma.wma_final.price_window.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("hma_final_weighted_sum", state.hma.wma_final.weighted_sum)?;
    updated_state.set_item("hma_final_value", state.hma.wma_final.current_value)?;
    updated_state.set_item("hma_final_initialized", state.hma.wma_final.is_initialized)?;
    updated_state.set_item(
        "hma_intermediate_series",
        state.hma.intermediate_series.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("hma_value", state.hma.current_value)?;
    updated_state.set_item("hma_initialized", state.hma.is_initialized)?;
    updated_state.set_item("lsma_length", state.lsma.length)?;
    updated_state.set_item("lsma_x_values", &state.lsma.x_values)?;
    updated_state.set_item("lsma_sum_x", state.lsma.sum_x)?;
    updated_state.set_item("lsma_sum_x2", state.lsma.sum_x2)?;
    updated_state.set_item("lsma_denom", state.lsma.denom)?;
    updated_state.set_item(
        "lsma_window",
        state.lsma.price_window.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("lsma_sum_y", state.lsma.sum_y)?;
    updated_state.set_item("lsma_sum_xy", state.lsma.sum_xy)?;
    updated_state.set_item("lsma_value", state.lsma.current_value)?;
    updated_state.set_item("lsma_initialized", state.lsma.is_initialized)?;
    updated_state.set_item("kama_length", state.kama.length)?;
    updated_state.set_item("kama_fast_sc", state.kama.fast_sc)?;
    updated_state.set_item("kama_slow_sc", state.kama.slow_sc)?;
    updated_state.set_item(
        "kama_window",
        state.kama.price_window.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("kama_volatility_sum", state.kama.volatility_sum)?;
    updated_state.set_item("kama_value", state.kama.current_value)?;
    updated_state.set_item("kama_initialized", state.kama.is_initialized)?;
    updated_state.set_item("equity_ema", state.equity_ema)?;
    updated_state.set_item("equity_hma", state.equity_hma)?;
    updated_state.set_item("equity_wma", state.equity_wma)?;
    updated_state.set_item("equity_dema", state.equity_dema)?;
    updated_state.set_item("equity_lsma", state.equity_lsma)?;
    updated_state.set_item("equity_kama", state.equity_kama)?;
    updated_state.set_item("decay", state.decay)?;
    updated_state.set_item("la", state.la)?;
    updated_state.set_item("long_threshold", state.long_threshold)?;
    updated_state.set_item("short_threshold", state.short_threshold)?;
    updated_state.set_item("ema_length", state.ema_length)?;
    updated_state.set_item(
        "price_window",
        state.price_window.iter().collect::<Vec<_>>(),
    )?;
    updated_state.set_item("initialized", state.initialized)?;

    Ok((signal, updated_state.into()))
}
