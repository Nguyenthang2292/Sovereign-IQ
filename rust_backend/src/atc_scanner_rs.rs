use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use lru::LruCache;
use std::num::NonZeroUsize;

/// Calculate weighted score for a timeframe signal.
#[pyfunction]
#[pyo3(signature = (signal_type, tf_weight, strength, use_signal_strength=false))]
pub fn calculate_weighted_score(
    signal_type: &str,
    tf_weight: f64,
    strength: f64,
    use_signal_strength: bool,
) -> PyResult<f64> {
    match signal_type {
        "LONG" => {
            if use_signal_strength {
                Ok(tf_weight * strength.abs())
            } else {
                Ok(tf_weight)
            }
        }
        "SHORT" => {
            if use_signal_strength {
                Ok(tf_weight * strength)
            } else {
                Ok(-tf_weight)
            }
        }
        _ => Ok(0.0),
    }
}

struct ParsedSignalData {
    longs: HashSet<String>,
    shorts: HashSet<String>,
    strengths: HashMap<String, f64>,
}

/// Aggregate signals across timeframes.
#[pyfunction]
#[pyo3(signature = (symbols, results_by_tf, weights, threshold, use_signal_strength=false))]
pub fn aggregate_signals(
    py: Python<'_>,
    symbols: Vec<String>,
    results_by_tf: HashMap<String, PyObject>,
    weights: HashMap<String, f64>,
    threshold: f64,
    use_signal_strength: bool,
) -> PyResult<Vec<PyObject>> {
    // Pre-parse results from Python objects to Rust structures for efficiency
    let mut parsed_results: HashMap<String, ParsedSignalData> = HashMap::new();

    for (tf, res_obj) in results_by_tf {
        let res_dict = res_obj.downcast_bound::<pyo3::types::PyDict>(py)?;

        let longs_obj = res_dict
            .get_item("longs")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'longs' key"))?;
        let longs: HashSet<String> = longs_obj.extract()?;

        let shorts_obj = res_dict
            .get_item("shorts")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'shorts' key"))?;
        let shorts: HashSet<String> = shorts_obj.extract()?;

        let strengths_obj = res_dict.get_item("strengths")?.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'strengths' key")
        })?;
        let strengths: HashMap<String, f64> = strengths_obj.extract()?;

        parsed_results.insert(
            tf,
            ParsedSignalData {
                longs,
                shorts,
                strengths,
            },
        );
    }

    let mut final_signals = Vec::new();

    for symbol in symbols {
        let mut score = 0.0;
        let mut details = HashMap::new();
        let mut strengths_map = HashMap::new();

        // Check each timeframe
        for (tf, res) in &parsed_results {
            let tf_weight = weights.get(tf).copied().unwrap_or(0.0);
            let strength = res.strengths.get(&symbol).copied().unwrap_or(0.0);

            strengths_map.insert(tf, strength);

            let signal_type = if res.longs.contains(&symbol) {
                "LONG"
            } else if res.shorts.contains(&symbol) {
                "SHORT"
            } else {
                "NEUTRAL"
            };

            details.insert(tf, signal_type);

            if signal_type == "LONG" {
                if use_signal_strength {
                    score += tf_weight * strength.abs();
                } else {
                    score += tf_weight;
                }
            } else if signal_type == "SHORT" {
                if use_signal_strength {
                    score += tf_weight * strength;
                } else {
                    score -= tf_weight;
                }
            }
        }

        // Apply threshold
        let final_signal_type = if score > threshold {
            "LONG"
        } else if score < -threshold {
            "SHORT"
        } else {
            "NEUTRAL"
        };

        if final_signal_type != "NEUTRAL" {
            let result_dict = pyo3::types::PyDict::new(py);
            result_dict.set_item("symbol", &symbol)?;
            result_dict.set_item("score", (score * 100.0).round() / 100.0)?;
            result_dict.set_item("signal_type", final_signal_type)?;
            result_dict.set_item("details", &details)?;
            result_dict.set_item("strengths", &strengths_map)?;

            final_signals.push(result_dict.into());
        }
    }

    Ok(final_signals)
}

/// Cached scan result data structure.
#[derive(Clone, Debug)]
struct CacheEntry {
    longs: HashSet<String>,
    shorts: HashSet<String>,
    strengths: HashMap<String, f64>,
    timestamp: f64,
}

/// Thread-safe LRU cache for ATC scan results.
///
/// Uses RwLock<LruCache> to provide:
/// - Thread-safe concurrent reads (multiple readers)
/// - Exclusive writes (single writer)
/// - LRU eviction policy
/// - TTL-based expiration
#[pyclass]
pub struct ScanCache {
    cache: Arc<RwLock<LruCache<String, CacheEntry>>>,
    ttl_seconds: f64,
}

#[pymethods]
impl ScanCache {
    /// Create a new ScanCache.
    ///
    /// Args:
    ///     capacity: Maximum number of cache entries (default: 1000)
    ///     ttl_seconds: Time-to-live for cache entries in seconds (default: 60.0)
    #[new]
    #[pyo3(signature = (capacity=1000, ttl_seconds=60.0))]
    fn new(capacity: usize, ttl_seconds: f64) -> PyResult<Self> {
        let cap = NonZeroUsize::new(capacity)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Capacity must be > 0"))?;

        Ok(ScanCache {
            cache: Arc::new(RwLock::new(LruCache::new(cap))),
            ttl_seconds,
        })
    }

    /// Get cached result for a symbol-timeframe key.
    ///
    /// Args:
    ///     key: Cache key (format: "symbol:timeframe")
    ///
    /// Returns:
    ///     Dict with 'longs', 'shorts', 'strengths' if found and not expired,
    ///     None otherwise
    fn get(&self, py: Python<'_>, key: String) -> PyResult<Option<PyObject>> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Time error: {}", e)))?
            .as_secs_f64();

        // Acquire read lock
        let mut cache = self.cache.write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        if let Some(entry) = cache.get(&key) {
            // Check if entry is expired
            if current_time - entry.timestamp > self.ttl_seconds {
                // Remove expired entry
                cache.pop(&key);
                return Ok(None);
            }

            // Return cached data as Python dict
            let result = pyo3::types::PyDict::new(py);
            result.set_item("longs", entry.longs.clone())?;
            result.set_item("shorts", entry.shorts.clone())?;
            result.set_item("strengths", entry.strengths.clone())?;

            return Ok(Some(result.into()));
        }

        Ok(None)
    }

    /// Store scan result in cache.
    ///
    /// Args:
    ///     key: Cache key (format: "symbol:timeframe")
    ///     longs: Set of symbols with LONG signals
    ///     shorts: Set of symbols with SHORT signals
    ///     strengths: Dict mapping symbol to signal strength
    fn set(
        &self,
        key: String,
        longs: HashSet<String>,
        shorts: HashSet<String>,
        strengths: HashMap<String, f64>,
    ) -> PyResult<()> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Time error: {}", e)))?
            .as_secs_f64();

        let entry = CacheEntry {
            longs,
            shorts,
            strengths,
            timestamp: current_time,
        };

        // Acquire write lock
        let mut cache = self.cache.write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        cache.put(key, entry);

        Ok(())
    }

    /// Check if key exists in cache and is not expired.
    ///
    /// Args:
    ///     key: Cache key to check
    ///
    /// Returns:
    ///     True if key exists and is not expired, False otherwise
    fn contains(&self, key: String) -> PyResult<bool> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Time error: {}", e)))?
            .as_secs_f64();

        // Acquire read lock
        let cache = self.cache.read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        if let Some(entry) = cache.peek(&key) {
            Ok(current_time - entry.timestamp <= self.ttl_seconds)
        } else {
            Ok(false)
        }
    }

    /// Clear all cache entries.
    fn clear(&self) -> PyResult<()> {
        let mut cache = self.cache.write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        cache.clear();
        Ok(())
    }

    /// Get current cache size.
    ///
    /// Returns:
    ///     Number of entries currently in cache
    fn len(&self) -> PyResult<usize> {
        let cache = self.cache.read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        Ok(cache.len())
    }

    /// Get cache capacity.
    ///
    /// Returns:
    ///     Maximum number of entries the cache can hold
    fn capacity(&self) -> PyResult<usize> {
        let cache = self.cache.read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        Ok(cache.cap().get())
    }

    /// Remove expired entries from cache.
    ///
    /// Returns:
    ///     Number of expired entries removed
    fn remove_expired(&self) -> PyResult<usize> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Time error: {}", e)))?
            .as_secs_f64();

        let mut cache = self.cache.write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        let mut expired_keys = Vec::new();

        // Collect expired keys (can't remove while iterating)
        for (key, entry) in cache.iter() {
            if current_time - entry.timestamp > self.ttl_seconds {
                expired_keys.push(key.clone());
            }
        }

        let count = expired_keys.len();

        // Remove expired entries
        for key in expired_keys {
            cache.pop(&key);
        }

        Ok(count)
    }

    /// String representation of cache state.
    fn __repr__(&self) -> PyResult<String> {
        let cache = self.cache.read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e)))?;

        Ok(format!(
            "ScanCache(size={}/{}, ttl={}s)",
            cache.len(),
            cache.cap().get(),
            self.ttl_seconds
        ))
    }
}

