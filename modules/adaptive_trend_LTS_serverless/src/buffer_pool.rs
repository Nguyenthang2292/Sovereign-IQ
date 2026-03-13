use ndarray::Array1;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    // Pool stores raw vectors instead of Array1 so we can preserve allocation capacity
    // across check-out/check-in cycles (including when a larger allocation serves a
    // smaller request).
    static BUFFER_POOL: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::with_capacity(MAX_POOL_BUFFERS));
}

const MAX_POOL_BUFFERS: usize = 16;
const MISS_LOG_SAMPLE_WINDOW: u64 = 64;
const MISS_LOG_THRESHOLD: f64 = 0.30;

static BUFFER_CHECKOUTS: AtomicU64 = AtomicU64::new(0);
static BUFFER_MISSES: AtomicU64 = AtomicU64::new(0);

fn record_checkout(was_miss: bool, requested_size: usize) {
    let checkouts = BUFFER_CHECKOUTS.fetch_add(1, Ordering::Relaxed) + 1;
    if was_miss {
        BUFFER_MISSES.fetch_add(1, Ordering::Relaxed);
    }

    if checkouts < MISS_LOG_SAMPLE_WINDOW || checkouts % MISS_LOG_SAMPLE_WINDOW != 0 {
        return;
    }

    let misses = BUFFER_MISSES.load(Ordering::Relaxed);
    let miss_rate = misses as f64 / checkouts as f64;
    if miss_rate >= MISS_LOG_THRESHOLD {
        crate::log_warn!(
            "Buffer pool miss rate high: {:.1}% (misses={}, checkouts={}, latest_request={}, per-thread cap={}).",
            miss_rate * 100.0,
            misses,
            checkouts,
            requested_size,
            MAX_POOL_BUFFERS
        );
    }
}

fn checkout_buffer(size: usize, fill_value: f64) -> Vec<f64> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let mut best_larger_index: Option<usize> = None;
        let mut best_larger_len = usize::MAX;

        for i in 0..pool.len() {
            let buffer_len = pool[i].len();
            if buffer_len == size {
                let mut buffer = pool.swap_remove(i);
                buffer.fill(fill_value);
                record_checkout(false, size);
                return buffer;
            }

            if buffer_len > size && buffer_len < best_larger_len {
                best_larger_len = buffer_len;
                best_larger_index = Some(i);
            }
        }

        if let Some(i) = best_larger_index {
            let mut buffer = pool.swap_remove(i);
            // Truncate logical length for caller while preserving allocation capacity.
            // The full capacity is restored in `return_buffer` so the pool does not
            // permanently degrade larger allocations into smaller ones.
            buffer.truncate(size);
            buffer.fill(fill_value);
            record_checkout(false, size);
            return buffer;
        }

        record_checkout(true, size);
        vec![fill_value; size]
    })
}

/// Get a buffer from the pool, creating a new one if necessary.
///
/// Reuses exact-size buffers first, then reuses the smallest larger buffer.
///
/// When a larger allocation is reused for a smaller request, only its logical length
/// is truncated for the caller; underlying capacity is preserved and restored on return.
/// This prevents capacity degradation in mixed-size workloads.
pub fn get_buffer(size: usize) -> Array1<f64> {
    Array1::from_vec(checkout_buffer(size, f64::NAN))
}

/// Return a buffer to the pool for reuse.
///
/// Restores logical length to the vector capacity before storing so a previously
/// large allocation remains available for future large requests.
pub fn return_buffer(buffer: Array1<f64>) {
    let mut raw = buffer.into_raw_vec();
    let target_len = raw.capacity();
    if target_len > raw.len() {
        raw.resize(target_len, f64::NAN);
    }

    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < MAX_POOL_BUFFERS {
            pool.push(raw);
        }
    })
}

/// Get a buffer from the pool, initialized with zeros.
#[cfg(test)]
pub fn get_buffer_zero(size: usize) -> Array1<f64> {
    Array1::from_vec(checkout_buffer(size, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_pool() {
        BUFFER_POOL.with(|pool| pool.borrow_mut().clear());
    }

    #[test]
    fn test_buffer_pool() {
        clear_pool();

        let buf1 = get_buffer(100);
        assert_eq!(buf1.len(), 100);
        return_buffer(buf1);

        let buf2 = get_buffer(100);
        assert_eq!(buf2.len(), 100);
        return_buffer(buf2);
    }

    #[test]
    fn test_buffer_pool_reuse_larger() {
        clear_pool();

        let buf1 = get_buffer(200);
        let original_ptr = buf1.as_ptr();
        assert_eq!(buf1.len(), 200);
        return_buffer(buf1);

        let buf2 = get_buffer(150);
        assert_eq!(buf2.len(), 150);
        assert_eq!(
            original_ptr,
            buf2.as_ptr(),
            "Expected to reuse the larger buffer allocation"
        );
        return_buffer(buf2);

        // Regression guard: after serving a smaller request, the pool should still
        // be able to satisfy the original larger size from the same allocation.
        let buf3 = get_buffer(200);
        assert_eq!(buf3.len(), 200);
        assert_eq!(
            original_ptr,
            buf3.as_ptr(),
            "Pool must preserve larger allocation capacity across mixed-size reuse"
        );
        return_buffer(buf3);
    }

    #[test]
    fn test_buffer_pool_reuse_larger_zero_initialized() {
        clear_pool();

        let mut buf1 = get_buffer(64);
        buf1.fill(42.0);
        return_buffer(buf1);

        let buf2 = get_buffer_zero(32);
        assert_eq!(buf2.len(), 32);
        assert!(buf2.iter().all(|&value| value == 0.0));
        return_buffer(buf2);
    }
}
