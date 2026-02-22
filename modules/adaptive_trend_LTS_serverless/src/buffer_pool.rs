use ndarray::Array1;
use std::cell::RefCell;

thread_local! {
    static BUFFER_POOL: RefCell<Vec<Array1<f64>>> = RefCell::new(Vec::with_capacity(16));
}

/// Get a buffer from the pool, creating a new one if necessary.
pub fn get_buffer(size: usize) -> Array1<f64> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for i in 0..pool.len() {
            if pool[i].len() >= size {
                let mut buf = pool.swap_remove(i);
                if buf.len() > size {
                    buf = buf.slice_move(ndarray::s![..size]);
                }
                buf.fill(f64::NAN);
                return buf;
            }
        }
        Array1::from_elem(size, f64::NAN)
    })
}

/// Return a buffer to the pool for reuse.
pub fn return_buffer(buffer: Array1<f64>) {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 16 {
            pool.push(buffer);
        }
    })
}

/// Get a buffer from the pool, initialized with zeros.
pub fn get_buffer_zero(size: usize) -> Array1<f64> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for i in 0..pool.len() {
            if pool[i].len() >= size {
                let mut buf = pool.swap_remove(i);
                if buf.len() > size {
                    buf = buf.slice_move(ndarray::s![..size]);
                }
                buf.fill(0.0);
                return buf;
            }
        }
        Array1::zeros(size)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool() {
        let buf1 = get_buffer(100);
        assert_eq!(buf1.len(), 100);
        return_buffer(buf1);

        let buf2 = get_buffer(100);
        assert_eq!(buf2.len(), 100);
        return_buffer(buf2);
    }

    #[test]
    fn test_buffer_pool_reuse_larger() {
        let buf1 = get_buffer(200);
        assert_eq!(buf1.len(), 200);
        return_buffer(buf1);

        let buf2 = get_buffer(150);
        assert_eq!(buf2.len(), 150);
        return_buffer(buf2);
    }
}
