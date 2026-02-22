#[cfg(test)]
mod tests {
    use ndarray::{s, Array1};

    #[test]
    fn test_ndarray_slice() {
        let a = Array1::<f64>::zeros(10);
        let b = a.slice_move(s![..5]);
        assert_eq!(b.len(), 5);
    }
}
