use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod pelt;

#[pyfunction]
#[pyo3(signature = (returns, penalty, min_size, model = "l2"))]
fn detect_change_points_pelt_rs(
    returns: Vec<f64>,
    penalty: f64,
    min_size: usize,
    model: &str,
) -> PyResult<Vec<usize>> {
    pelt::detect_change_points_pelt_rs(&returns, penalty, min_size, model)
        .map_err(PyValueError::new_err)
}

#[pymodule]
fn rust_extensions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_change_points_pelt_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_change_points() {
        let mut data = Vec::new();
        // Segment 1: mean 0.0
        for _ in 0..100 {
            data.push(0.0);
        }
        // Segment 2: mean 5.0
        for _ in 0..100 {
            data.push(5.0);
        }
        
        let penalty = 10.0;
        let bps = detect_change_points_pelt_rs(data, penalty, 10, "l2").unwrap();
        // Should find a change point around 100, and end at 200
        assert_eq!(bps.len(), 2);
        assert_eq!(bps[0], 100);
        assert_eq!(bps[1], 200);
    }

    #[test]
    fn test_detect_change_points_normal_model() {
        let mut data = Vec::new();
        // Segment 1: low variance around mean 0
        for i in 0..100 {
            data.push(if i % 2 == 0 { 0.05 } else { -0.05 });
        }
        // Segment 2: high variance around mean 0
        for i in 0..100 {
            data.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }

        let penalty = 10.0;
        let bps = detect_change_points_pelt_rs(data, penalty, 10, "normal").unwrap();
        assert_eq!(bps.len(), 2);
        assert_eq!(bps[0], 100);
        assert_eq!(bps[1], 200);
    }
}
