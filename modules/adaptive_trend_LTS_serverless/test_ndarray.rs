fn main() {
    let mut a = ndarray::Array1::<f64>::zeros(10);
    let b = a.slice_move(ndarray::s![..5]);
    println!("{}", b.len());
}
