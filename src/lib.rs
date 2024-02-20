use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
fn get_stats(
    ids: Vec<usize>,
    counts: Option<HashMap<(usize, usize), usize>>,
) -> PyResult<HashMap<(usize, usize), usize>> {
    let mut counts = counts.unwrap_or(HashMap::new());
    for pair in ids.windows(2) {
        // rustic way to iterate consecutive elements ;)
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
    Ok(counts)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _minbpe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_stats, m)?)?;
    Ok(())
}
