use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
use std::collections::HashMap;

fn get_stats(
    ids: Vec<usize>,
    counts: Option<HashMap<(usize, usize), usize>>,
) -> Result<HashMap<(usize, usize), usize>, ()> {
    let mut counts = counts.unwrap_or(HashMap::new());
    for pair in ids.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
    Ok(counts)
}

fn merge(ids: Vec<usize>, pair: (usize, usize), idx: usize) -> Result<Vec<usize>, ()> {
    let mut newids = vec![];
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == pair.0 && i < ids.len() - 1 && ids[i + 1] == pair.1 {
            newids.push(idx);
            i += 2;
        } else {
            newids.push(ids[i]);
            i += 1;
        }
    }
    Ok(newids)
}

#[pyfunction]
fn train_basic_bpe(
    ids: Vec<usize>,
    num_merges: usize,
) -> PyResult<(HashMap<(usize, usize), usize>, Py<PyDict>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let mut merges = HashMap::new();
    let mut ids = ids;
    let mut vocab = PyDict::new(py);

    for idx in 0..256 {
        let bytes_value = PyBytes::new(py, &[idx as u8])?;
        vocab.set_item(idx, bytes_value)?;
    }

    for i in 0..num_merges {
        let stats = get_stats(ids.clone(), None)?;
        let pair = *stats.iter().max_by_key(|(_, &v)| v).unwrap().0;
        let idx = 256 + i;
        ids = merge(ids, pair, idx)?;
        merges.insert(pair, idx);

        let mut new_vocab = vec![];
        new_vocab.extend_from_slice(
            &vocab
                .get_item(pair.0)
                .unwrap()
                .extract::<PyBytes>(py)?
                .as_bytes(),
        );
        new_vocab.extend_from_slice(
            &vocab
                .get_item(pair.1)
                .unwrap()
                .extract::<PyBytes>(py)?
                .as_bytes(),
        );

        let bytes_value = PyBytes::new(py, &new_vocab)?;
        vocab.set_item(idx, bytes_value)?;
    }

    Ok((merges, vocab.into()))
}
/// A Python module implemented in Rust.
#[pymodule]
fn _minbpe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_stats, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(train_basic_bpe, m)?)?;
    Ok(())
}
