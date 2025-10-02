mod conversions;
mod egraph;
mod error;
mod extract;
mod py_object_sort;
mod serialize;
mod termdag;
mod utils;

use pyo3::prelude::*;

/// Bindings for egglog rust library
#[pymodule]
fn bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configure Rayon thread pool from env var, defaulting to 1 if unset/invalid.
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    pyo3_log::init();

    m.add_class::<crate::py_object_sort::PyObjectSort>()?;
    m.add_class::<crate::serialize::SerializedEGraph>()?;
    m.add_class::<crate::egraph::EGraph>()?;
    m.add_class::<crate::egraph::Value>()?;
    m.add_class::<crate::error::EggSmolError>()?;
    m.add_class::<crate::termdag::TermDag>()?;
    m.add_class::<crate::conversions::UserDefinedCommandOutput>()?;
    m.add_class::<crate::conversions::Function>()?;
    m.add_class::<crate::extract::Extractor>()?;
    m.add_class::<crate::extract::CostModel>()?;
    crate::conversions::add_structs_to_module(m)?;
    crate::conversions::add_enums_to_module(m)?;

    Ok(())
}
