mod conversions;
mod egraph;
mod error;
mod extract;
mod freeze;
mod py_object_sort;
mod serialize;
mod termdag;
mod tracing_otel;
mod utils;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
#[pyo3(signature = (*, exporter, endpoint=None))]
fn setup_tracing(py: Python<'_>, exporter: &str, endpoint: Option<&str>) -> PyResult<()> {
    let exporter = exporter.to_string();
    let endpoint = endpoint.map(str::to_string);
    py.detach(move || crate::tracing_otel::setup_tracing(&exporter, endpoint.as_deref()))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pyfunction]
fn shutdown_tracing(py: Python<'_>) -> PyResult<()> {
    py.detach(crate::tracing_otel::shutdown_tracing)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

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

    m.add_class::<crate::serialize::SerializedEGraph>()?;
    m.add_class::<crate::egraph::EGraph>()?;
    m.add_class::<crate::egraph::Value>()?;
    m.add_class::<crate::error::EggSmolError>()?;
    m.add_class::<crate::termdag::TermDag>()?;
    m.add_class::<crate::conversions::UserDefinedCommandOutput>()?;
    m.add_class::<crate::conversions::Function>()?;
    m.add_class::<crate::extract::Extractor>()?;
    m.add_class::<crate::extract::CostModel>()?;
    m.add_class::<crate::freeze::FrozenRow>()?;
    m.add_class::<crate::freeze::FrozenFunction>()?;
    m.add_class::<crate::freeze::FrozenEGraph>()?;
    m.add_function(wrap_pyfunction!(setup_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_tracing, m)?)?;
    crate::conversions::add_structs_to_module(m)?;
    crate::conversions::add_enums_to_module(m)?;

    Ok(())
}
