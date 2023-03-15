mod conversions;
mod egraph;
mod error;
mod parse;
mod utils;

use pyo3::prelude::*;

/// Bindings for egg-smol rust library
#[pymodule]
fn bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<crate::egraph::EGraph>()?;
    m.add_class::<crate::error::EggSmolError>()?;
    m.add_function(wrap_pyfunction!(crate::parse::parse, m)?)?;

    crate::conversions::add_structs_to_module(m)?;
    crate::conversions::add_enums_to_module(m)?;

    Ok(())
}
