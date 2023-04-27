mod conversions;
mod egraph;
mod error;
mod utils;

use pyo3::prelude::*;

/// Bindings for egg-smol rust library
#[pymodule]
fn bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<crate::egraph::EGraph>()?;
    m.add_class::<crate::error::EggSmolError>()?;
    m.add("HIGH_COST", egg_smol::HIGH_COST)?;

    crate::conversions::add_structs_to_module(m)?;
    crate::conversions::add_enums_to_module(m)?;

    Ok(())
}
