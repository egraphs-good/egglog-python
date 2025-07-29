mod conversions;
mod egraph;
mod error;
mod py_object_sort;
mod serialize;
mod termdag;
mod utils;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;

/// Bindings for egglog rust library
#[pymodule]
fn bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    pyo3_log::init();

    m.add_class::<crate::py_object_sort::PyObjectSort>()?;
    m.add_class::<crate::serialize::SerializedEGraph>()?;
    m.add_class::<crate::egraph::EGraph>()?;
    m.add_class::<crate::error::EggSmolError>()?;
    m.add_class::<crate::termdag::TermDag>()?;

    crate::conversions::add_structs_to_module(m)?;
    crate::conversions::add_enums_to_module(m)?;

    Ok(())
}
