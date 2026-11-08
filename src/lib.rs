mod error;
mod protobuf;
mod py_object_sort;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Encoded protocol transport with the native Python-object provider installed.
// PyO3 rejects overlapping mutable borrows of an engine instance.
#[pymodule(gil_used = false)]
fn bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<crate::protobuf::ProtoEngine>()?;
    m.add(
        "PrimitiveFailure",
        m.py().get_type::<crate::error::PrimitiveFailure>(),
    )?;
    m.add_function(wrap_pyfunction!(crate::protobuf::_encode_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(crate::protobuf::_decode_scalar, m)?)?;
    Ok(())
}
