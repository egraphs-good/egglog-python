use pyo3::prelude::*;

// Create exceptions with class instead of create_exception! macro
// https://github.com/PyO3/pyo3/issues/295#issuecomment-852358088
#[pyclass(extends=pyo3::exceptions::PyException)]
pub struct EggSmolError {
    #[pyo3(get)]
    context: String,
}

#[pymethods]
impl EggSmolError {
    #[new]
    fn new(context: String) -> Self {
        EggSmolError { context }
    }
}

// Wrap the egglog::Error so we can automatically convert from it to the PyErr
// and so return it from each function automatically
// https://pyo3.rs/latest/function/error_handling.html#foreign-rust-error-types
// TODO: Create classes for each of these errors
pub enum WrappedError {
    // Add additional context for egglog error
    Egglog(egglog::Error, String),
    ParseError(egglog::ast::ParseError),
    Py(PyErr),
}

// Convert from the WrappedError to the PyErr by creating a new Python error
impl From<WrappedError> for PyErr {
    fn from(error: WrappedError) -> Self {
        match error {
            WrappedError::Egglog(error, str) => {
                PyErr::new::<EggSmolError, _>(error.to_string() + &str)
            }
            WrappedError::Py(error) => error,
            WrappedError::ParseError(error) => PyErr::new::<EggSmolError, _>(error.to_string()),
        }
    }
}

// Convert from an egglog::Error to a WrappedError
impl From<egglog::Error> for WrappedError {
    fn from(other: egglog::Error) -> Self {
        Self::Egglog(other, String::new())
    }
}

impl From<egglog::ast::ParseError> for WrappedError {
    fn from(other: egglog::ast::ParseError) -> Self {
        Self::ParseError(other)
    }
}

// Convert from a PyErr to a WrappedError
impl From<PyErr> for WrappedError {
    fn from(other: PyErr) -> Self {
        Self::Py(other)
    }
}

// Use similar to PyResult, wraps a result type and can be converted to PyResult
pub type EggResult<T> = Result<T, WrappedError>;
