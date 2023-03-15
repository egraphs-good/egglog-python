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

type ParseError<'a> = lalrpop_util::ParseError<usize, lalrpop_util::lexer::Token<'a>, String>;

// Wrap the egg_smol::Error so we can automatically convert from it to the PyErr
// and so return it from each function automatically
// https://pyo3.rs/latest/function/error_handling.html#foreign-rust-error-types
// TODO: Create classes for each of these errors
pub enum WrappedError {
    EggSmol(egg_smol::Error),
    Parse(String),
}

// Convert from the WrappedError to the PyErr by creating a new Python error
impl From<WrappedError> for PyErr {
    fn from(error: WrappedError) -> Self {
        match error {
            WrappedError::EggSmol(e) => PyErr::new::<EggSmolError, _>(e.to_string()),
            WrappedError::Parse(e) => PyErr::new::<EggSmolError, _>(e),
        }
    }
}

// Convert from an egg_smol::Error to a WrappedError
impl From<egg_smol::Error> for WrappedError {
    fn from(other: egg_smol::Error) -> Self {
        Self::EggSmol(other)
    }
}

// Convert from a parse error to a WrappedError
impl From<ParseError<'_>> for WrappedError {
    fn from(other: ParseError<'_>) -> Self {
        Self::Parse(other.to_string())
    }
}

// Use similar to PyResult, wraps a result type and can be converted to PyResult
pub type EggResult<T> = Result<T, WrappedError>;
