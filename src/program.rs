use crate::conversions::Command;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A versioned program owned by Rust, without conversion through Python AST wrappers.
#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub struct Program(pub(crate) egglog::program::Program);

/// Ordered submitted commands and their outcomes; failures may have partial effects.
#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub struct CommandRecord(pub(crate) egglog::program::CommandRecord);

#[pymethods]
impl CommandRecord {
    fn to_json(&self) -> PyResult<String> {
        self.0
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn program(&self) -> PyResult<Program> {
        self.0
            .program()
            .map(Program)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl Program {
    #[new]
    #[pyo3(signature = (*commands))]
    fn new(commands: Vec<Command>) -> PyResult<Self> {
        egglog::program::Program::new(commands.into_iter().map(Into::into).collect())
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (source, /, filename=None))]
    fn parse(source: &str, filename: Option<String>) -> PyResult<Self> {
        egglog::program::Program::parse(filename, source)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        egglog::program::Program::from_json(json)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        self.0
            .to_json()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Generate the JSON Schema from the same Rust types used for execution.
    #[staticmethod]
    fn json_schema() -> PyResult<String> {
        serde_json::to_string_pretty(&egglog::program::Program::schema())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Render diagnostic source; use to_replayable_egglog for checked source export.
    fn to_egglog(&self) -> String {
        self.0.to_egglog()
    }

    fn to_replayable_egglog(&self) -> PyResult<String> {
        self.0
            .to_replayable_egglog()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}
