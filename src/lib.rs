use pyo3::prelude::*;

#[pyclass(unsendable)]
struct EGraph {
    egraph: egg_smol::EGraph,
}

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
#[pymethods]
impl EGraph {
    #[new]
    fn new() -> Self {
        Self {
            egraph: egg_smol::EGraph::default(),
        }
    }

    fn parse_and_run_program(&mut self, program: &str) -> PyResult<Vec<String>> {
        self.egraph
            .parse_and_run_program(program)
            .map_err(|e| PyErr::new::<EggSmolError, _>(e.to_string()))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn bindings(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EGraph>()?;
    m.add_class::<EggSmolError>()?;
    Ok(())
}
