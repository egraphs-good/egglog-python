use pyo3::prelude::*;

/// EGraph()
/// --
///
/// Create an empty EGraph.
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

// Convert a Python Variant object into a rust variable, by getting the attributes
fn get_variant(obj: &PyAny) -> PyResult<egg_smol::ast::Variant> {
    // TODO: Is there a way to do this more automatically?
    Ok(egg_smol::ast::Variant {
        name: obj
            .getattr(pyo3::intern!(obj.py(), "name"))?
            .extract::<String>()?
            .into(),
        cost: obj.getattr(pyo3::intern!(obj.py(), "cost"))?.extract()?,
        types: obj
            .getattr(pyo3::intern!(obj.py(), "types"))?
            .extract::<Vec<String>>()?
            .into_iter()
            .map(|x| x.into())
            .collect(),
    })
}

#[pymethods]
impl EGraph {
    #[new]
    fn new() -> Self {
        Self {
            egraph: egg_smol::EGraph::default(),
        }
    }

    /// declare_sort($self, name)
    /// --
    ///
    /// Declare a new sort with the given name.
    fn declare_sort(&mut self, name: &str) -> PyResult<()> {
        // TODO: Should the name be a symbol? If so, how should we expose that
        // to Python?
        self.egraph
            .declare_sort(name)
            .map_err(|e| PyErr::new::<EggSmolError, _>(e.to_string()))
    }

    fn declare_constructor(
        &mut self,
        #[pyo3(from_py_with = "get_variant")] variant: egg_smol::ast::Variant,
        sort: &str,
    ) -> PyResult<()> {
        self.egraph
            .declare_constructor(variant, sort)
            .map_err(|e| PyErr::new::<EggSmolError, _>(e.to_string()))
    }

    /// parse_and_run_program($self, input)
    /// --
    ///
    /// Parse the input string as a program and run it on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    fn parse_and_run_program(&mut self, input: &str) -> PyResult<Vec<String>> {
        self.egraph
            .parse_and_run_program(input)
            .map_err(|e| PyErr::new::<EggSmolError, _>(e.to_string()))
    }
}

/// Bindings for egg-smol rust library
#[pymodule]
fn bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EGraph>()?;
    m.add_class::<EggSmolError>()?;
    Ok(())
}
