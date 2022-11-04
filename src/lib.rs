mod conversions;
mod error;
use conversions::*;
use error::*;
use pyo3::prelude::*;

/// EGraph()
/// --
///
/// Create an empty EGraph.
#[pyclass(unsendable)]
struct EGraph {
    egraph: egg_smol::EGraph,
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
    fn declare_sort(&mut self, name: &str) -> EggResult<()> {
        // TODO: Should the name be a symbol? If so, how should we expose that
        // to Python?
        self.egraph.declare_sort(name)?;
        Ok({})
    }

    /// declare_constructor($self, variant, sort)
    /// --
    ///
    /// Declare a new datatype constructor.
    fn declare_constructor(&mut self, variant: WrappedVariant, sort: &str) -> EggResult<()> {
        self.egraph.declare_constructor(variant.into(), sort)?;
        Ok({})
    }

    /// parse_and_run_program($self, input)
    /// --
    ///
    /// Parse the input string as a program and run it on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    fn parse_and_run_program(&mut self, input: &str) -> EggResult<Vec<String>> {
        let res = self.egraph.parse_and_run_program(input)?;
        Ok(res)
    }
}

/// Bindings for egg-smol rust library
#[pymodule]
fn bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EGraph>()?;
    m.add_class::<EggSmolError>()?;
    Ok(())
}
