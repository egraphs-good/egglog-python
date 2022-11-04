mod error;
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
    fn declare_constructor(
        &mut self,
        #[pyo3(from_py_with = "get_variant")] variant: egg_smol::ast::Variant,
        sort: &str,
    ) -> EggResult<()> {
        self.egraph.declare_constructor(variant, sort)?;
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
