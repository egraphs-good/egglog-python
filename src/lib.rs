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

    /// Define a rewrite rule, returning the name of the rule
    #[pyo3(text_signature = "($self, rewrite)")]
    fn add_rewrite(&mut self, rewrite: WrappedRewrite) -> EggResult<String> {
        let res = self.egraph.add_rewrite(rewrite.into())?;
        Ok(res.to_string())
    }

    /// Define a new named value.
    #[pyo3(
        text_signature = "($self, name, expr, cost=None)",
        signature = "(name, expr, cost=None)"
    )]
    fn define(&mut self, name: String, expr: WrappedExpr, cost: Option<usize>) -> EggResult<()> {
        self.egraph.define(name.into(), expr.into(), cost)?;
        Ok(())
    }

    /// Declare a new function definition.
    #[pyo3(text_signature = "($self, decl)")]
    fn declare_function(&mut self, decl: WrappedFunctionDecl) -> EggResult<()> {
        self.egraph.declare_function(&decl.into())?;
        Ok(())
    }

    /// Declare a new sort with the given name.
    #[pyo3(text_signature = "($self, name)")]
    fn declare_sort(&mut self, name: &str) -> EggResult<()> {
        self.egraph.declare_sort(name)?;
        Ok({})
    }

    /// Declare a new datatype constructor.
    #[pyo3(text_signature = "($self, variant, sort)")]
    fn declare_constructor(&mut self, variant: WrappedVariant, sort: &str) -> EggResult<()> {
        self.egraph.declare_constructor(variant.into(), sort)?;
        Ok({})
    }

    /// Parse the input string as a program and run it on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    #[pyo3(text_signature = "($self, input)")]
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
