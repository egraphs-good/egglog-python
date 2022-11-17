mod conversions;
mod error;
mod utils;

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

    /// Extract the best expression of a given value. Will also return
    // variants number of additional options.
    // #[pyo3(text_signature = "($self, value, variants=0)")]
    // fn extract_expr(
    //     &mut self,
    //     value: WrappedExpr,
    //     variants: usize,
    // ) -> EggResult<(usize, WrappedExpr, Vec<WrappedExpr>)> {
    //     let (cost, expr, exprs) = self.egraph.extract_expr(value.into(), variants)?;
    //     Ok((
    //         cost,
    //         expr.into(),
    //         exprs.into_iter().map(|x| x.into()).collect(),
    //     ))
    // }

    // /// Check that a fact is true in the egraph.
    #[pyo3(text_signature = "($self, fact)")]
    fn check_fact(&mut self, fact: Fact_) -> EggResult<()> {
        self.egraph.check_fact(&fact.into())?;
        Ok({})
    }

    // /// Run the rules on the egraph until it reaches a fixpoint, specifying the max number of iterations.
    // /// Returns a tuple of the total time spen searching, applying, and rebuilding.
    #[pyo3(text_signature = "($self, limit)")]
    fn run_rules(
        &mut self,
        limit: usize,
    ) -> EggResult<(WrappedDuration, WrappedDuration, WrappedDuration)> {
        let [search, apply, rebuild] = self.egraph.run_rules(limit);
        Ok((search.into(), apply.into(), rebuild.into()))
    }

    // /// Define a rewrite rule, returning the name of the rule
    #[pyo3(text_signature = "($self, rewrite)")]
    fn add_rewrite(&mut self, rewrite: Rewrite) -> EggResult<String> {
        let res = self.egraph.add_rewrite(rewrite.into())?;
        Ok(res.to_string())
    }

    // /// Define a new named value.
    #[pyo3(
        text_signature = "($self, name, expr, cost=None)",
        signature = "(name, expr, cost=None)"
    )]
    fn define(&mut self, name: String, expr: Expr, cost: Option<usize>) -> EggResult<()> {
        self.egraph.define(name.into(), expr.into(), cost)?;
        Ok(())
    }

    // /// Declare a new function definition.
    #[pyo3(text_signature = "($self, decl)")]
    fn declare_function(&mut self, decl: FunctionDecl) -> EggResult<()> {
        self.egraph.declare_function(&decl.into())?;
        Ok(())
    }

    // /// Declare a new sort with the given name.
    #[pyo3(text_signature = "($self, name)")]
    fn declare_sort(&mut self, name: &str) -> EggResult<()> {
        self.egraph.declare_sort(name)?;
        Ok({})
    }

    // /// Declare a new datatype constructor.
    #[pyo3(text_signature = "($self, variant, sort)")]
    fn declare_constructor(&mut self, variant: Variant, sort: &str) -> EggResult<()> {
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

    add_structs_to_module(m)?;
    add_enums_to_module(m)?;

    Ok(())
}
