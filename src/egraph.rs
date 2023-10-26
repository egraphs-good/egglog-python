// Wrapper around EGraph type

use crate::conversions::*;
use crate::error::EggResult;
use crate::py_object_sort::PyObjectSort;

use egglog::sort::Sort;
use egglog::SerializeConfig;
use log::info;
use pyo3::{prelude::*, PyTraverseError, PyVisit};
use std::path::PathBuf;
use std::sync::Arc;

/// EGraph()
/// --
///
/// Create an empty EGraph.
#[pyclass(
    unsendable,
    text_signature = "(*, fact_directory=None, seminaive=True, terms_encoding=False)"
)]
pub struct EGraph {
    egraph: egglog::EGraph,
    py_object_arcsort: Arc<PyObjectSort>,
}

#[pymethods]
impl EGraph {
    #[new]
    #[pyo3(signature = (*, fact_directory=None, seminaive=true, terms_encoding=false))]
    fn new(fact_directory: Option<PathBuf>, seminaive: bool, terms_encoding: bool) -> Self {
        let mut egraph = egglog::EGraph::default();
        egraph.fact_directory = fact_directory;
        egraph.seminaive = seminaive;
        if terms_encoding {
            egraph.enable_terms_encoding();
        }
        let py_object_arcsort = Arc::new(PyObjectSort::new("PyObject".into()));
        egraph.add_arcsort(py_object_arcsort.clone()).unwrap();
        Self {
            egraph,
            py_object_arcsort,
        }
    }

    /// Parse a program into a list of commands.
    #[pyo3(signature = (input, /))]
    fn parse_program(&mut self, input: &str) -> EggResult<Vec<Command>> {
        info!("Parsing program");
        let commands = self.egraph.parse_program(input)?;
        Ok(commands.into_iter().map(|x| x.into()).collect())
    }

    /// Run a series of commands on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    #[pyo3(signature=(*commands))]
    fn run_program(&mut self, commands: Vec<Command>) -> EggResult<Vec<String>> {
        let commands: Vec<egglog::ast::Command> = commands.into_iter().map(|x| x.into()).collect();
        info!("Running commands {:?}", commands);
        let res = self.egraph.run_program(commands)?;
        Ok(res)
    }

    /// Gets the last expressions extracted from the EGraph, if the last command
    /// was a Simplify or Extract command.
    #[pyo3(signature = ())]
    fn extract_report(&mut self) -> Option<ExtractReport> {
        info!("Getting last extract report");
        self.egraph
            .get_extract_report()
            .as_ref()
            .map(|report| report.into())
    }

    /// Gets the last run report from the EGraph, if the last command
    /// was a run or simplify command.
    #[pyo3(signature = ())]
    fn run_report(&mut self) -> Option<RunReport> {
        info!("Getting last run report");
        self.egraph
            .get_run_report()
            .as_ref()
            .map(|report| report.into())
    }

    /// Returns the EGraph as graphviz string.
    #[pyo3(
        signature = (*, max_functions=None, max_calls_per_function=None, n_inline_leaves=0, split_primitive_outputs=false),
        text_signature = "(self, *, max_functions=None, max_calls_per_function=None, n_inline_leaves=0, split_primitive_outputs=False)"
    )]
    fn to_graphviz_string(
        &self,
        max_functions: Option<usize>,
        max_calls_per_function: Option<usize>,
        n_inline_leaves: usize,
        split_primitive_outputs: bool,
    ) -> String {
        info!("Getting graphviz");
        // TODO: Expose full serialized e-graph in the future
        let mut serialized = self.egraph.serialize(SerializeConfig {
            max_functions,
            max_calls_per_function,
            include_temporary_functions: false,
            split_primitive_outputs,
        });
        for _ in 0..n_inline_leaves {
            serialized.inline_leaves();
        }
        info!("Serialized egraph: {:?}", serialized);
        serialized.to_dot()
    }

    /// Register a Python object with the EGraph and return the Expr which represents it.
    #[pyo3(signature = (obj, /))]
    fn save_object(&mut self, obj: PyObject) -> EggResult<Expr> {
        info!("Adding Python object {:?}", obj);
        let value = self.py_object_arcsort.store(obj);
        let expr = self.py_object_arcsort.make_expr(&self.egraph, value).1;
        Ok(expr.into())
    }

    /// Retrieve a Python object from the EGraph.
    #[pyo3(signature = (expr, /))]
    fn load_object(&mut self, expr: Expr) -> EggResult<PyObject> {
        let expr: egglog::ast::Expr = expr.into();
        info!("Loading Python object {:?}", expr);
        let (_, value) =
            self.egraph
                .eval_expr(&expr, Some(self.py_object_arcsort.clone()), false)?;
        let (_, obj) = self.py_object_arcsort.load(&value);
        Ok(obj)
    }

    // Integrate with Python garbage collector
    // https://pyo3.rs/main/class/protocols#garbage-collector-integration

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.py_object_arcsort
            .objects
            .lock()
            .unwrap()
            .values()
            .try_for_each(|obj| {
                visit.call(obj)?;
                Ok(())
            })
    }

    fn __clear__(&mut self) {
        self.py_object_arcsort.objects.lock().unwrap().clear();
    }
}
