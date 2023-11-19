// Wrapper around EGraph type

use crate::conversions::*;
use crate::error::EggResult;
use crate::py_object_sort::{ArcPyObjectSort, MyPyObject, PyObjectSort};
use crate::serialize::SerializedEGraph;

use egglog::sort::{BoolSort, F64Sort, I64Sort, StringSort};
use egglog::SerializeConfig;
use log::info;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

/// EGraph()
/// --
///
/// Create an empty EGraph.
#[pyclass(
    unsendable,
    text_signature = "(py_object_sort=None, *, fact_directory=None, seminaive=True, terms_encoding=False)"
)]
pub struct EGraph {
    egraph: egglog::EGraph,
    py_object_arcsort: Option<Arc<PyObjectSort>>,
}

#[pymethods]
impl EGraph {
    #[new]
    #[pyo3(signature = (py_object_sort=None, *, fact_directory=None, seminaive=true, terms_encoding=false))]
    fn new(
        py_object_sort: Option<ArcPyObjectSort>,
        fact_directory: Option<PathBuf>,
        seminaive: bool,
        terms_encoding: bool,
    ) -> Self {
        let mut egraph = egglog::EGraph::default();
        egraph.fact_directory = fact_directory;
        egraph.seminaive = seminaive;
        if terms_encoding {
            egraph.enable_terms_encoding();
        }
        let py_object_arcsort = if let Some(py_object_sort) = py_object_sort {
            egraph.add_arcsort(py_object_sort.0.clone()).unwrap();
            Some(py_object_sort.0)
        } else {
            None
        };
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

    /// Serialize the EGraph to a SerializedEGraph object.
    #[pyo3(
        signature = (*, max_functions=None, max_calls_per_function=None, include_temporary_functions=false, split_primitive_outputs=false),
        text_signature = "(self, *, max_functions=None, max_calls_per_function=None, include_temporary_functions=False, split_primitive_outputs=False)"
    )]
    fn serialize(
        &self,
        max_functions: Option<usize>,
        max_calls_per_function: Option<usize>,
        include_temporary_functions: bool,
        split_primitive_outputs: bool,
    ) -> SerializedEGraph {
        SerializedEGraph {
            egraph: self.egraph.serialize(SerializeConfig {
                max_functions,
                max_calls_per_function,
                include_temporary_functions,
                split_primitive_outputs,
            }),
        }
    }

    /// Evaluates an expression in the EGraph and returns the result as a Python object.
    #[pyo3(signature = (expr, /))]
    fn eval_py_object(&mut self, expr: Expr) -> EggResult<MyPyObject> {
        self.eval_sort(expr, self.py_object_arcsort.clone().unwrap())
    }
    #[pyo3(signature = (expr, /))]
    fn eval_i64(&mut self, expr: Expr) -> EggResult<i64> {
        self.eval_sort(expr, Arc::new(I64Sort::new("i64".into())))
    }

    #[pyo3(signature = (expr, /))]
    fn eval_f64(&mut self, expr: Expr) -> EggResult<f64> {
        self.eval_sort(expr, Arc::new(F64Sort::new("f64".into())))
    }

    #[pyo3(signature = (expr, /))]
    fn eval_string(&mut self, expr: Expr) -> EggResult<String> {
        let s: egglog::ast::Symbol =
            self.eval_sort(expr, Arc::new(StringSort::new("String".into())))?;
        Ok(s.to_string())
    }

    #[pyo3(signature = (expr, /))]
    fn eval_bool(&mut self, expr: Expr) -> EggResult<bool> {
        self.eval_sort(expr, Arc::new(BoolSort::new("bool".into())))
    }

    #[pyo3(signature = (expr, /))]
    fn eval_rational(&mut self, _py: Python<'_>, expr: Expr) -> EggResult<PyObject> {
        // Need to get actual sort for rational, this hack doesnt work.
        // todo!();
        // For rational we need the actual sort on the e-graph, because it contains state
        // There isn't a public way to get a sort right now, so until there is, we use a hack where we create
        // a dummy expression of that sort, and use eval_expr to get the sort
        let one = egglog::ast::Expr::Lit(egglog::ast::Literal::Int(1));
        let arcsort = self
            .egraph
            .eval_expr(
                &egglog::ast::Expr::Call("rational".into(), vec![one.clone(), one]),
                None,
                false,
            )
            .unwrap()
            .0;
        let expr: egglog::ast::Expr = expr.into();
        let (_, _value) = self.egraph.eval_expr(&expr, Some(arcsort.clone()), false)?;
        // Need to get actual sort for rational, this hack doesnt work.
        todo!();
        // let r = num_rational::Rational64::load(&arcsort, &value);

        // // let r: num_rational::Rational64 =
        // //     self.eval_sort(expr, Arc::downcast::<RationalSort>(arcsort).unwrap())?;
        // let frac = py.import("fractions")?;
        // let f = frac.call_method(
        //     "Fraction",
        //     (r.numer().into_py(py), r.denom().into_py(py)),
        //     None,
        // )?;
        // Ok(f.into())
    }
}

impl EGraph {
    fn eval_sort<T: egglog::sort::Sort, V: egglog::sort::FromSort<Sort = T>>(
        &mut self,
        expr: Expr,
        arcsort: Arc<T>,
    ) -> EggResult<V> {
        let expr: egglog::ast::Expr = expr.into();
        let (_, value) = self.egraph.eval_expr(&expr, Some(arcsort.clone()), false)?;
        Ok(V::load(&arcsort, &value))
    }
}
