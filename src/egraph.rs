// Wrapper around EGraph type

use crate::conversions::*;
use crate::error::{EggResult, WrappedError};
use crate::freeze::FrozenEGraph;
use crate::py_object_sort::{PyObjectErrorState, PyObjectSort, PyPickledValue, load};
use crate::serialize::SerializedEGraph;
use crate::termdag::TermDag;
use crate::tracing_otel;

use egglog::prelude::add_base_sort;
use egglog::{RawValues, Read as _, SerializeConfig, span};
use log::info;
use num_rational::{BigRational, Rational64};
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::PathBuf;

fn run_with_py_error<T>(py_error: &PyObjectErrorState, f: impl FnOnce() -> T) -> PyResult<T> {
    py_error.lock().unwrap().take();
    let result = catch_unwind(AssertUnwindSafe(f));
    if let Some(error) = py_error.lock().unwrap().take() {
        return Err(error);
    }
    match result {
        Ok(value) => Ok(value),
        Err(error) => resume_unwind(error),
    }
}

fn error_is_replayable_by_fail(command: &egglog::ast::Command, error: &egglog::Error) -> bool {
    use egglog::Error;
    use egglog::ast::Command;

    matches!(
        (command, error),
        (
            Command::Check(..),
            Error::CheckError(..) | Error::BackendError(_)
        ) | (
            Command::Action(_),
            Error::BackendError(_) | Error::SubsumeMergeError(..)
        ) | (
            Command::RunSchedule(_),
            Error::BackendError(_) | Error::NoSuchRuleset(..) | Error::ParseError(_)
        ) | (
            Command::Extract(..),
            Error::ExtractError(_) | Error::BackendError(_)
        ) | (Command::Pop(..), Error::Pop(_))
            | (Command::Fail(..), Error::ExpectFail(_))
            // Registered commands are resolved and typechecked by their
            // implementation during `run_command`, so any error from a sole
            // user-defined command can be reproduced by wrapping it in `fail`.
            | (Command::UserDefined(..), _)
    )
}

/// EGraph()
/// --
///
/// Create an empty EGraph.
#[pyclass(unsendable)]
pub struct EGraph {
    pub(crate) egraph: egglog::EGraph,
    cmds: Option<String>,
    py_error: PyObjectErrorState,
}

impl EGraph {
    fn run_parsed_commands(
        &mut self,
        py: Python<'_>,
        commands: Vec<egglog::ast::Command>,
        parsed_from_source: bool,
    ) -> EggResult<Vec<CommandOutput>> {
        let failed_command = (commands.len() == 1).then(|| commands[0].clone());
        let cmds_str = commands
            .iter()
            .map(|command| format!("{command}\n"))
            .collect::<String>();
        let span = if parsed_from_source {
            tracing::info_span!(
                "bindings.parse_and_run_program",
                command_count = commands.len(),
                commands = tracing::field::display(cmds_str.trim_end())
            )
        } else {
            tracing::info_span!(
                "bindings.run_program",
                command_count = commands.len(),
                commands = tracing::field::display(cmds_str.trim_end())
            )
        };
        let _entered = span.enter();
        info!("Running commands:\n{}", cmds_str);
        let res = run_with_py_error(&self.py_error, || {
            py.detach(|| self.egraph.run_program(commands))
        })?;
        match res {
            Err(error) => {
                if failed_command
                    .as_ref()
                    .is_some_and(|command| error_is_replayable_by_fail(command, &error))
                {
                    Err(WrappedError::ReplayableEgglog(error))
                } else {
                    Err(WrappedError::Egglog(error))
                }
            }
            Ok(outputs) => {
                if let Some(cmds) = &mut self.cmds {
                    cmds.push_str(&cmds_str);
                }
                Ok(outputs.into_iter().map(|o| o.into()).collect())
            }
        }
    }
}

#[pymethods]
impl EGraph {
    #[new]
    #[pyo3(signature = (*, fact_directory=None, seminaive=true, record=false, num_threads=1, no_decomp=false))]
    fn new(
        fact_directory: Option<PathBuf>,
        seminaive: bool,
        record: bool,
        num_threads: usize,
        no_decomp: bool,
    ) -> Self {
        let mut egraph = egglog_experimental::new_experimental_egraph();
        egraph.fact_directory = fact_directory;
        egraph.seminaive = seminaive;
        egraph.set_num_threads(num_threads);
        egraph.no_decomp = no_decomp;
        let py_error = PyObjectErrorState::default();
        add_base_sort(
            &mut egraph,
            PyObjectSort {
                py_error: py_error.clone(),
            },
            span!(),
        )
        .unwrap();
        Self {
            egraph,
            cmds: record.then(String::new),
            py_error,
        }
    }

    /// Parse a program into a list of commands.
    #[pyo3(signature = (input, /, filename=None))]
    fn parse_program(&mut self, input: &str, filename: Option<String>) -> EggResult<Vec<Command>> {
        let commands = self
            .egraph
            .parser
            .get_program_from_string(filename, input)?;
        Ok(commands.into_iter().map(|x| x.into()).collect())
    }

    /// Parse a program and immediately run the parsed commands on the EGraph.
    #[pyo3(signature = (input, /, filename=None, traceparent=None, tracestate=None))]
    fn parse_and_run_program(
        &mut self,
        py: Python<'_>,
        input: &str,
        filename: Option<String>,
        traceparent: Option<String>,
        tracestate: Option<String>,
    ) -> EggResult<Vec<CommandOutput>> {
        let _context_guard =
            tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
        let commands = self
            .egraph
            .parser
            .get_program_from_string(filename, input)?;
        self.run_parsed_commands(py, commands, true)
    }

    /// Run a series of commands on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    #[pyo3(signature=(*commands, traceparent=None, tracestate=None))]
    fn run_program(
        &mut self,
        py: Python<'_>,
        commands: Vec<Command>,
        traceparent: Option<String>,
        tracestate: Option<String>,
    ) -> EggResult<Vec<CommandOutput>> {
        let _context_guard =
            tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
        let commands: Vec<egglog::ast::Command> = commands.into_iter().map(|x| x.into()).collect();
        self.run_parsed_commands(py, commands, false)
    }

    /// Returns the text of successfully run commands when recording is enabled.
    fn commands(&self) -> Option<String> {
        self.cmds.clone()
    }

    /// Return the number of worker threads configured for this EGraph.
    fn num_threads(&self) -> usize {
        self.egraph.num_threads()
    }

    /// Set the number of worker threads used by this EGraph.
    fn set_num_threads(&mut self, num_threads: usize) {
        self.egraph.set_num_threads(num_threads);
    }

    /// Return whether rule decomposition is disabled globally for this EGraph.
    fn no_decomp(&self) -> bool {
        self.egraph.no_decomp
    }

    /// Set whether rule decomposition is disabled for subsequently registered rules.
    fn set_no_decomp(&mut self, no_decomp: bool) {
        self.egraph.no_decomp = no_decomp;
    }

    /// Serialize the EGraph to a SerializedEGraph object.
    #[pyo3(
        signature = (root_eclasses, *, max_functions=None, max_calls_per_function=None, include_temporary_functions=false, traceparent=None, tracestate=None),
        text_signature = "(self, root_eclasses, *, max_functions=None, max_calls_per_function=None, include_temporary_functions=False, traceparent=None, tracestate=None)"
    )]
    fn serialize(
        &mut self,
        root_eclasses: Vec<Expr>,
        max_functions: Option<usize>,
        max_calls_per_function: Option<usize>,
        include_temporary_functions: bool,
        traceparent: Option<String>,
        tracestate: Option<String>,
    ) -> EggResult<SerializedEGraph> {
        let _context_guard =
            tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
        let span = tracing::info_span!(
            "bindings.serialize",
            root_eclass_count = root_eclasses.len()
        );
        let _entered = span.enter();
        let res = Python::attach(|py| {
            run_with_py_error(&self.py_error, || {
                py.detach(|| -> Result<_, egglog::Error> {
                    let root_eclasses: Vec<_> = root_eclasses
                        .into_iter()
                        .map(|x| self.egraph.eval_expr(&egglog::ast::Expr::from(x)))
                        .collect::<Result<_, _>>()?;
                    Ok(self.egraph.serialize(SerializeConfig {
                        max_functions,
                        max_calls_per_function,
                        include_temporary_functions,
                        root_eclasses,
                    }))
                })
            })
        })??;
        Ok(SerializedEGraph {
            egraph: res.egraph,
            truncated_functions: res.truncated_functions,
            discarded_functions: res.discarded_functions,
        })
    }

    fn set_report_level(&mut self, level: ReportLevel) {
        self.egraph.set_report_level(level.into());
    }

    fn lookup_function(&self, name: &str, key: Vec<Value>) -> EggResult<Option<Value>> {
        let is_constructor = self.egraph.get_function(name).is_some_and(|function| {
            function.func_type().subtype == egglog::ast::FunctionSubtype::Constructor
        });
        let value = self.egraph.read(|state| {
            let key = RawValues(key.into_iter().map(|value| value.0).collect());
            if is_constructor {
                state.eclass_of(name, key)
            } else {
                state.lookup(name, key)
            }
        })?;
        Ok(value.map(Value))
    }

    /// Extract `value` using its runtime sort. The value must come from `eval_expr`
    /// on this e-graph, and `sort` must be the sort returned with it. Values from
    /// another e-graph or paired with another existing sort are unsupported.
    fn extract_value(&self, value: Value, sort: &str) -> EggResult<(TermDag, usize, u64)> {
        let sort = self.egraph.get_sort_by_name(sort).ok_or_else(|| {
            WrappedError::Egglog(egglog::TypeError::UndefinedSort(sort.to_owned(), span!()).into())
        })?;
        let (termdag, term, cost) = self.egraph.extract_value(sort, value.0)?;
        Ok((TermDag(termdag), term, cost))
    }

    #[pyo3(signature = (expr, *, traceparent=None, tracestate=None))]
    fn eval_expr(
        &mut self,
        py: Python<'_>,
        expr: Expr,
        traceparent: Option<String>,
        tracestate: Option<String>,
    ) -> EggResult<(String, Value)> {
        let _context_guard =
            tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
        let span = tracing::info_span!("bindings.eval_expr");
        let _entered = span.enter();
        let expr: egglog::ast::Expr = expr.into();
        let res = run_with_py_error(&self.py_error, || {
            py.detach(|| self.egraph.eval_expr(&expr))
        })??;
        Ok((res.0.name().to_string(), Value(res.1)))
    }

    fn value_to_i64(&self, v: Value) -> i64 {
        self.egraph.value_to_base(v.0)
    }

    fn value_to_bigint<'py>(&self, py: Python<'py>, v: Value) -> PyResult<Bound<'py, PyAny>> {
        let bi: egglog::sort::Z = self.egraph.value_to_base(v.0);
        Ok(bi.0.into_pyobject(py)?.into_any())
    }

    fn value_to_bigrat(&self, v: Value) -> BigRational {
        let bi: egglog::sort::Q = self.egraph.value_to_base(v.0);
        bi.0
    }

    fn value_to_f64(&self, v: Value) -> f64 {
        let f: egglog::sort::F = self.egraph.value_to_base(v.0);
        f.0.into_inner()
    }

    fn value_to_string(&self, v: Value) -> String {
        let s: egglog::sort::S = self.egraph.value_to_base(v.0);
        s.0
    }

    fn value_to_bool(&self, v: Value) -> bool {
        self.egraph.value_to_base(v.0)
    }
    fn value_to_rational(&self, v: Value) -> Rational64 {
        let r: egglog_experimental::R = self.egraph.value_to_base(v.0);
        r.0
    }

    fn value_to_pyobject<'a>(&self, py: Python<'a>, v: Value) -> PyResult<Bound<'a, PyAny>> {
        let ident = self.egraph.value_to_base::<PyPickledValue>(v.0);
        load(py, &ident)
    }

    fn value_to_map(&self, v: Value) -> BTreeMap<Value, Value> {
        let mc = self
            .egraph
            .value_to_container::<egglog::sort::MapContainer>(v.0)
            .unwrap();
        mc.data
            .iter()
            .map(|(k, v)| (Value(*k), Value(*v)))
            .collect()
    }

    fn value_to_multiset(&self, v: Value) -> Vec<Value> {
        let mc = self
            .egraph
            .value_to_container::<egglog::sort::MultiSetContainer>(v.0)
            .unwrap();
        mc.data.iter().map(|k| Value(*k)).collect()
    }

    fn value_to_set(&self, v: Value) -> BTreeSet<Value> {
        let sc = self
            .egraph
            .value_to_container::<egglog::sort::SetContainer>(v.0)
            .unwrap();
        sc.data.iter().map(|k| Value(*k)).collect()
    }

    fn value_to_vec(&self, v: Value) -> Vec<Value> {
        let vc = self
            .egraph
            .value_to_container::<egglog::sort::VecContainer>(v.0)
            .unwrap();
        vc.data.iter().map(|x| Value(*x)).collect()
    }

    fn value_to_function(&self, v: Value) -> (String, Vec<Value>) {
        let fc = self
            .egraph
            .value_to_container::<egglog::sort::FunctionContainer>(v.0)
            .unwrap();
        (
            fc.2.clone(),
            fc.1.iter().map(|(_, v)| Value(*v)).collect::<Vec<_>>(),
        )
    }

    fn freeze(&self) -> FrozenEGraph {
        FrozenEGraph::from_egraph(&self.egraph)
    }
}

/// Wrapper around Egglog Value. Represents either a primitive base value or a reference to an e-class.
#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Clone)]
#[pyclass(eq, frozen, ord, hash, str = "{0:?}")]
pub struct Value(pub egglog::Value);

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;

    #[test]
    fn python_error_from_worker_precedes_backend_panic() {
        Python::initialize();
        let error = PyObjectErrorState::default();
        *error.lock().unwrap() = Some(PyValueError::new_err("stale"));

        let worker_error = error.clone();
        let result = run_with_py_error(&error, || {
            assert!(worker_error.lock().unwrap().is_none());
            std::thread::spawn(move || {
                *worker_error.lock().unwrap() = Some(PyValueError::new_err("worker boom"));
            })
            .join()
            .unwrap();
            panic!("secondary backend panic");
        });

        let captured = result.unwrap_err();
        assert!(error.lock().unwrap().is_none());
        Python::attach(|py| {
            assert!(captured.is_instance_of::<PyValueError>(py));
            assert_eq!(captured.value(py).to_string(), "worker boom");
        });
    }
}
