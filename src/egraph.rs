// Wrapper around EGraph type

use crate::conversions::*;
use crate::error::{EggResult, WrappedError};
use crate::py_object_sort::{PyObjectIdent, PyObjectSort};
use crate::serialize::SerializedEGraph;

use egglog::prelude::add_base_sort;
use egglog::{SerializeConfig, span};
use log::info;
use num_bigint::BigInt;
use num_rational::{BigRational, Rational64};
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// EGraph()
/// --
///
/// Create an empty EGraph.
#[pyclass(unsendable)]
pub struct EGraph {
    pub(crate) egraph: egglog::EGraph,
    cmds: Option<String>,
}

#[pymethods]
impl EGraph {
    #[new]
    #[pyo3(
        signature = (py_object_sort=None, *, fact_directory=None, seminaive=true, record=false),
        text_signature = "(py_object_sort=None, *, fact_directory=None, seminaive=True, record=False)"
    )]
    fn new(
        py_object_sort: Option<PyObjectSort>,
        fact_directory: Option<PathBuf>,
        seminaive: bool,
        record: bool,
    ) -> Self {
        let mut egraph = egglog_experimental::new_experimental_egraph();
        egraph.fact_directory = fact_directory;
        egraph.seminaive = seminaive;
        if let Some(py_object_sort) = py_object_sort {
            add_base_sort(&mut egraph, py_object_sort, span!()).unwrap();
        }
        Self {
            egraph,
            cmds: if record { Some(String::new()) } else { None },
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

    /// Run a series of commands on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    #[pyo3(signature=(*commands))]
    fn run_program(
        &mut self,
        py: Python<'_>,
        commands: Vec<Command>,
    ) -> EggResult<Vec<CommandOutput>> {
        let commands: Vec<egglog::ast::Command> = commands.into_iter().map(|x| x.into()).collect();
        let mut cmds_str = String::new();
        for cmd in &commands {
            cmds_str = cmds_str + &cmd.to_string() + "\n";
        }
        info!("Running commands:\n{}", cmds_str);
        let res = py.detach(|| {
            self.egraph.run_program(commands).map_err(|e| {
                WrappedError::Egglog(e, "\nWhen running commands:\n".to_string() + &cmds_str)
            })
        });
        if res.is_ok()
            && let Some(cmds) = &mut self.cmds
        {
            cmds.push_str(&cmds_str);
        }
        res.map(|xs| xs.iter().map(|o| o.into()).collect())
    }

    /// Returns the text of the commands that have been run so far, if `record` was passed.
    #[pyo3(signature = ())]
    fn commands(&self) -> Option<String> {
        self.cmds.clone()
    }

    /// Serialize the EGraph to a SerializedEGraph object.
    #[pyo3(
        signature = (root_eclasses, *, max_functions=None, max_calls_per_function=None, include_temporary_functions=false),
        text_signature = "(self, root_eclasses, *, max_functions=None, max_calls_per_function=None, include_temporary_functions=False)"
    )]
    fn serialize(
        &mut self,
        py: Python<'_>,
        root_eclasses: Vec<Expr>,
        max_functions: Option<usize>,
        max_calls_per_function: Option<usize>,
        include_temporary_functions: bool,
    ) -> SerializedEGraph {
        py.detach(|| {
            let root_eclasses: Vec<_> = root_eclasses
                .into_iter()
                .map(|x| self.egraph.eval_expr(&egglog::ast::Expr::from(x)).unwrap())
                .collect();
            let res = self.egraph.serialize(SerializeConfig {
                max_functions,
                max_calls_per_function,
                include_temporary_functions,
                root_eclasses,
            });
            SerializedEGraph {
                egraph: res.egraph,
                truncated_functions: res.truncated_functions,
                discarded_functions: res.discarded_functions,
            }
        })
    }

    fn lookup_function(&self, name: &str, key: Vec<Value>) -> Option<Value> {
        self.egraph
            .lookup_function(
                name,
                key.into_iter().map(|v| v.0).collect::<Vec<_>>().as_slice(),
            )
            .map(Value)
    }

    fn eval_expr(&mut self, expr: Expr) -> EggResult<(String, Value)> {
        let expr: egglog::ast::Expr = expr.into();
        self.egraph
            .eval_expr(&expr)
            .map(|(s, v)| (s.name().to_string(), Value(v)))
            .map_err(|e| WrappedError::Egglog(e, format!("\nWhen evaluating expr: {expr}")))
    }

    fn value_to_i64(&self, v: Value) -> i64 {
        self.egraph.value_to_base(v.0)
    }

    fn value_to_bigint(&self, v: Value) -> BigInt {
        let bi: egglog::sort::Z = self.egraph.value_to_base(v.0);
        bi.0
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

    fn value_to_pyobject(
        &self,
        py: Python<'_>,
        py_object_sort: PyObjectSort,
        v: Value,
    ) -> Py<PyAny> {
        let ident = self.egraph.value_to_base::<PyObjectIdent>(v.0);
        py_object_sort.load(py, ident).unbind()
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

    // fn dynamic_cost_model_enode_cost(
    //     &self,
    //     func: String,
    //     args: Vec<Value>,
    // ) -> EggResult<DefaultCost> {
    //     let func = self.egraph.get_function(&func).ok_or_else(|| {
    //         WrappedError::Py(PyRuntimeError::new_err(format!("No such function: {func}")))
    //     })?;
    //     let vals: Vec<egglog::Value> = args.into_iter().map(|v| v.0).collect();
    //     let row = FunctionRow {
    //         vals: &vals,
    //         subsumed: false,
    //     };
    //     Ok(egglog_experimental::DynamicCostModel {}.enode_cost(&self.egraph, &func, &row))
    // }
}

/// Wrapper around Egglog Value. Represents either a primitive base value or a reference to an e-class.
#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Clone)]
#[pyclass(eq, frozen, hash, str = "{0:?}")]
pub struct Value(pub egglog::Value);
