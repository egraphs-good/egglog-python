// Wrapper around EGraph type

use crate::conversions::*;
use crate::error::{EggResult, WrappedError};
use crate::py_object_sort::ArcPyObjectSort;
use crate::serialize::SerializedEGraph;

use egglog::{span, SerializeConfig};
use log::info;
use pyo3::prelude::*;
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
        py_object_sort: Option<ArcPyObjectSort>,
        fact_directory: Option<PathBuf>,
        seminaive: bool,
        record: bool,
    ) -> Self {
        let mut egraph = egglog_experimental::new_experimental_egraph();
        egraph.fact_directory = fact_directory;
        egraph.seminaive = seminaive;
        if let Some(py_object_sort) = py_object_sort {
            egraph
                .add_arcsort(py_object_sort.0.clone(), span!())
                .unwrap();
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
    fn run_program(&mut self, commands: Vec<Command>) -> EggResult<Vec<String>> {
        let commands: Vec<egglog::ast::Command> = commands.into_iter().map(|x| x.into()).collect();
        let mut cmds_str = String::new();
        for cmd in &commands {
            cmds_str = cmds_str + &cmd.to_string() + "\n";
        }
        info!("Running commands:\n{}", cmds_str);

        let res = self.egraph.run_program(commands).map_err(|e| {
            WrappedError::Egglog(e, "\nWhen running commands:\n".to_string() + &cmds_str)
        });
        if res.is_ok() {
            if let Some(cmds) = &mut self.cmds {
                cmds.push_str(&cmds_str);
            }
        }
        res
    }

    /// Returns the text of the commands that have been run so far, if `record` was passed.
    #[pyo3(signature = ())]
    fn commands(&self) -> Option<String> {
        self.cmds.clone()
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
        signature = (root_eclasses, *, max_functions=None, max_calls_per_function=None, include_temporary_functions=false),
        text_signature = "(self, root_eclasses, *, max_functions=None, max_calls_per_function=None, include_temporary_functions=False)"
    )]
    fn serialize(
        &mut self,
        root_eclasses: Vec<Expr>,
        max_functions: Option<usize>,
        max_calls_per_function: Option<usize>,
        include_temporary_functions: bool,
    ) -> SerializedEGraph {
        let root_eclasses: Vec<_> = root_eclasses
            .into_iter()
            .map(|x| self.egraph.eval_expr(&egglog::ast::Expr::from(x)).unwrap())
            .collect();
        SerializedEGraph {
            egraph: self.egraph.serialize(SerializeConfig {
                max_functions,
                max_calls_per_function,
                include_temporary_functions,
                root_eclasses,
            }),
        }
    }
}
