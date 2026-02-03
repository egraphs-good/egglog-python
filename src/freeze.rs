// Freeze an egglog, turning it an immutable structure that can be printed, serialized, or added back to an e-graph.

use std::{collections::HashMap, fmt::Display};

use egglog::EGraph;
use pyo3::prelude::*;

use crate::{conversions::Command, egraph::Value};

#[pyclass(eq, frozen, get_all)]
#[derive(PartialEq, Eq, Clone, Hash)]
pub struct FrozenRow {
    subsumed: bool,
    inputs: Vec<Value>,
    output: Value,
}

#[pyclass(eq, frozen, hash, get_all)]
#[derive(PartialEq, Eq, Clone, Hash)]
pub struct FrozenFunction {
    input_sorts: Vec<String>,
    output_sort: String,
    rows: Vec<FrozenRow>,
}

#[pyclass(eq, frozen, get_all)]
#[derive(PartialEq, Eq, Clone)]
pub struct FrozenEGraph {
    functions: HashMap<String, FrozenFunction>,
}

impl FrozenEGraph {
    /// Convert this frozen e-graph into a list of egglog commands that can reconstruct it, giv
    pub fn from_egraph(egraph: &EGraph) -> FrozenEGraph {
        let mut functions = HashMap::new();
        for (fname, func) in &egraph.functions {
            let mut rows = Vec::new();
            egraph.backend.for_each(func.backend_id, |row| {
                let frozen_row = FrozenRow {
                    subsumed: row.subsumed,
                    inputs: row.vals[..row.vals.len() - 1]
                        .iter()
                        .cloned()
                        .map(Value)
                        .collect(),
                    output: Value(*row.vals.last().unwrap()),
                };
                rows.push(frozen_row);
            });
            let frozen_function = FrozenFunction {
                input_sorts: func
                    .schema()
                    .input
                    .iter()
                    .map(|s| s.name().to_string())
                    .collect(),
                output_sort: func.schema().output.name().to_string(),
                rows,
            };
            functions.insert(fname.clone(), frozen_function);
        }

        FrozenEGraph { functions }
    }
}
