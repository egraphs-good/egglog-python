// Freeze an egglog, turning it into an immutable structure that can be printed, serialized, or added back to an e-graph.

use egglog::{EGraph, ast::FunctionSubtype};
use indexmap::IndexMap;
use pyo3::prelude::*;

use crate::egraph::Value;

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
    is_let_binding: bool,
    rows: Vec<FrozenRow>,
}

#[pyclass(eq, frozen, get_all)]
#[derive(PartialEq, Eq, Clone)]
pub struct FrozenEGraph {
    functions: IndexMap<String, FrozenFunction>,
}

impl FrozenEGraph {
    /// Convert a live `EGraph` into an immutable `FrozenEGraph` snapshot.
    pub fn from_egraph(egraph: &EGraph) -> FrozenEGraph {
        let mut functions = IndexMap::new();
        for (fname, func) in egraph.functions_iter() {
            let mut rows = Vec::new();
            match func.func_type().subtype {
                FunctionSubtype::Constructor => egraph
                    .constructor_enodes(fname, |enode| {
                        rows.push(FrozenRow {
                            subsumed: enode.subsumed,
                            inputs: enode.children.iter().copied().map(Value).collect(),
                            output: Value(enode.eclass),
                        });
                    })
                    .unwrap(),
                FunctionSubtype::Custom => egraph
                    .function_entries(fname, |entry| {
                        rows.push(FrozenRow {
                            subsumed: entry.subsumed,
                            inputs: entry.inputs.iter().copied().map(Value).collect(),
                            output: Value(entry.output),
                        });
                    })
                    .unwrap(),
            }
            let func_type = func.func_type();
            let frozen_function = FrozenFunction {
                input_sorts: func_type
                    .input
                    .iter()
                    .map(|s| s.name().to_string())
                    .collect(),
                output_sort: func_type.output.name().to_string(),
                rows,
                is_let_binding: func.is_let_binding(),
            };
            functions.insert(fname.clone(), frozen_function);
        }

        FrozenEGraph { functions }
    }
}
