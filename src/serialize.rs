use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::egraph::EGraph;

#[pyclass()]
pub struct SerializedEGraph {
    pub egraph: egraph_serialize::EGraph,
}

#[pymethods]
impl SerializedEGraph {
    /// Inline all leaf nodes into their parents.
    fn inline_leaves(&mut self) -> usize {
        self.egraph.inline_leaves()
    }

    /// Inline repeatedly until no more changes are made.
    fn saturate_inline_leaves(&mut self) {
        self.egraph.saturate_inline_leaves()
    }

    /// Serialize the egraph to a dot string.
    fn to_dot(&self) -> String {
        self.egraph.to_dot()
    }

    /// Serialize the egraph to a json string.
    fn to_json(&self) -> String {
        serde_json::to_string(&self.egraph).unwrap()
    }

    /// Split all primitive nodes, as well as other ops that match, into seperate e-classes
    fn split_classes(&mut self, egraph: &EGraph, ops: HashSet<String>) {
        self.egraph.split_classes(|id, node| {
            egraph.egraph.from_node_id(id).is_primitive() || ops.contains(&node.op)
        })
    }

    /// Map each op name to a new op name.
    fn map_ops(&mut self, map: HashMap<String, String>) {
        for (_, node) in self.egraph.nodes.iter_mut() {
            if let Some(new_op) = map.get(&node.op) {
                node.op = new_op.clone();
            }
        }
    }
}
