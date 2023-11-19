use std::collections::HashMap;

use pyo3::prelude::*;

#[pyclass(
    unsendable,
    text_signature = "(py_object_sort=None, *, fact_directory=None, seminaive=True, terms_encoding=False)"
)]
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

    /// Map each op name to a new op name.
    fn map_ops(&mut self, map: HashMap<String, String>) {
        for (_, node) in self.egraph.nodes.iter_mut() {
            if let Some(new_op) = map.get(&node.op) {
                node.op = new_op.clone();
            }
        }
    }
}
