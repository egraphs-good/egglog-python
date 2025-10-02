use std::cmp::Ordering;

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{conversions::Term, egraph::EGraph, egraph::Value, termdag::TermDag};

#[derive(Debug)]
// We have to store the result, since the cost model does not return errors
struct Cost(Py<PyAny>);

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        Python::attach(|py| self.0.bind(py).compare(other.0.bind(py)).unwrap())
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Cost {
    fn eq(&self, other: &Self) -> bool {
        Python::attach(|py| self.0.bind(py).eq(other.0.bind(py))).unwrap()
    }
}

impl Eq for Cost {}

impl Clone for Cost {
    fn clone(&self) -> Self {
        Python::attach(|py| Cost(self.0.clone_ref(py)))
    }
}

impl egglog::extract::Cost for Cost {
    fn identity() -> Self {
        panic!("Should never be called from Rust directly");
    }

    fn unit() -> Self {
        panic!("Should never be called from Rust directly");
    }

    fn combine(self, _other: &Self) -> Self {
        panic!("Should never be called from Rust directly");
    }
}

/// Cost model defined by Python functions.
#[derive(Debug)]
#[pyclass(
    frozen,
    str = "CostModel({fold:?}, {enode_cost:?}, {container_cost:?}, {base_value_cost:?}"
)]
pub struct CostModel {
    /// Function mapping from a term's head and its children's costs to the term's total cost.
    /// (head: str, head_cost: COST, children_costs: list[COST]) -> COST
    fold: Py<PyAny>,
    /// Function mapping from an expression node to its cost.
    /// (func_name: str, args: list[Value]) -> COST
    enode_cost: Py<PyAny>,
    /// Function mapping from a container value to its cost given the costs of its elements.
    /// (sort_name: str, value: Value, element_costs: list[COST]) -> COST
    container_cost: Py<PyAny>,
    /// Function mapping from a base value to its cost.
    /// (sort_name: str, value: Value) -> COST
    base_value_cost: Py<PyAny>,
}

#[pymethods]
impl CostModel {
    #[new]
    fn new(
        fold: Py<PyAny>,
        enode_cost: Py<PyAny>,
        container_cost: Py<PyAny>,
        base_value_cost: Py<PyAny>,
    ) -> Self {
        CostModel {
            fold,
            enode_cost,
            container_cost,
            base_value_cost,
        }
    }
}

impl Clone for CostModel {
    fn clone(&self) -> Self {
        Python::attach(|py| CostModel {
            fold: self.fold.clone_ref(py),
            enode_cost: self.enode_cost.clone_ref(py),
            container_cost: self.container_cost.clone_ref(py),
            base_value_cost: self.base_value_cost.clone_ref(py),
        })
    }
}

impl egglog::extract::CostModel<Cost> for CostModel {
    fn fold(&self, head: &str, children_cost: &[Cost], head_cost: Cost) -> Cost {
        Cost(Python::attach(|py| {
            let head_cost = head_cost.0.clone_ref(py);
            let children_cost = children_cost
                .into_iter()
                .cloned()
                .map(|c| c.0.clone_ref(py))
                .collect::<Vec<_>>();
            self.fold
                .call1(py, (head, head_cost, children_cost))
                .unwrap()
        }))
    }

    fn enode_cost(
        &self,
        egraph: &egglog::EGraph,
        func: &egglog::Function,
        row: &egglog::FunctionRow<'_>,
    ) -> Cost {
        Python::attach(|py| {
            let mut values = row.vals.iter().map(|v| Value(*v)).collect::<Vec<_>>();
            // Remove last element which is the output
            // this is not needed because the only thing we can do with the output is look up an analysis
            // which we can also do with the original function
            values.pop().unwrap();
            Cost(self.enode_cost.call1(py, (func.name(), values)).unwrap())
        })
    }

    fn container_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
        element_costs: &[Cost],
    ) -> Cost {
        Cost(Python::attach(|py| {
            let element_costs = element_costs
                .into_iter()
                .cloned()
                .map(|c| c.0.clone_ref(py))
                .collect::<Vec<_>>();
            self.container_cost
                .call1(py, (sort.name(), Value(value), element_costs))
                .unwrap()
        }))
    }

    // https://github.com/PyO3/pyo3/issues/1190
    fn base_value_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
    ) -> Cost {
        Python::attach(|py| {
            Cost(
                self.base_value_cost
                    .call1(py, (sort.name(), Value(value)))
                    .unwrap(),
            )
        })
    }
}

// TODO: Don't progress just return an error if there was an exception?

#[pyclass(unsendable)]
pub struct Extractor(egglog::extract::Extractor<Cost>);

#[pymethods]
impl Extractor {
    /// Create a new extractor from the given egraph and cost model.
    ///
    /// Bulk of the computation happens at initialization time.
    /// The later extractions only reuses saved results.
    /// This means a new extractor must be created if the egraph changes.
    /// Holding a reference to the egraph would enforce this but prevents the extractor being reused.
    ///
    /// For convenience, if the rootsorts is `None`, it defaults to extract all extractable rootsorts.
    #[new]
    fn new(
        py: Python<'_>,
        rootsorts: Option<Vec<String>>,
        egraph: &EGraph,
        cost_model: CostModel,
    ) -> PyResult<Self> {
        let egraph = &egraph.egraph;
        // Transforms sorts to arcsorts, returning an error if any are unknown
        let rootsorts = rootsorts
            .map(|rs| {
                rs.into_iter()
                    .map(|s| egraph.get_sort_by_name(&s).cloned())
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PyValueError::new_err("Unknown sort in rootsorts"))
            })
            .map_or(Ok(None), |r| r.map(Some))?;
        let extractor =
            egglog::extract::Extractor::compute_costs_from_rootsorts(rootsorts, egraph, cost_model);
        if let Some(err) = PyErr::take(py) {
            return Err(err);
        };
        Ok(Extractor(extractor))
    }

    /// Extract the best term of a value from a given sort.
    ///
    /// This function expects the sort to be already computed,
    /// which can be one of the rootsorts, or reachable from rootsorts, or primitives, or containers of computed sorts.
    fn extract_best(
        &self,
        py: Python<'_>,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: String,
    ) -> PyResult<(Py<PyAny>, Term)> {
        let sort = egraph
            .egraph
            .get_sort_by_name(&sort)
            .ok_or(PyValueError::new_err("Unknown sort"))?;
        let (cost, term) = self
            .0
            .extract_best_with_sort(&egraph.egraph, &mut termdag.0, value.0, sort.clone())
            .ok_or(PyValueError::new_err("Unextractable root".to_string()))?;
        Ok((cost.0.clone_ref(py), term.into()))
    }

    /// Extract variants of an e-class.
    ///
    /// The variants are selected by first picking `nvariants` e-nodes with the lowest cost from the e-class
    /// and then extracting a term from each e-node.
    fn extract_variants(
        &self,
        py: Python<'_>,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
        sort: String,
    ) -> PyResult<Vec<(Py<PyAny>, Term)>> {
        let sort = egraph
            .egraph
            .get_sort_by_name(&sort)
            .ok_or(PyValueError::new_err("Unknown sort"))?;
        let variants = self.0.extract_variants_with_sort(
            &egraph.egraph,
            &mut termdag.0,
            value.0,
            nvariants,
            sort.clone(),
        );
        Ok(variants
            .into_iter()
            .map(|(cost, term)| (cost.0.clone_ref(py), term.into()))
            .collect())
    }
}
