use std::{cmp::Ordering, sync::Arc};

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{conversions::Term, egraph::EGraph, egraph::Value, termdag::TermDag};

#[derive(Debug, Clone)]
// Wrap in Arc so we can clone efficiently
// https://pyo3.rs/main/migration.html#pyclone-is-now-gated-behind-the-py-clone-feature
struct Cost(Arc<Py<PyAny>>);

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
///
/// If not provided, default to the same behavior as DynamicCostModel so that fast paths can be used when possible.
#[derive(Debug, Clone)]
#[pyclass(
    frozen,
    str = "CostModel({fold:?}, {enode_cost:?}, {container_cost:?}, {base_value_cost:?}"
)]
pub struct CostModel {
    /// Function mapping from a term's head and its children's costs to the term's total cost.
    /// If None, simply sums the children's costs and the head cost.
    /// (head: str, head_cost: COST, children_costs: list[COST]) -> COST | None
    fold: Option<Arc<Py<PyAny>>>,
    /// Function mapping from an expression node to its cost.
    /// If None, defaults to the the set cost of of the value, or the cost of the function if it is known, or 1 otherwise.
    /// (func_name: str, args: list[Value]) -> COST | None
    enode_cost: Option<Arc<Py<PyAny>>>,
    /// Function mapping from a container value to its cost given the costs of its elements.
    /// If none, sums the element costs starting at 0 for an empty container.
    /// (sort_name: str, value: Value, element_costs: list[COST]) -> COST | None
    container_cost: Option<Arc<Py<PyAny>>>,
    /// Function mapping from a base value to its cost.
    /// If none, defaults to 1.
    /// (sort_name: str, value: Value) -> COST | None
    base_value_cost: Option<Arc<Py<PyAny>>>,
}

#[pymethods]
impl CostModel {
    #[new]
    fn new(
        fold: Option<Py<PyAny>>,
        enode_cost: Option<Py<PyAny>>,
        container_cost: Option<Py<PyAny>>,
        base_value_cost: Option<Py<PyAny>>,
    ) -> Self {
        CostModel {
            fold: fold.map(Arc::new),
            enode_cost: enode_cost.map(Arc::new),
            container_cost: container_cost.map(Arc::new),
            base_value_cost: base_value_cost.map(Arc::new),
        }
    }
}

impl egglog::extract::CostModel<Cost> for CostModel {
    fn fold(&self, head: &str, children_cost: &[Cost], head_cost: Cost) -> Cost {
        Cost(Arc::new(Python::attach(|py| match &self.fold {
            Some(fold) => {
                let children_cost = children_cost
                    .iter()
                    .map(|c| c.0.clone_ref(py))
                    .collect::<Vec<_>>();
                let res = fold.call1(py, (head, head_cost.0.clone_ref(py), children_cost));
                res.unwrap()
            }
            // copied from TreeAdditiveCostModel but changed type of cost
            None => children_cost
                .iter()
                .fold(head_cost.0.bind(py).clone(), |s, c| {
                    s.add(c.0.clone_ref(py)).unwrap()
                })
                .unbind(),
        })))
    }

    fn enode_cost(
        &self,
        egraph: &egglog::EGraph,
        func: &egglog::Function,
        row: &egglog::FunctionRow<'_>,
    ) -> Cost {
        Cost(Arc::new(Python::attach(|py| match &self.enode_cost {
            Some(enode_cost) => {
                let mut values = row.vals.iter().map(|v| Value(*v)).collect::<Vec<_>>();
                // Remove last element which is the output
                // this is not needed because the only thing we can do with the output is look up an analysis
                // which we can also do with the original function
                values.pop().unwrap();
                let res = enode_cost.call1(py, (func.name(), values));
                res.unwrap()
            }
            None => egglog_experimental::DynamicCostModel {}
                .enode_cost(egraph, func, row)
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        })))
    }

    fn container_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
        element_costs: &[Cost],
    ) -> Cost {
        Cost(Arc::new(Python::attach(|py| match &self.container_cost {
            Some(container_cost) => {
                let element_costs = element_costs
                    .iter()
                    .map(|c| c.0.clone_ref(py))
                    .collect::<Vec<_>>();
                let res = container_cost.call1(py, (sort.name(), Value(value), element_costs));
                res.unwrap()
            }
            None => element_costs
                .iter()
                .fold(0i64.into_pyobject(py).unwrap().as_any().clone(), |s, c| {
                    s.add(c.0.clone_ref(py)).unwrap()
                })
                .unbind(),
        })))
    }

    // https://github.com/PyO3/pyo3/issues/1190
    fn base_value_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
    ) -> Cost {
        Cost(Arc::new(Python::attach(|py| match &self.base_value_cost {
            Some(base_value_cost) => base_value_cost
                .call1(py, (sort.name(), Value(value)))
                .unwrap(),
            None => 1i64.into_pyobject(py).unwrap().as_any().clone().unbind(),
        })))
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
