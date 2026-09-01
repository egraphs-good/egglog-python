use std::{
    cmp::Ordering,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::Arc,
};

use egglog::{
    Term, TermId,
    extract::{
        DagCostModel as EggDagCostModel, MonoidCost, TreeCostModel as EggTreeCostModel,
        TreeCostModelFromDag, TreeExtractor,
    },
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{egraph::EGraph, egraph::Value, termdag::TermDag, tracing_otel};

/// Private unwind payload used to cross Rust APIs whose cost traits cannot
/// return errors. Only this payload is translated back to Python; unrelated
/// Rust panics continue unwinding normally.
struct PythonCostError(PyErr);

fn python_or_unwind<T>(result: PyResult<T>) -> T {
    result.unwrap_or_else(|err| resume_unwind(Box::new(PythonCostError(err))))
}

fn catch_python_cost_error<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<PythonCostError>() {
            Ok(error) => Err(error.0),
            Err(payload) => resume_unwind(payload),
        },
    }
}

#[derive(Debug)]
struct Cost(Arc<Py<PyAny>>);

impl Cost {
    fn from_py(value: Py<PyAny>) -> Self {
        Self(Arc::new(value))
    }

    fn to_py(&self, py: Python<'_>) -> Py<PyAny> {
        self.0.as_ref().clone_ref(py)
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        Python::attach(|py| python_or_unwind(self.0.bind(py).compare(other.0.bind(py))))
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Cost {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Cost {}

impl Clone for Cost {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(Debug)]
struct TreeEnodeCost {
    head: String,
    annotation: Py<PyAny>,
}

#[derive(Debug)]
struct TreeContainerCost {
    sort: String,
    value: Value,
}

/// Tree cost model defined by Python functions.
#[derive(Debug)]
#[pyclass(
    frozen,
    str = "CostModel({fold:?}, {enode_cost:?}, {container_cost:?}, {base_value_cost:?}"
)]
pub struct CostModel {
    /// Function mapping from a term's head and its children's costs to the term's total cost.
    /// (head: str, head_cost: ENODE_COST, children_costs: list[COST]) -> COST
    fold: Py<PyAny>,
    /// Function mapping from an expression node to an annotation consumed by `fold`.
    /// (func_name: str, args: list[Value]) -> ENODE_COST
    enode_cost: Py<PyAny>,
    /// Function mapping from a container value and element costs to its total cost.
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
        Self {
            fold,
            enode_cost,
            container_cost,
            base_value_cost,
        }
    }
}

impl Clone for CostModel {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            fold: self.fold.clone_ref(py),
            enode_cost: self.enode_cost.clone_ref(py),
            container_cost: self.container_cost.clone_ref(py),
            base_value_cost: self.base_value_cost.clone_ref(py),
        })
    }
}

impl EggTreeCostModel<Cost> for CostModel {
    type EnodeCost = TreeEnodeCost;
    type ContainerCost = TreeContainerCost;

    fn fold_enode_cost(&self, enode_cost: Self::EnodeCost, children_cost: &[Cost]) -> Cost {
        Python::attach(|py| {
            let children_cost = children_cost
                .iter()
                .map(|cost| cost.to_py(py))
                .collect::<Vec<_>>();
            Cost::from_py(python_or_unwind(
                self.fold
                    .call1(py, (enode_cost.head, enode_cost.annotation, children_cost)),
            ))
        })
    }

    fn enode_cost(
        &self,
        _egraph: &egglog::EGraph,
        func: &egglog::Function,
        enode: &egglog::Enode<'_>,
    ) -> Self::EnodeCost {
        Python::attach(|py| {
            let values = enode
                .children
                .iter()
                .map(|value| Value(*value))
                .collect::<Vec<_>>();
            TreeEnodeCost {
                head: func.name().to_owned(),
                annotation: python_or_unwind(self.enode_cost.call1(py, (func.name(), values))),
            }
        })
    }

    fn fold_container_cost(
        &self,
        container_cost: Self::ContainerCost,
        element_costs: &[Cost],
    ) -> Cost {
        Python::attach(|py| {
            let element_costs = element_costs
                .iter()
                .map(|cost| cost.to_py(py))
                .collect::<Vec<_>>();
            Cost::from_py(python_or_unwind(self.container_cost.call1(
                py,
                (container_cost.sort, container_cost.value, element_costs),
            )))
        })
    }

    fn container_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
    ) -> Self::ContainerCost {
        TreeContainerCost {
            sort: sort.name().to_owned(),
            value: Value(value),
        }
    }

    // https://github.com/PyO3/pyo3/issues/1190
    fn base_value_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
    ) -> Cost {
        Python::attach(|py| {
            Cost::from_py(python_or_unwind(
                self.base_value_cost.call1(py, (sort.name(), Value(value))),
            ))
        })
    }
}

#[derive(Debug)]
struct DagCostContext {
    identity: Arc<Py<PyAny>>,
}

#[derive(Clone, Debug)]
enum DagCost {
    Identity,
    Value {
        value: Arc<Py<PyAny>>,
        context: Arc<DagCostContext>,
    },
}

impl DagCost {
    fn value(value: Py<PyAny>, context: Arc<DagCostContext>) -> Self {
        Self::Value {
            value: Arc::new(value),
            context,
        }
    }

    fn to_py(&self, py: Python<'_>, context: &Arc<DagCostContext>) -> Py<PyAny> {
        match self {
            Self::Identity => context.identity.as_ref().clone_ref(py),
            Self::Value { value, .. } => value.as_ref().clone_ref(py),
        }
    }

    fn compare_values(left: &Arc<Py<PyAny>>, right: &Arc<Py<PyAny>>) -> Ordering {
        Python::attach(|py| python_or_unwind(left.bind(py).compare(right.bind(py))))
    }

    fn ensure_same_context(left: &Arc<DagCostContext>, right: &Arc<DagCostContext>) {
        if !Arc::ptr_eq(left, right) {
            python_or_unwind::<()>(Err(PyValueError::new_err(
                "cannot combine costs from different extraction contexts",
            )));
        }
    }
}

impl Ord for DagCost {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Identity, Self::Identity) => Ordering::Equal,
            (
                Self::Value {
                    value: left,
                    context: left_context,
                },
                Self::Value {
                    value: right,
                    context: right_context,
                },
            ) => {
                Self::ensure_same_context(left_context, right_context);
                Self::compare_values(left, right)
            }
            (Self::Identity, Self::Value { value, context }) => {
                Self::compare_values(&context.identity, value)
            }
            (Self::Value { value, context }, Self::Identity) => {
                Self::compare_values(value, &context.identity)
            }
        }
    }
}

impl PartialOrd for DagCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DagCost {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for DagCost {}

impl MonoidCost for DagCost {
    fn identity() -> Self {
        Self::Identity
    }

    fn combine(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Identity, Self::Identity) => Self::Identity,
            (Self::Identity, value @ Self::Value { .. }) => value.clone(),
            (value @ Self::Value { .. }, Self::Identity) => value,
            (
                Self::Value {
                    value: left,
                    context: left_context,
                },
                Self::Value {
                    value: right,
                    context: right_context,
                },
            ) => {
                Self::ensure_same_context(&left_context, right_context);
                let combined = Python::attach(|py| {
                    python_or_unwind(left.bind(py).add(right.bind(py))).unbind()
                });
                Self::value(combined, left_context)
            }
        }
    }
}

/// Additive marginal DAG cost model defined by Python functions.
#[derive(Debug)]
#[pyclass(
    frozen,
    str = "DagCostModel({identity:?}, {enode_cost:?}, {container_cost:?}, {base_value_cost:?}"
)]
pub struct DagCostModel {
    identity: Py<PyAny>,
    enode_cost: Py<PyAny>,
    container_cost: Py<PyAny>,
    base_value_cost: Py<PyAny>,
}

#[pymethods]
impl DagCostModel {
    #[new]
    fn new(
        identity: Py<PyAny>,
        enode_cost: Py<PyAny>,
        container_cost: Py<PyAny>,
        base_value_cost: Py<PyAny>,
    ) -> Self {
        Self {
            identity,
            enode_cost,
            container_cost,
            base_value_cost,
        }
    }
}

impl DagCostModel {
    fn runtime(&self, py: Python<'_>) -> RuntimeDagCostModel {
        RuntimeDagCostModel {
            context: Arc::new(DagCostContext {
                identity: Arc::new(self.identity.clone_ref(py)),
            }),
            enode_cost: Arc::new(self.enode_cost.clone_ref(py)),
            container_cost: Arc::new(self.container_cost.clone_ref(py)),
            base_value_cost: Arc::new(self.base_value_cost.clone_ref(py)),
        }
    }
}

#[derive(Clone, Debug)]
struct RuntimeDagCostModel {
    context: Arc<DagCostContext>,
    enode_cost: Arc<Py<PyAny>>,
    container_cost: Arc<Py<PyAny>>,
    base_value_cost: Arc<Py<PyAny>>,
}

impl EggDagCostModel<DagCost> for RuntimeDagCostModel {
    fn enode_cost(
        &self,
        _egraph: &egglog::EGraph,
        func: &egglog::Function,
        enode: &egglog::Enode<'_>,
    ) -> DagCost {
        Python::attach(|py| {
            let values = enode
                .children
                .iter()
                .map(|value| Value(*value))
                .collect::<Vec<_>>();
            DagCost::value(
                python_or_unwind(self.enode_cost.call1(py, (func.name(), values))),
                self.context.clone(),
            )
        })
    }

    fn container_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
    ) -> DagCost {
        Python::attach(|py| {
            DagCost::value(
                python_or_unwind(self.container_cost.call1(py, (sort.name(), Value(value)))),
                self.context.clone(),
            )
        })
    }

    fn base_value_cost(
        &self,
        _egraph: &egglog::EGraph,
        sort: &egglog::ArcSort,
        value: egglog::Value,
    ) -> DagCost {
        Python::attach(|py| {
            DagCost::value(
                python_or_unwind(self.base_value_cost.call1(py, (sort.name(), Value(value)))),
                self.context.clone(),
            )
        })
    }
}

#[derive(Copy, Clone)]
enum ExtractionMode {
    Tree,
    GreedyDag,
}

fn extraction_mode(name: &str) -> PyResult<ExtractionMode> {
    match name {
        "tree" => Ok(ExtractionMode::Tree),
        "greedy-dag" => Ok(ExtractionMode::GreedyDag),
        _ => Err(PyValueError::new_err(format!(
            "unknown extractor {name:?}; expected 'tree' or 'greedy-dag'"
        ))),
    }
}

fn roots_from_names(
    egraph: &egglog::EGraph,
    roots: Vec<(String, Value)>,
) -> PyResult<Vec<(egglog::ArcSort, egglog::Value)>> {
    roots
        .into_iter()
        .map(|(sort, value)| {
            let arcsort = egraph
                .get_sort_by_name(&sort)
                .cloned()
                .ok_or_else(|| PyValueError::new_err(format!("unknown sort {sort:?}")))?;
            Ok((arcsort, value.0))
        })
        .collect()
}

fn rootsorts_from_names(
    egraph: &egglog::EGraph,
    rootsorts: Option<&[String]>,
) -> PyResult<Option<Vec<egglog::ArcSort>>> {
    rootsorts
        .map(|rootsorts| {
            rootsorts
                .iter()
                .map(|sort| {
                    egraph
                        .get_sort_by_name(sort)
                        .cloned()
                        .ok_or_else(|| PyValueError::new_err(format!("unknown sort {sort:?}")))
                })
                .collect()
        })
        .transpose()
}

fn copy_termdag(source: &egglog::TermDag, target: &mut egglog::TermDag) -> Vec<TermId> {
    let mut copied = Vec::with_capacity(source.size());
    for term in 0..source.size() {
        let target_term = match source.get(term).clone() {
            Term::Lit(literal) => target.lit(literal),
            Term::Var(variable) => target.var(variable),
            Term::App(head, children) => {
                let children = children.into_iter().map(|child| copied[child]).collect();
                target.app(head, children)
            }
        };
        copied.push(target_term);
    }
    copied
}

/// Compatibility facade for the former owned core extractor. The current core
/// extractor borrows its e-graph, so this object stores preparation inputs and
/// prepares locally for each extraction call.
#[pyclass(unsendable)]
pub struct Extractor {
    rootsorts: Option<Vec<String>>,
    cost_model: CostModel,
}

#[pymethods]
impl Extractor {
    /// Create a new extractor facade from the given egraph and cost model.
    ///
    /// For convenience, if the rootsorts is `None`, it defaults to all
    /// extractable rootsorts.
    #[new]
    #[pyo3(signature = (rootsorts, egraph, cost_model, *, traceparent=None, tracestate=None))]
    fn new(
        rootsorts: Option<Vec<String>>,
        egraph: &EGraph,
        cost_model: CostModel,
        traceparent: Option<String>,
        tracestate: Option<String>,
    ) -> PyResult<Self> {
        let _context_guard =
            tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
        let span = tracing::info_span!(
            "bindings.extractor.new",
            has_rootsorts = rootsorts.is_some()
        );
        let _entered = span.enter();
        if let Some(rootsorts) = &rootsorts {
            for sort in rootsorts {
                if egraph.egraph.get_sort_by_name(sort).is_none() {
                    return Err(PyValueError::new_err(format!("unknown sort {sort:?}")));
                }
            }
        }
        Ok(Self {
            rootsorts,
            cost_model,
        })
    }

    /// Extract the best term of a value from a given sort.
    #[pyo3(signature = (egraph, termdag, value, sort, *, traceparent=None, tracestate=None))]
    fn extract_best(
        &self,
        py: Python<'_>,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: String,
        traceparent: Option<String>,
        tracestate: Option<String>,
    ) -> PyResult<(Py<PyAny>, TermId)> {
        let _context_guard =
            tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
        let span = tracing::info_span!("bindings.extractor.extract_best", sort = %sort);
        let _entered = span.enter();
        let sort = egraph
            .egraph
            .get_sort_by_name(&sort)
            .cloned()
            .ok_or_else(|| PyValueError::new_err("unknown sort"))?;
        let rootsorts = rootsorts_from_names(&egraph.egraph, self.rootsorts.as_deref())?;
        let extracted = catch_python_cost_error(|| {
            let extractor = TreeExtractor::compute_costs_from_rootsorts(
                rootsorts,
                &egraph.egraph,
                self.cost_model.clone(),
            );
            let mut local_termdag = egglog::TermDag::default();
            extractor
                .extract_best_with_sort(&mut local_termdag, value.0, sort)
                .map(|extracted| (local_termdag, extracted.cost, extracted.term))
        })?
        .ok_or_else(|| PyValueError::new_err("unextractable root"))?;
        let (local_termdag, cost, term) = extracted;
        let term = copy_termdag(&local_termdag, &mut termdag.0)[term];
        Ok((cost.to_py(py), term))
    }

    /// Extract variants of an e-class.
    #[pyo3(signature = (egraph, termdag, value, nvariants, sort, *, traceparent=None, tracestate=None))]
    fn extract_variants(
        &self,
        py: Python<'_>,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        nvariants: usize,
        sort: String,
        traceparent: Option<String>,
        tracestate: Option<String>,
    ) -> PyResult<Vec<(Py<PyAny>, TermId)>> {
        let _context_guard =
            tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
        let span = tracing::info_span!("bindings.extractor.extract_variants", sort = %sort, variant_count = nvariants);
        let _entered = span.enter();
        let sort = egraph
            .egraph
            .get_sort_by_name(&sort)
            .cloned()
            .ok_or_else(|| PyValueError::new_err("unknown sort"))?;
        let rootsorts = rootsorts_from_names(&egraph.egraph, self.rootsorts.as_deref())?;
        let (local_termdag, variants) = catch_python_cost_error(|| {
            let extractor = TreeExtractor::compute_costs_from_rootsorts(
                rootsorts,
                &egraph.egraph,
                self.cost_model.clone(),
            );
            let mut local_termdag = egglog::TermDag::default();
            let variants =
                extractor.extract_variants_with_sort(&mut local_termdag, value.0, nvariants, sort);
            (local_termdag, variants)
        })?;
        let copied = copy_termdag(&local_termdag, &mut termdag.0);
        Ok(variants
            .into_iter()
            .map(|variant| (variant.cost.to_py(py), copied[variant.term]))
            .collect())
    }
}

/// Extract the best term for each root with a custom additive marginal model.
#[pyfunction]
#[pyo3(signature = (egraph, roots, cost_model, *, extractor="tree", traceparent=None, tracestate=None))]
pub fn extract_best_with_dag_cost_model(
    py: Python<'_>,
    egraph: &EGraph,
    roots: Vec<(String, Value)>,
    cost_model: &DagCostModel,
    extractor: &str,
    traceparent: Option<String>,
    tracestate: Option<String>,
) -> PyResult<(TermDag, Vec<Option<(Py<PyAny>, TermId)>>)> {
    let _context_guard =
        tracing_otel::attach_parent_context(traceparent.as_deref(), tracestate.as_deref());
    let span = tracing::info_span!(
        "bindings.extract_best_with_dag_cost_model",
        root_count = roots.len(),
        extractor
    );
    let _entered = span.enter();
    let roots = roots_from_names(&egraph.egraph, roots)?;
    let mode = extraction_mode(extractor)?;
    let runtime = cost_model.runtime(py);
    let context = runtime.context.clone();
    let extracted = catch_python_cost_error(|| match mode {
        ExtractionMode::Tree => egraph
            .egraph
            .extract_best_with_cost_model(roots, TreeCostModelFromDag(runtime)),
        ExtractionMode::GreedyDag => {
            egglog_experimental::extract_best_greedy_dag(&egraph.egraph, roots, runtime)
        }
    })?
    .map_err(crate::error::WrappedError::Egglog)?;
    let terms = extracted
        .terms
        .into_iter()
        .map(|term| term.map(|term| (term.cost.to_py(py, &context), term.term)))
        .collect();
    Ok((TermDag(extracted.termdag), terms))
}
