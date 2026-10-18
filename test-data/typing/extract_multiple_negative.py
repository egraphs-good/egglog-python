from typing import assert_type

from egglog import BaseExpr, DagCostModel, EGraph, i64


def tree_model(egraph: EGraph, expr: BaseExpr, children_costs: list[bool]) -> bool:
    return any(children_costs)


def marginal_cost(graph: EGraph, node: BaseExpr) -> float:
    return 1.0


graph = EGraph()
model = DagCostModel(marginal_cost, 0.0)
tree_variants = graph.extract_multiple(i64(1), 2, include_cost=True, cost_model=tree_model)
dag_batches = graph.extract_multiple([i64(1)], 2, include_cost=True, cost_model=model)
assert_type(tree_variants, list[tuple[i64, str]])  # error: assert-type
assert_type(dag_batches, list[tuple[i64, float]])  # error: assert-type
