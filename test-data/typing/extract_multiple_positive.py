from typing import assert_type

from egglog import BaseExpr, DagCostModel, EGraph, i64


def tree_model(egraph: EGraph, expr: BaseExpr, children_costs: list[bool]) -> bool:
    return any(children_costs)


def marginal_cost(graph: EGraph, node: BaseExpr) -> float:
    return 1.0


graph = EGraph()
model = DagCostModel(marginal_cost, 0.0)
assert_type(graph.extract_multiple(i64(1), 2), list[i64])
assert_type(graph.extract_multiple([i64(1)], 2), list[list[i64]])
assert_type(graph.extract_multiple(i64(1), 2, cost_model=tree_model), list[i64])
assert_type(graph.extract_multiple([i64(1)], 2, cost_model=model), list[list[i64]])
assert_type(graph.extract_multiple(i64(1), 2, include_cost=True), list[tuple[i64, int]])
assert_type(graph.extract_multiple([i64(1)], 2, include_cost=True), list[list[tuple[i64, int]]])
assert_type(graph.extract_multiple(i64(1), 2, include_cost=True, cost_model=tree_model), list[tuple[i64, bool]])
assert_type(graph.extract_multiple([i64(1)], 2, include_cost=True, cost_model=tree_model), list[list[tuple[i64, bool]]])
assert_type(graph.extract_multiple(i64(1), 2, include_cost=True, cost_model=model), list[tuple[i64, float]])
assert_type(graph.extract_multiple([i64(1)], 2, include_cost=True, cost_model=model), list[list[tuple[i64, float]]])
