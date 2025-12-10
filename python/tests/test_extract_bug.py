"""
Tests extraction with a DAG-based cost model.
from https://github.com/egraphs-good/egglog-python/issues/387#issuecomment-3628927075
"""

from dataclasses import dataclass, field

from egglog import *
from egglog import bindings

# A cost model, approximately equivalent to, greedy_dag_cost_model,
# which operates purely on the `bindings` level, for the sake of
# minimization.

ENode = tuple[str, tuple[bindings.Value, ...]]


@dataclass
class DAGCostValue:
    """Cost value for DAG-based extraction."""

    cost: int
    _values: dict[ENode, int]

    def __eq__(self, rhs: object) -> bool:
        if not isinstance(rhs, DAGCostValue):
            return False
        return self.cost == rhs.cost

    def __lt__(self, other: "DAGCostValue") -> bool:
        return self.cost < other.cost

    def __le__(self, other: "DAGCostValue") -> bool:
        return self.cost <= other.cost

    def __gt__(self, other: "DAGCostValue") -> bool:
        return self.cost > other.cost

    def __ge__(self, other: "DAGCostValue") -> bool:
        return self.cost >= other.cost

    def __hash__(self) -> int:
        return hash(self.cost)

    def __str__(self) -> str:
        return f"DAGCostValue(cost={self.cost})"

    def __repr__(self) -> str:
        return f"DAGCostValue(cost={self.cost}, nchildren={len(self._values)})"


@dataclass
class DAGCost:
    """
    DAG-based cost model for e-graph extraction.

    This cost model counts each unique e-node once, implementing
    a greedy DAG extraction strategy.
    """

    graph: bindings.EGraph
    cache: dict[ENode, DAGCostValue] = field(default_factory=dict)

    def merge_costs(self, costs: list[DAGCostValue], node: ENode, self_cost: int = 0) -> DAGCostValue:
        # if node in self.cache:
        #     return self.cache[node]
        values: dict[ENode, int] = {}
        for child in costs:
            values.update(child._values)
        cost = DAGCostValue(cost=sum(values.values(), start=self_cost), _values=values)
        cost._values[node] = self_cost
        # self.cache[node] = cost
        # print(f"merge {costs=} out={cost}")
        return cost

    def cost_fold(self, fn: str, enode: ENode, children_costs: list[DAGCostValue]) -> DAGCostValue:
        return self.merge_costs(children_costs, enode, 1)
        # print(f"fold {fn=} {out=}")

    def enode_cost(self, name: str, args: list[bindings.Value]) -> ENode:
        return (name, tuple(args))

    def container_cost(self, tp: str, value: bindings.Value, element_costs: list[DAGCostValue]) -> DAGCostValue:
        return self.merge_costs(element_costs, (tp, (value,)), 1)

    def base_value_cost(self, tp: str, value: bindings.Value) -> DAGCostValue:
        return self.merge_costs([], (tp, (value,)), 1)

    @property
    def egg_cost_model(self) -> bindings.CostModel:
        return bindings.CostModel(
            fold=self.cost_fold,
            enode_cost=self.enode_cost,
            container_cost=self.container_cost,
            base_value_cost=self.base_value_cost,
        )


def test_dag_cost_model():
    graph = EGraph()

    commands = graph._egraph.parse_program("""
    (sort S)

    (constructor Si (i64)     S)
    (constructor Swide (S S S S S S S S) S )
    (constructor Ssa (S)       S)
    (constructor Ssb (S)      S)
    (constructor Ssc (S)      S)
    (constructor Sp (S S)     S)


    (let w
    (Swide (Si 0) (Si 1) (Si 2) (Si 3) (Si 4) (Si 5) (Si 6) (Si 7)))

    (let l (Ssa (Ssb (Ssc (Si 0)))))
    (let x (Ssa w))
    (let v (Sp w x))

    (union x l)
    """)
    graph._egraph.run_program(*commands)

    cost_model = DAGCost(graph._egraph)
    extractor = bindings.Extractor(["S"], graph._egraph, cost_model.egg_cost_model)
    termdag = bindings.TermDag()
    value = graph._egraph.lookup_function("v", [])
    assert value is not None
    cost, _term = extractor.extract_best(graph._egraph, termdag, value, "S")

    assert cost.cost in {19, 21}
