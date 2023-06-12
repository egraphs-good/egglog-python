from __future__ import annotations

import pytest
from egglog.declarations import *
from egglog.type_constraint_solver import *


def test_type_inference() -> None:
    i64 = TypeRefWithVars("i64")
    unit = TypeRefWithVars("Unit")
    K, V = ClassTypeVarRef(0), ClassTypeVarRef(1)
    map = TypeRefWithVars("Map", (K, V))
    map_i64_unit = TypeRefWithVars("Map", (i64, unit))

    cs = TypeConstraintSolver()
    assert cs.infer_return_type([i64], i64, None, [i64.to_just()]) == i64.to_just()
    with pytest.raises(TypeConstraintError):
        cs.infer_return_type([i64], i64, None, [unit.to_just()])
    with pytest.raises(TypeConstraintError):
        cs.infer_return_type([], i64, None, [unit.to_just()])

    assert cs.infer_return_type([map, K], V, None, [map_i64_unit.to_just(), i64.to_just()]) == unit.to_just()

    with pytest.raises(TypeConstraintError):
        cs.infer_return_type([map, K], V, None, [map_i64_unit.to_just(), unit.to_just()])

    bound_cs = TypeConstraintSolver.from_type_parameters([i64.to_just(), unit.to_just()])
    assert bound_cs.infer_return_type([K], V, None, [i64.to_just()]) == unit.to_just()

    with pytest.raises(TypeConstraintError):
        bound_cs.infer_return_type([K], V, None, [unit.to_just()])

    # Test variable args
    assert (
        cs.infer_return_type([map, K], V, V, [map_i64_unit.to_just(), i64.to_just(), unit.to_just(), unit.to_just()])
        == unit.to_just()
    )
    with pytest.raises(TypeConstraintError):
        cs.infer_return_type([map, K], V, V, [map_i64_unit.to_just(), i64.to_just(), unit.to_just(), i64.to_just()])
