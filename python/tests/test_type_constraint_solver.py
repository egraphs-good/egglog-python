from __future__ import annotations

import pytest
from egg_smol.declarations import *
from egg_smol.type_constraint_solver import *


def test_type_inference() -> None:
    i64 = TypeRefWithVars("i64")
    unit = TypeRefWithVars("Unit")
    K, V = ClassTypeVarRef(0), ClassTypeVarRef(1)
    map = TypeRefWithVars("Map", (K, V))
    map_i64_unit = TypeRefWithVars("Map", (i64, unit))

    cs = TypeConstraintSolver()
    assert cs.infer_return_type([i64], i64, [i64.to_just()]) == i64.to_just()
    with pytest.raises(TypeConstraintError):
        cs.infer_return_type([i64], i64, [unit.to_just()])
    with pytest.raises(TypeConstraintError):
        cs.infer_return_type([], i64, [unit.to_just()])

    assert cs.infer_return_type([map, K], V, [map_i64_unit.to_just(), i64.to_just()]) == unit.to_just()

    with pytest.raises(TypeConstraintError):
        cs.infer_return_type([map, K], V, [map_i64_unit.to_just(), unit.to_just()])

    bound_cs = TypeConstraintSolver.from_type_parameters([i64.to_just(), unit.to_just()])
    assert bound_cs.infer_return_type([K], V, [i64.to_just()]) == unit.to_just()

    with pytest.raises(TypeConstraintError):
        bound_cs.infer_return_type([K], V, [unit.to_just()])
