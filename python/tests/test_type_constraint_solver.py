from __future__ import annotations

import pytest
from egglog.declarations import *
from egglog.type_constraint_solver import *

i64 = JustTypeRef("i64")
unit = JustTypeRef("Unit")
K, V = ClassTypeVarRef("K"), ClassTypeVarRef("V")
map = TypeRefWithVars("Map", (K, V))
map_i64_unit = JustTypeRef("Map", (i64, unit))
decls = Declarations(
    _classes={
        "Map": ClassDecl(
            type_vars = ("K", "V")
        )
    }
)

def test_simple() -> None:
    assert TypeConstraintSolver(Declarations()).infer_return_type([i64.to_var()], i64.to_var(), None, [i64], None) == i64

def test_wrong_arg() -> None:
    with pytest.raises(TypeConstraintError):
        TypeConstraintSolver(Declarations()).infer_return_type([i64.to_var()], i64.to_var(), None, [unit], None)

def test_wrong_number_args() -> None:
    with pytest.raises(TypeConstraintError):
        TypeConstraintSolver(Declarations()).infer_return_type([], i64.to_var(), None, [unit], "Map")


def test_generic() -> None:
    assert TypeConstraintSolver(decls).infer_return_type([map, K], V, None, [map_i64_unit, i64], "Map") == unit

def test_generic_wrong() -> None:
    with pytest.raises(TypeConstraintError):
        TypeConstraintSolver(decls).infer_return_type([map, K], V, None, [map_i64_unit, unit], "Map")

def test_variable() -> None:
    assert (
        TypeConstraintSolver(decls).infer_return_type([map, K], V, V, [map_i64_unit, i64, unit, unit], "Map")
        == unit
    )

def test_variable_wrong() -> None:
    with pytest.raises(TypeConstraintError):
        TypeConstraintSolver(decls).infer_return_type([map, K], V, V, [map_i64_unit, i64, unit, i64], "Map")

def test_bound() -> None:
    bound_cs = TypeConstraintSolver(decls)
    bound_cs.bind_class(map_i64_unit)
    assert bound_cs.infer_return_type([K], V, None, [i64], "Map") == unit

def test_bound_wrong():
    bound_cs = TypeConstraintSolver(decls)
    bound_cs.bind_class(map_i64_unit)
    with pytest.raises(TypeConstraintError):
        bound_cs.infer_return_type([K], V, None, [unit], "Map")

