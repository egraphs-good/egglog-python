from __future__ import annotations

import pytest

from egglog.declarations import *
from egglog.type_constraint_solver import *

i64 = JustTypeRef("i64")
unit = JustTypeRef("Unit")
K, V = ClassTypeVarRef("K", __name__), ClassTypeVarRef("V", __name__)
map = TypeRefWithVars("Map", (K, V))
map_i64_unit = JustTypeRef("Map", (i64, unit))
decls = Declarations(_classes={"Map": ClassDecl(type_vars=(K, V))})


def test_simple() -> None:
    tcs = TypeConstraintSolver(Declarations())
    tcs.infer_typevars(i64.to_var(), i64)
    assert tcs.substitute_typevars(i64.to_var()) == i64


def test_wrong_arg() -> None:
    tcs = TypeConstraintSolver(Declarations())
    with pytest.raises(TypeConstraintError):
        tcs.infer_typevars(i64.to_var(), unit)


def test_generic() -> None:
    tcs = TypeConstraintSolver(Declarations())
    tcs.infer_typevars(map, map_i64_unit, "Map")
    tcs.infer_typevars(K, i64, "Map")
    assert tcs.substitute_typevars(V, "Map") == unit


def test_generic_wrong() -> None:
    tcs = TypeConstraintSolver(Declarations())
    tcs.infer_typevars(map, map_i64_unit, "Map")
    with pytest.raises(TypeConstraintError):
        tcs.infer_typevars(K, unit, "Map")


def test_bound() -> None:
    bound_cs = TypeConstraintSolver(decls)
    bound_cs.bind_class(map_i64_unit)
    bound_cs.infer_typevars(K, i64, "Map")
    assert bound_cs.substitute_typevars(V, "Map") == unit


def test_bound_wrong():
    bound_cs = TypeConstraintSolver(decls)
    bound_cs.bind_class(map_i64_unit)
    with pytest.raises(TypeConstraintError):
        bound_cs.infer_typevars(K, unit, "Map")
