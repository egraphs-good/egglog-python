from __future__ import annotations

import pytest

from egglog.declarations import *
from egglog.type_constraint_solver import *

i64 = JustTypeRef(Ident("i64"))
unit = JustTypeRef(Ident("Unit"))
K, V = ClassTypeVarRef(Ident("K")), ClassTypeVarRef(Ident("V"))
map = TypeRefWithVars(Ident("Map"), (K, V))
map_i64_unit = JustTypeRef(Ident("Map"), (i64, unit))
decls = Declarations(_classes={Ident("Map"): ClassDecl(type_vars=(K, V))})


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
    tcs.infer_typevars(map, map_i64_unit, Ident("Map"))
    tcs.infer_typevars(K, i64, Ident("Map"))
    assert tcs.substitute_typevars(V, Ident("Map")) == unit


def test_generic_wrong() -> None:
    tcs = TypeConstraintSolver(Declarations())
    tcs.infer_typevars(map, map_i64_unit, Ident("Map"))
    with pytest.raises(TypeConstraintError):
        tcs.infer_typevars(K, unit, Ident("Map"))


def test_bound() -> None:
    bound_cs = TypeConstraintSolver(decls)
    bound_cs.bind_class(map_i64_unit)
    bound_cs.infer_typevars(K, i64, Ident("Map"))
    assert bound_cs.substitute_typevars(V, Ident("Map")) == unit


def test_bound_wrong():
    bound_cs = TypeConstraintSolver(decls)
    bound_cs.bind_class(map_i64_unit)
    with pytest.raises(TypeConstraintError):
        bound_cs.infer_typevars(K, unit, Ident("Map"))
