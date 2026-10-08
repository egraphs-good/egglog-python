from __future__ import annotations

import doctest
import math
import struct
from copy import copy

import pytest

from egglog import expr_parts, f64
from egglog.declarations import *
from egglog.exp import array_api
from egglog.runtime import *
from egglog.thunk import *
from egglog.type_constraint_solver import *


def test_public_f64_structural_identity_and_signed_zero():
    source = float("nan")
    left, same_source, fresh = f64(source), f64(source), f64(float("nan"))
    assert expr_parts(left) == expr_parts(same_source)
    assert bool(left == same_source)
    assert hash(left) == hash(same_source)
    assert expr_parts(left) != expr_parts(fresh)
    assert not bool(left == fresh)
    assert expr_parts(left) == expr_parts(copy(left))
    assert hash(left) == hash(copy(left))
    assert left.value is source
    assert math.isnan(fresh.value)
    assert struct.pack("!d", left.value) == struct.pack("!d", fresh.value)

    positive, negative = f64(0.0), f64(-0.0)
    assert expr_parts(positive) == expr_parts(negative)
    assert bool(positive == negative)
    assert hash(positive) == hash(negative)
    assert struct.pack("!d", positive.value) != struct.pack("!d", negative.value)


def test_type_str():
    decls = Declarations(
        _classes={
            Ident.builtin("i64"): ClassDecl(),
            Ident.builtin("Map"): ClassDecl(type_vars=(TypeVarRef(Ident.builtin("K")), TypeVarRef(Ident.builtin("V")))),
        }
    )
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("i64")))
    Map = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("Map")))
    assert str(i64) == "i64"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


def test_function_call():
    decls = Declarations(
        _classes={
            Ident.builtin("i64"): ClassDecl(),
        },
        _functions={
            Ident.builtin("one"): FunctionDecl(FunctionSignature(return_type=TypeRefWithVars(Ident.builtin("i64")))),
        },
    )
    one = RuntimeFunction(Thunk.value(decls), Thunk.value(FunctionRef(Ident.builtin("one"))))
    assert (
        one().__egg_typed_expr__  # type: ignore[union-attr]
        == TypedExprDecl(JustTypeRef(Ident.builtin("i64")), CallDecl(FunctionRef(Ident.builtin("one"))))
    )


def test_classmethod_call():
    K, V = TypeVarRef(Ident.builtin("K")), TypeVarRef(Ident.builtin("V"))
    decls = Declarations(
        _classes={
            Ident.builtin("i64"): ClassDecl(),
            Ident.builtin("unit"): ClassDecl(),
            Ident.builtin("Map"): ClassDecl(
                type_vars=(K, V),
                class_methods={
                    "create": FunctionDecl(FunctionSignature(return_type=TypeRefWithVars(Ident.builtin("Map"), (K, V))))
                },
            ),
        },
    )
    Map = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("Map")))
    with pytest.raises(TypeConstraintError):
        Map.create()
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("i64")))
    unit = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("unit")))
    assert (
        Map[i64, unit].create().__egg_typed_expr__  # type: ignore[union-attr]
        == TypedExprDecl(
            JustTypeRef(Ident.builtin("Map"), (JustTypeRef(Ident.builtin("i64")), JustTypeRef(Ident.builtin("unit")))),
            CallDecl(
                ClassMethodRef(Ident.builtin("Map"), "create"),
                (),
                (JustTypeRef(Ident.builtin("i64")), JustTypeRef(Ident.builtin("unit"))),
            ),
        )
    )


def test_expr_special():
    decls = Declarations(
        _classes={
            Ident.builtin("i64"): ClassDecl(
                methods={
                    "__add__": FunctionDecl(
                        FunctionSignature(
                            (TypeRefWithVars(Ident.builtin("i64")), TypeRefWithVars(Ident.builtin("i64"))),
                            ("a", "b"),
                            (None, None),
                            TypeRefWithVars(Ident.builtin("i64")),
                        )
                    )
                },
                class_methods={
                    "__init__": FunctionDecl(
                        FunctionSignature(
                            (TypeRefWithVars(Ident.builtin("i64")),),
                            ("self",),
                            (None,),
                            TypeRefWithVars(Ident.builtin("i64")),
                        )
                    )
                },
            ),
        },
    )
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("i64")))
    one = i64(1)
    res = one + one  # type: ignore[operator]
    assert res.__egg_typed_expr__ == TypedExprDecl(
        JustTypeRef(Ident.builtin("i64")),
        CallDecl(
            MethodRef(Ident.builtin("i64"), "__add__"),
            (
                TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(1)),
                TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(1)),
            ),
        ),
    )


def test_class_variable():
    decls = Declarations(
        _classes={
            Ident.builtin("i64"): ClassDecl(
                class_variables={"one": ConstantDecl(JustTypeRef(Ident.builtin("i64")), None)}
            ),
        },
    )
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("i64")))
    one = i64.one
    assert isinstance(one, RuntimeExpr)
    assert one.__egg_typed_expr__ == TypedExprDecl(
        JustTypeRef(Ident.builtin("i64")), CallDecl(ClassVariableRef(Ident.builtin("i64"), "one"))
    )


def test_runtime_class_attr_lookup_is_stable():
    assert array_api.TupleInt.__getitem__ is array_api.TupleInt.__getitem__
    assert array_api.TupleInt.__dict__["__getitem__"] is array_api.TupleInt.__dict__["__getitem__"]


def test_doctest_finder_collects_runtime_function_docstrings():
    names = {test.name for test in doctest.DocTestFinder().find(array_api)}
    assert {
        "egglog.exp.array_api.TupleInt.__getitem__",
        "egglog.exp.array_api.TupleInt.if_",
        "egglog.exp.array_api.TupleTupleInt.product",
        "egglog.exp.array_api.Value.diff",
        "egglog.exp.array_api.RecursiveValue.__getitem__",
    } <= names
