from __future__ import annotations

import pytest

from egglog.declarations import *
from egglog.runtime import *
from egglog.thunk import *
from egglog.type_constraint_solver import *


def test_type_str():
    decls = Declarations(
        _classes={
            Ident.builtin("i64"): ClassDecl(),
            Ident.builtin("Map"): ClassDecl(type_vars=(ClassTypeVarRef(Ident.builtin("K")), ClassTypeVarRef(Ident.builtin("V")))),
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
    K, V = ClassTypeVarRef(Ident.builtin("K")), ClassTypeVarRef(Ident.builtin("V"))
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
                            (TypeRefWithVars(Ident.builtin("i64")),), ("self",), (None,), TypeRefWithVars(Ident.builtin("i64"))
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
            (TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(1)), TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(1))),
        ),
    )


def test_class_variable():
    decls = Declarations(
        _classes={
            Ident.builtin("i64"): ClassDecl(class_variables={"one": ConstantDecl(JustTypeRef(Ident.builtin("i64")), None)}),
        },
    )
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars(Ident.builtin("i64")))
    one = i64.one
    assert isinstance(one, RuntimeExpr)
    assert one.__egg_typed_expr__ == TypedExprDecl(JustTypeRef(Ident.builtin("i64")), CallDecl(ClassVariableRef(Ident.builtin("i64"), "one")))
