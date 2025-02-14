from __future__ import annotations

import pytest

from egglog.declarations import *
from egglog.runtime import *
from egglog.thunk import *
from egglog.type_constraint_solver import *


def test_type_str():
    decls = Declarations(
        _classes={
            "i64": ClassDecl(),
            "Map": ClassDecl(type_vars=(ClassTypeVarRef("K", __name__), ClassTypeVarRef("V", __name__))),
        }
    )
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars("i64"))
    Map = RuntimeClass(Thunk.value(decls), TypeRefWithVars("Map"))
    assert str(i64) == "i64"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


def test_function_call():
    decls = Declarations(
        _classes={
            "i64": ClassDecl(),
        },
        _functions={
            "one": FunctionDecl(FunctionSignature(return_type=TypeRefWithVars("i64"))),
        },
    )
    one = RuntimeFunction(Thunk.value(decls), Thunk.value(FunctionRef("one")))
    assert (
        one().__egg_typed_expr__  # type: ignore[union-attr]
        == TypedExprDecl(JustTypeRef("i64"), CallDecl(FunctionRef("one")))
    )


def test_classmethod_call():
    K, V = ClassTypeVarRef("K", __name__), ClassTypeVarRef("V", __name__)
    decls = Declarations(
        _classes={
            "i64": ClassDecl(),
            "unit": ClassDecl(),
            "Map": ClassDecl(
                type_vars=(K, V),
                class_methods={"create": FunctionDecl(FunctionSignature(return_type=TypeRefWithVars("Map", (K, V))))},
            ),
        },
    )
    Map = RuntimeClass(Thunk.value(decls), TypeRefWithVars("Map"))
    with pytest.raises(TypeConstraintError):
        Map.create()  # type: ignore[operator]
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars("i64"))
    unit = RuntimeClass(Thunk.value(decls), TypeRefWithVars("unit"))
    assert (
        Map[i64, unit].create().__egg_typed_expr__  # type: ignore[union-attr, operator]
        == TypedExprDecl(
            JustTypeRef("Map", (JustTypeRef("i64"), JustTypeRef("unit"))),
            CallDecl(ClassMethodRef("Map", "create"), (), (JustTypeRef("i64"), JustTypeRef("unit"))),
        )
    )


def test_expr_special():
    decls = Declarations(
        _classes={
            "i64": ClassDecl(
                methods={
                    "__add__": FunctionDecl(
                        FunctionSignature(
                            (TypeRefWithVars("i64"), TypeRefWithVars("i64")),
                            ("a", "b"),
                            (None, None),
                            TypeRefWithVars("i64"),
                        )
                    )
                },
                class_methods={
                    "__init__": FunctionDecl(
                        FunctionSignature((TypeRefWithVars("i64"),), ("self",), (None,), TypeRefWithVars("i64"))
                    )
                },
            ),
        },
    )
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars("i64"))
    one = i64(1)
    res = one + one  # type: ignore[operator]
    assert res.__egg_typed_expr__ == TypedExprDecl(
        JustTypeRef("i64"),
        CallDecl(
            MethodRef("i64", "__add__"),
            (TypedExprDecl(JustTypeRef("i64"), LitDecl(1)), TypedExprDecl(JustTypeRef("i64"), LitDecl(1))),
        ),
    )


def test_class_variable():
    decls = Declarations(
        _classes={
            "i64": ClassDecl(class_variables={"one": ConstantDecl(JustTypeRef("i64"), None)}),
        },
    )
    i64 = RuntimeClass(Thunk.value(decls), TypeRefWithVars("i64"))
    one = i64.one
    assert isinstance(one, RuntimeExpr)
    assert one.__egg_typed_expr__ == TypedExprDecl(JustTypeRef("i64"), CallDecl(ClassVariableRef("i64", "one")))
