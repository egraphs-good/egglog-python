from __future__ import annotations

from egg_smol.declarations import *
from egg_smol.runtime import *
from egg_smol.type_constraint_solver import *


def test_type_str():
    decls = Declarations(
        classes={
            "i64": ClassDecl(),
            "Map": ClassDecl(n_type_vars=2),
        }
    )
    i64 = RuntimeClass(decls, "i64")
    Map = RuntimeClass(decls, "Map")
    assert str(i64) == "i64"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


def test_function_call():
    decls = Declarations(
        classes={
            "i64": ClassDecl(),
        },
        functions={
            "one": FunctionDecl(
                (),
                TypeRefWithVars("i64"),
            ),
        },
    )
    one = RuntimeFunction(decls, "one")
    assert one().__egg_parts__ == RuntimeExpr(decls, JustTypeRef("i64"), CallDecl(FunctionRef("one"))).__egg_parts__


def test_classmethod_call():
    from pytest import raises

    K, V = ClassTypeVarRef(0), ClassTypeVarRef(1)
    decls = Declarations(
        classes={
            "i64": ClassDecl(),
            "unit": ClassDecl(),
            "Map": ClassDecl(n_type_vars=2),
        },
        functions={
            "create": FunctionDecl(
                (),
                TypeRefWithVars("Map", (K, V)),
            )
        },
    )
    Map = RuntimeClass(decls, "Map")
    with raises(TypeConstraintError):
        Map.create()
    i64 = RuntimeClass(decls, "i64")
    unit = RuntimeClass(decls, "unit")
    assert (
        Map[i64, unit].create().__egg_parts__
        == RuntimeExpr(
            decls,
            JustTypeRef("Map", (JustTypeRef("i64"), JustTypeRef("unit"))),
            CallDecl(
                ClassMethodRef("Map", "create"),
                (),
                (JustTypeRef("i64"), JustTypeRef("unit")),
            ),
        ).__egg_parts__
    )


def test_expr_special():
    decls = Declarations(
        classes={
            "i64": ClassDecl(
                methods={
                    "__add__": FunctionDecl(
                        (TypeRefWithVars("i64"), TypeRefWithVars("i64")),
                        TypeRefWithVars("i64"),
                    )
                },
                class_methods={
                    "__init__": FunctionDecl(
                        (TypeRefWithVars("i64"),),
                        TypeRefWithVars("i64"),
                    )
                },
            ),
        },
    )
    i64 = RuntimeClass(decls, "i64")
    one = i64(1)  # type: ignore
    res = one + one  # type: ignore
    expected_res = RuntimeExpr(
        decls,
        JustTypeRef("i64"),
        CallDecl(MethodRef("i64", "__add__"), (LitDecl(1), LitDecl(1))),
    )
    assert res.parts == expected_res.__egg_parts__
