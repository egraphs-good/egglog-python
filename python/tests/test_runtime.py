from __future__ import annotations

from egglog.declarations import *
from egglog.runtime import *
from egglog.type_constraint_solver import *


def test_type_str():
    decls = ModuleDeclarations(
        Declarations(
            _classes={
                "i64": ClassDecl(),
                "Map": ClassDecl(n_type_vars=2),
            }
        )
    )
    i64 = RuntimeClass(decls, "i64")
    Map = RuntimeClass(decls, "Map")
    assert str(i64) == "i64"
    assert str(Map[i64, i64]) == "Map[i64, i64]"


def test_function_call():
    decls = ModuleDeclarations(
        Declarations(
            _classes={
                "i64": ClassDecl(),
            },
            _functions={
                "one": FunctionDecl(
                    (),
                    (),
                    (),
                    TypeRefWithVars("i64"),
                    False,
                ),
            },
        )
    )
    one = RuntimeFunction(decls, "one")
    assert (
        one().__egg_typed_expr__  # type: ignore
        == RuntimeExpr(decls, TypedExprDecl(JustTypeRef("i64"), CallDecl(FunctionRef("one")))).__egg_typed_expr__
    )


def test_classmethod_call():
    from pytest import raises

    K, V = ClassTypeVarRef(0), ClassTypeVarRef(1)
    decls = ModuleDeclarations(
        Declarations(
            _classes={
                "i64": ClassDecl(),
                "unit": ClassDecl(),
                "Map": ClassDecl(
                    n_type_vars=2,
                    class_methods={
                        "create": FunctionDecl(
                            (),
                            (),
                            (),
                            TypeRefWithVars("Map", (K, V)),
                            False,
                        )
                    },
                ),
            }
        )
    )
    Map = RuntimeClass(decls, "Map")
    with raises(TypeConstraintError):
        Map.create()  # type: ignore
    i64 = RuntimeClass(decls, "i64")
    unit = RuntimeClass(decls, "unit")
    assert (
        Map[i64, unit].create().__egg_typed_expr__  # type: ignore
        == RuntimeExpr(
            decls,
            TypedExprDecl(
                JustTypeRef("Map", (JustTypeRef("i64"), JustTypeRef("unit"))),
                CallDecl(
                    ClassMethodRef("Map", "create"),
                    (),
                    (JustTypeRef("i64"), JustTypeRef("unit")),
                ),
            ),
        ).__egg_typed_expr__
    )


def test_expr_special():
    decls = ModuleDeclarations(
        Declarations(
            _classes={
                "i64": ClassDecl(
                    methods={
                        "__add__": FunctionDecl(
                            (TypeRefWithVars("i64"), TypeRefWithVars("i64")),
                            (),
                            (None, None),
                            TypeRefWithVars("i64"),
                            False,
                        )
                    },
                    class_methods={
                        "__init__": FunctionDecl(
                            (TypeRefWithVars("i64"),),
                            (),
                            (None,),
                            TypeRefWithVars("i64"),
                            False,
                        )
                    },
                ),
            },
        )
    )
    i64 = RuntimeClass(decls, "i64")
    one = i64(1)  # type: ignore
    res = one + one  # type: ignore
    expected_res = RuntimeExpr(
        decls,
        TypedExprDecl(
            JustTypeRef("i64"),
            CallDecl(
                MethodRef("i64", "__add__"),
                (TypedExprDecl(JustTypeRef("i64"), LitDecl(1)), TypedExprDecl(JustTypeRef("i64"), LitDecl(1))),
            ),
        ),
    )
    assert res.__egg_typed_expr__ == expected_res.__egg_typed_expr__


def test_class_variable():
    decls = ModuleDeclarations(
        Declarations(
            _classes={
                "i64": ClassDecl(class_variables={"one": JustTypeRef("i64")}),
            },
        )
    )
    i64 = RuntimeClass(decls, "i64")
    one = i64.one
    assert isinstance(one, RuntimeExpr)
    assert (
        one.__egg_typed_expr__
        == RuntimeExpr(
            decls, TypedExprDecl(JustTypeRef("i64"), CallDecl(ClassVariableRef("i64", "one")))
        ).__egg_typed_expr__
    )
