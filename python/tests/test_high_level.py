# mypy: disable-error-code="empty-body"
from __future__ import annotations

import importlib
import pathlib
from collections.abc import Callable, Iterator
from copy import copy
from fractions import Fraction
from functools import partial
from typing import ClassVar, TypeAlias, TypeVar, cast
from unittest.mock import MagicMock

import pytest

import egglog.builtins as egg_builtins
from egglog import *
from egglog.declarations import (
    BUILTIN_EGG_FN_NAMES,
    BUILTIN_EGG_SORT_NAMES,
    CallableDecl,
    CallDecl,
    ClassDecl,
    Declarations,
    FunctionDecl,
    FunctionRef,
    FunctionSignature,
    HasDeclarations,
    Ident,
    JustTypeRef,
    MethodRef,
    TypedExprDecl,
    TypeRefWithVars,
)
from egglog.runtime import RuntimeExpr, RuntimeFunction


class TestExprStr:
    def test_unwrap_lit(self):
        assert str(i64(1) + 1) == "i64(1) + 1"
        assert str(i64(1).max(2)) == "i64(1).max(2)"

    def test_ne(self):
        assert str(ne(i64(1)).to(i64(2))) == "ne(i64(1)).to(i64(2))"


def test_eqsat_basic():
    egraph = EGraph()

    class Math(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @classmethod
        def var(cls, v: StringLike) -> Math: ...

        def __add__(self, other: Math) -> Math: ...

        def __mul__(self, other: Math) -> Math: ...

    # expr1 = 2 * (x + 3)
    expr1 = egraph.let("expr1", Math(2) * (Math.var("x") + Math(3)))

    # expr2 = 6 + 2 * x
    expr2 = egraph.let("expr2", Math(6) + Math(2) * Math.var("x"))

    a, b, c = vars_("a b c", Math)
    x, y = vars_("x y", i64)

    egraph.register(
        rewrite(a + b).to(b + a),
        rewrite(a * (b + c)).to((a * b) + (a * c)),
        rewrite(Math(x) + Math(y)).to(Math(x + y)),
        rewrite(Math(x) * Math(y)).to(Math(x * y)),
    )

    egraph.run(10)

    egraph.check(eq(expr1).to(expr2))


def test_let_auto_prefixes_global_names(capfd: pytest.CaptureFixture[str]):
    egraph = EGraph(save_egglog_string=True)

    x = egraph.let("x", i64(1))
    egraph.check(eq(x).to(i64(1)))

    captured = capfd.readouterr()
    assert "should start with `$`" not in captured.err
    assert "(let $x " in egraph.as_egglog_string


def test_synthetic_lets_use_reserved_expr_names() -> None:
    class LetNum(Expr):
        @classmethod
        def var(cls, v: StringLike) -> LetNum: ...

    egraph = EGraph(save_egglog_string=True)
    expr = LetNum.var("x")
    egraph._add_decls(expr)

    egraph._state._transform_let(expr.__egg_typed_expr__)

    assert '(let $__expr_0 (LetNum_var "x"))' in egraph.as_egglog_string


def test_synthetic_lets_skip_explicit_let_conflicts() -> None:
    class LetConflictNum(Expr):
        @classmethod
        def var(cls, v: StringLike) -> LetConflictNum: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.let("__expr_0", LetConflictNum.var("explicit"))
    expr = LetConflictNum.var("synthetic")
    egraph._add_decls(expr)

    egraph._state._transform_let(expr.__egg_typed_expr__)

    egglog_string = egraph.as_egglog_string
    assert '(let $__expr_0 (LetConflictNum_var "explicit"))' in egglog_string
    assert '(let $__expr_1 (LetConflictNum_var "synthetic"))' in egglog_string


def test_synthetic_let_names_do_not_shadow_default_rewrite_variables() -> None:
    default_ruleset = ruleset(name="synthetic-let-shadow-default-rewrite")

    class LetShadowDefaultNum(Expr, ruleset=default_ruleset):
        def __init__(self, value: i64Like) -> None: ...

        @classmethod
        def make(cls, value: i64Like) -> LetShadowDefaultNum:
            return LetShadowDefaultNum(value)

    egraph = EGraph(save_egglog_string=True)
    expr = LetShadowDefaultNum(3)
    egraph._add_decls(expr)
    egraph._state._transform_let(expr.__egg_typed_expr__)

    egraph.register(LetShadowDefaultNum.make(i64(1)))
    egraph.run(run(default_ruleset))

    egglog_string = egraph.as_egglog_string
    assert "(let $__expr_0 (LetShadowDefaultNum___init__ 3))" in egglog_string
    assert "(rewrite (LetShadowDefaultNum_make _0) (LetShadowDefaultNum___init__ _0)" in egglog_string


def test_save_egglog_string_defaults_to_file_backed() -> None:
    egraph = EGraph()

    assert egraph.as_egglog_string == ""
    assert egraph._state.egglog_file_state is not None
    assert egraph._state.egglog_file_state.path.endswith(".egg")
    assert pathlib.Path(egraph._state.egglog_file_state.path).exists()


def test_saved_egglog_string_uses_short_generated_sort_and_function_names() -> None:
    class Num(Expr):
        @classmethod
        def var(cls, v: StringLike) -> Num: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(Num.var("x"))
    egglog_string = egraph.as_egglog_string

    assert "(sort Num)" in egglog_string
    assert "(constructor Num_var (String) Num)" in egglog_string
    assert "test_high_level" not in egglog_string


def test_generated_names_fall_back_to_full_name_on_conflict() -> None:
    state = EGraph(save_egglog_string=True)._state
    ret1 = Ident("Ret", "pkg.one")
    ret2 = Ident("Ret", "pkg.two")
    fn1 = Ident("make", "pkg.one")
    fn2 = Ident("make", "pkg.two")
    state.__egg_decls__ |= Declarations(
        _classes={ret1: ClassDecl(), ret2: ClassDecl()},
        _functions={
            fn1: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret1))),
            fn2: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret2))),
        },
    )

    assert state.callable_ref_to_egg(FunctionRef(fn1))[0] == "make"
    assert state.callable_ref_to_egg(FunctionRef(fn2))[0] == "pkg_two_make"
    assert state.type_ref_to_egg(JustTypeRef(ret1)) == "Ret"
    assert state.type_ref_to_egg(JustTypeRef(ret2)) == "pkg.two.Ret"


def test_missing_function_lookup_does_not_reserve_generated_name() -> None:
    state = EGraph(save_egglog_string=True)._state
    ret = Ident("LookupRet", "pkg.lookup")
    fn = Ident("lookup_short_name", "pkg.lookup")
    state.__egg_decls__ |= Declarations(
        _classes={ret: ClassDecl()},
        _functions={fn: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret)))},
    )

    assert list(state.possible_egglog_functions(["lookup_short_name"])) == []
    assert state.callable_ref_to_egg(FunctionRef(fn))[0] == "lookup_short_name"


def test_generated_names_fall_back_from_builtin_names() -> None:
    state = EGraph(save_egglog_string=True)._state
    ret = Ident("BuiltinConflictRet", "pkg.builtin_conflict")
    fn = Ident("exp", "pkg.builtin_conflict")
    sort = Ident("Map", "pkg.builtin_conflict")
    state.__egg_decls__ |= Declarations(
        _classes={ret: ClassDecl(), sort: ClassDecl()},
        _functions={fn: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret)))},
    )

    assert state.callable_ref_to_egg(FunctionRef(fn))[0] == "pkg_builtin_conflict_exp"
    assert state.type_ref_to_egg(JustTypeRef(sort)) == "pkg.builtin_conflict.Map"


def test_builtin_name_reservations_cover_builtins_module_declarations() -> None:
    expected_fn_names = set[str]()
    expected_sort_names = set[str]()

    def add_callable_name(decl: CallableDecl | None) -> None:
        if decl is not None and decl.egg_name is not None:
            expected_fn_names.add(decl.egg_name)

    for name in egg_builtins.__all__:
        obj = getattr(egg_builtins, name, None)
        if not isinstance(obj, HasDeclarations):
            continue
        decls = obj.__egg_decls__
        for decl in decls._functions.values():
            add_callable_name(decl)
        for decl in decls._constants.values():
            add_callable_name(decl)
        for decl in decls._classes.values():
            if decl.builtin and decl.egg_name is not None:
                expected_sort_names.add(decl.egg_name)
            add_callable_name(decl.init)
            for callable_decl in (
                *decl.class_methods.values(),
                *decl.class_variables.values(),
                *decl.methods.values(),
                *decl.properties.values(),
            ):
                add_callable_name(callable_decl)

    assert expected_fn_names <= BUILTIN_EGG_FN_NAMES
    assert expected_sort_names <= BUILTIN_EGG_SORT_NAMES


def test_parameterized_sort_names_use_allocated_argument_names() -> None:
    egraph = EGraph(save_egglog_string=True)

    egraph.register(Map[i64, BigRat].empty())

    assert "(sort Map[i64,BigRat] (Map i64 BigRat))" in egraph.as_egglog_string


def test_non_constructor_map_empty_does_not_create_synthetic_let() -> None:
    egraph = EGraph(save_egglog_string=True)
    expr = Map[Map[String, i64], f64].empty()
    egraph._add_decls(expr)

    assert egraph._state._transform_let(expr.__egg_typed_expr__) is not None

    lines = egraph.as_egglog_string.splitlines()
    assert "(let $__expr_0 (map-empty))" not in lines
    assert not any(line.startswith("(fail ") for line in lines)


def test_non_constructor_maybe_none_does_not_create_synthetic_let() -> None:
    egraph = EGraph(save_egglog_string=True)
    expr = Maybe[Maybe[i64]].none()
    egraph._add_decls(expr)

    assert egraph._state._transform_let(expr.__egg_typed_expr__) is not None

    lines = egraph.as_egglog_string.splitlines()
    assert "(let $__expr_0 (maybe-none))" not in lines
    assert not any(line.startswith("(fail ") for line in lines)


def test_inferable_non_constructor_map_empty_does_not_create_synthetic_let() -> None:
    egraph = EGraph(save_egglog_string=True)
    expr = Map[String, i64].empty()
    egraph._add_decls(expr)

    assert egraph._state._transform_let(expr.__egg_typed_expr__) is not None

    lines = egraph.as_egglog_string.splitlines()
    assert "(let $__expr_0 (map-empty))" not in lines
    assert not any(line.startswith("(fail ") for line in lines)


def test_freeze_omits_synthetic_let_bindings() -> None:
    class FreezeLetNum(Expr):
        @classmethod
        def var(cls, v: StringLike) -> FreezeLetNum: ...

    egraph = EGraph(save_egglog_string=True)
    expr = FreezeLetNum.var("x")
    egraph._add_decls(expr)
    egraph._state._transform_let(expr.__egg_typed_expr__)

    assert "(let $__expr_0 " in egraph.as_egglog_string
    assert "$__expr_0" not in str(egraph.freeze())


def test_popped_explicit_lets_do_not_block_synthetic_let_names() -> None:
    class ScopedLetNum(Expr):
        @classmethod
        def var(cls, v: StringLike) -> ScopedLetNum: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.push()
    egraph.let("__expr_0", ScopedLetNum.var("pushed"))
    egraph.pop()
    expr = ScopedLetNum.var("synthetic")
    egraph._add_decls(expr)

    egraph._state._transform_let(expr.__egg_typed_expr__)

    assert '(let $__expr_0 (ScopedLetNum_var "synthetic"))' in egraph.as_egglog_string


def test_as_egglog_string_requires_flag() -> None:
    egraph = EGraph(save_egglog_string=False)

    with pytest.raises(ValueError, match="save_egglog_string=True"):
        _ = egraph.as_egglog_string
    assert egraph._state.egglog_file_state is None


def test_registering_bare_variable_expression_raises() -> None:
    egraph = EGraph()

    with pytest.raises(ValueError, match="must be calls"):
        egraph.register(var("x", i64))


def test_registering_let_reference_expression_raises() -> None:
    egraph = EGraph()
    x = egraph.let("x", i64(1))

    with pytest.raises(ValueError, match="must be calls"):
        egraph.register(x)


@pytest.mark.parametrize("save_egglog_string", [True, False])
def test_nested_rule_lowering_does_not_reuse_top_level_synthetic_lets(save_egglog_string: bool) -> None:
    egraph = EGraph(save_egglog_string=save_egglog_string)
    pair_rel = relation(f"pair_rel_ctx_{int(save_egglog_string)}", i64, i64)
    done_rel = relation(f"done_rel_ctx_{int(save_egglog_string)}", i64)
    repeated = i64(1) + i64(2)

    egraph.register(pair_rel(repeated, repeated))
    egraph.register(rule(pair_rel(repeated, repeated)).then(done_rel(i64(0))))

    egraph.run(1)
    egraph.check(done_rel(i64(0)))


def test_anonymous_combined_rulesets_use_deterministic_generated_names() -> None:
    first = ruleset(name="combined_name_probe_first")
    second = ruleset(name="combined_name_probe_second")
    combined = unstable_combine_rulesets(first, second)
    egraph = EGraph(save_egglog_string=True)

    egraph.run(combined)

    combined_line = next(
        line for line in egraph.as_egglog_string.splitlines() if line.startswith("(unstable-combined-ruleset ")
    )
    combined_name = combined_line.split()[1]
    assert combined_name.startswith("_combined_ruleset_")
    assert combined_name.removeprefix("_combined_ruleset_").isdigit()
    assert f"(run-schedule (run {combined_name}))" in egraph.as_egglog_string


def test_higher_order_builtin_callback_materializes_scalar_builtin_type_args() -> None:
    check_eq(map_fold_kv(lambda acc, k, v: acc + v, f64(0.0), Map[i64, f64].empty()), f64(0.0))


def test_higher_order_builtin_callback_materializes_parameterized_builtin_dummy_args() -> None:
    input_map = Map[i64, Maybe[f64]].empty().insert(i64(1), Maybe[f64].some(f64(2.5)))
    expected = Map[i64, f64].empty().insert(i64(1), f64(2.5))
    check_eq(map_map_values(lambda k, v: v.unwrap(), input_map), expected)


def test_higher_order_builtin_callback_materializes_rational_builtin_dummy_args() -> None:
    input_map = Map[i64, Rational].empty().insert(i64(1), Rational(1, 2))
    expected = Map[i64, f64].empty().insert(i64(1), f64(1.5))
    check_eq(map_map_values(lambda k, v: v.to_f64() + 1.0, input_map), expected)


def test_file_backed_errors_report_saved_file_line() -> None:
    egraph = EGraph()
    egraph.let("x", i64(1))
    egraph.let("y", i64(2))
    expected_line = len(egraph.as_egglog_string.splitlines()) + 1
    assert egraph._state.egglog_file_state is not None
    path = egraph._state.egglog_file_state.path

    with pytest.raises(EggSmolError) as exc_info:
        egraph.check(eq(i64(1)).to(i64(2)))

    error_text = exc_info.value.context
    assert path in error_text
    assert f"In {expected_line}:" in error_text
    lines = egraph.as_egglog_string.splitlines()
    assert "(fail (check (= 1 2))) ; Check failed:" in lines
    assert "(check (= 1 2))" not in lines


def test_unnamed_lambda_returning_builtin_is_eager() -> None:
    check_eq(
        map_fold_kv(
            lambda acc, k, v: acc + v,
            f64(0.0),
            Map[i64, f64].empty().insert(i64(1), f64(2.0)).insert(i64(2), f64(3.5)),
        ),
        f64(5.5),
    )


def test_unnamed_lambda_returning_eqsort_is_eager() -> None:
    class Box(Expr):
        def __init__(self, value: i64Like) -> None: ...

    expected = Map[i64, Box].empty().insert(i64(1), Box(i64(2)))
    check_eq(map_map_values(lambda k, v: Box(v), Map[i64, i64].empty().insert(i64(1), i64(2))), expected)


def test_named_builtin_return_function_is_eager() -> None:
    @function
    def add1(x: i64Like) -> i64:
        return x + 1

    check_eq(add1(i64(2)), i64(3))


def test_named_container_return_function_is_eager() -> None:
    @function
    def singleton_map(x: i64Like) -> Map[i64, i64]:
        return Map[i64, i64].empty().insert(x, x + 1)

    expected = Map[i64, i64].empty().insert(i64(2), i64(3))
    check_eq(singleton_map(i64(2)), expected)


def test_named_builtin_return_function_with_eqsort_input_is_eager() -> None:
    class Box(Expr):
        def __init__(self, value: i64Like) -> None: ...

    @function
    def box_score(box: Box) -> i64: ...

    @function
    def via_box(box: Box) -> i64:
        return box_score(box) + 1

    egraph = EGraph()
    egraph.register(set_(box_score(Box(i64(4)))).to(i64(9)))
    egraph.check(eq(via_box(Box(i64(4)))).to(i64(10)))


def test_named_builtin_return_function_with_none_body_stays_plain_function() -> None:
    @function
    def maybe_missing(x: i64Like) -> i64:
        return None

    egraph = EGraph()
    egraph.check_fail(eq(maybe_missing(i64(1))).to(i64(1)))
    egraph.register(set_(maybe_missing(i64(1))).to(i64(4)))
    egraph.check(eq(maybe_missing(i64(1))).to(i64(4)))


def test_named_eqsort_function_body_is_eager() -> None:
    class Box(Expr):
        def __init__(self, value: i64Like) -> None: ...

    @function
    def make_box(x: i64Like) -> Box:
        return Box(x)

    check_eq(make_box(i64(4)), Box(i64(4)))


def test_missing_function_row_inside_primitive_body_stays_undefined() -> None:
    @function
    def f_lookup(x: i64Like) -> i64: ...

    @function
    def via_lookup(x: i64Like) -> i64:
        return f_lookup(x) + 1

    egraph = EGraph()
    egraph.register(set_(f_lookup(i64(4))).to(i64(9)))
    egraph.check(eq(via_lookup(i64(4))).to(i64(10)))
    egraph.check_fail(eq(via_lookup(i64(5))).to(i64(0)))


def test_fib():
    egraph = EGraph()

    @function
    def fib(x: i64Like) -> i64: ...

    f0, f1, x = vars_("f0 f1 x", i64)
    egraph.register(
        set_(fib(0)).to(i64(1)),
        set_(fib(1)).to(i64(1)),
        rule(
            eq(f0).to(fib(x)),
            eq(f1).to(fib(x + 1)),
        ).then(set_(fib(x + 2)).to(f0 + f1)),
    )
    egraph.run(7)
    egraph.check(eq(fib(i64(7))).to(i64(21)))


def test_fib_demand():
    egraph = EGraph()

    class Num(Expr):
        def __init__(self, i: i64Like) -> None: ...

        def __add__(self, other: Num) -> Num: ...

    @function(cost=20)
    def fib(x: i64Like) -> Num: ...

    @egraph.register
    def _fib(a: i64, b: i64):
        yield rewrite(Num(a) + Num(b)).to(Num(a + b))
        yield rewrite(fib(a)).to(fib(a - 1) + fib(a - 2), a > 1)
        yield rewrite(fib(a)).to(Num(a), a <= 1)

    f7 = egraph.let("f7", fib(7))
    egraph.run(14)
    egraph.check(eq(f7).to(Num(13)))
    res = egraph.extract(f7)
    assert expr_parts(res) == expr_parts(Num(13))


def test_push_pop():
    egraph = EGraph()

    @function(merge=lambda old, new: old.max(new))
    def foo() -> i64: ...

    egraph.register(set_(foo()).to(i64(1)))
    egraph.check(eq(foo()).to(i64(1)))

    with egraph:
        egraph.register(set_(foo()).to(i64(2)))
        egraph.check(eq(foo()).to(i64(2)))

    egraph.check(eq(foo()).to(i64(1)))


def test_constants():
    egraph = EGraph()

    class A(Expr):
        pass

    one = constant("one", A)
    two = constant("two", A)

    egraph.register(union(one).with_(two))
    egraph.check(eq(one).to(two))


def test_class_vars():
    egraph = EGraph()

    class B(Expr):
        ONE: ClassVar[B]

    two = constant("two", B)

    egraph.register(union(B.ONE).with_(two))
    egraph.check(eq(B.ONE).to(two))


def test_extract_constant_twice():
    # Sometimes extracting a constant twice will give an error
    egraph = EGraph()

    class Numeric(Expr):
        ONE: ClassVar[Numeric]

    egraph.extract(Numeric.ONE)
    egraph.extract(Numeric.ONE)


def test_extract_include_cost():
    _, cost = EGraph().extract(i64(0), include_cost=True)
    assert cost == 1


def test_egraph_constructor_registers_actions():
    class ConstructorExpr(Expr):
        def __init__(self) -> None: ...

    @function
    def constructor_cost() -> i64: ...

    @function
    def constructor_lhs() -> ConstructorExpr: ...

    @function
    def constructor_rhs() -> ConstructorExpr: ...

    constructed = EGraph(
        ConstructorExpr(), set_(constructor_cost()).to(i64(1)), eq(constructor_lhs()).to(constructor_rhs())
    )

    expected = EGraph()
    expected.register(
        ConstructorExpr(),
        set_(constructor_cost()).to(i64(1)),
        eq(constructor_lhs()).to(constructor_rhs()),
    )

    assert str(constructed.freeze()) == str(expected.freeze())


def test_relation():
    egraph = EGraph()

    test_relation = relation("test_relation", i64, i64)
    egraph.register(test_relation(i64(1), i64(1)))


def test_variable_args():
    egraph = EGraph()
    egraph.check(Set(i64(1), i64(2)).contains(i64(1)))


def test_generic_sort():
    egraph = EGraph()
    egraph.check(Set(i64(1), i64(2)).contains(i64(1)))


def test_keyword_args():
    EGraph()

    @function
    def foo(x: i64Like, y: i64Like) -> i64: ...

    pos = expr_parts(foo(i64(1), i64(2)))
    assert expr_parts(foo(i64(1), y=i64(2))) == pos
    assert expr_parts(foo(y=i64(2), x=i64(1))) == pos


def test_keyword_args_init():
    EGraph()

    class Foo(Expr):
        def __init__(self, x: i64Like) -> None: ...

    assert expr_parts(Foo(1)) == expr_parts(Foo(x=1))


def test_property():
    egraph = EGraph()

    class Foo(Expr):
        def __init__(self) -> None: ...

        @property
        def bar(self) -> i64: ...

    egraph.register(set_(Foo().bar).to(i64(1)))
    egraph.check(eq(Foo().bar).to(i64(1)))


def test_default_args():
    EGraph()

    @function
    def foo(x: i64Like, y: i64Like = i64(1)) -> i64: ...

    assert expr_parts(foo(i64(1))) == expr_parts(foo(i64(1), i64(1)))

    assert str(foo(i64(1), i64(2))) == "foo(1, 2)"
    assert str(foo(i64(1), i64(1))) == "foo(1)"


class TestPyObject:
    def test_from_string(self):
        assert EGraph().extract(PyObject.from_string("foo")).value == "foo"

    def test_to_string(self):
        EGraph().check(PyObject("foo").to_string() == String("foo"))

    def test_dict_update(self):
        original_d = {"foo": "bar"}
        res = EGraph().extract(PyObject(original_d).dict_update("foo", "baz")).value
        assert res == {"foo": "baz"}
        assert original_d == {"foo": "bar"}

    def test_eval(self):
        assert EGraph().extract(py_eval("x + y", {"x": 10, "y": 20}, {})).value == 30

    @pytest.mark.xfail(reason="cant pickle locals")
    def test_eval_local(self):
        x = "hi"
        res = py_eval("my_add(x, y)", PyObject(locals()).dict_update("y", "there"), globals())
        assert EGraph().extract(res).value == "hithere"

    def test_exec(self):
        assert EGraph().extract(py_exec("x = 10")).value == {"x": 10}

    def test_exec_globals(self):
        assert EGraph().extract(py_exec("x = y + 1", {"y": 10})).value == {"x": 11}


def my_add(a, b):
    return a + b


def test_convert_int_float():
    egraph = EGraph()
    egraph.check(eq(i64(1)).to(f64(1.0).to_i64()))
    egraph.check(eq(f64(1.0)).to(f64.from_i64(i64(1))))


def test_f64_negation() -> None:
    egraph = EGraph()
    # expr1 = -2.0
    expr1 = egraph.let("expr1", -f64(2.0))

    # expr2 = 2.0
    expr2 = egraph.let("expr2", f64(2.0))

    # expr3 = -(-2.0)
    expr3 = egraph.let("expr3", -(-f64(2.0)))  # noqa: B002
    egraph.check(eq(expr1).to(-expr2))
    egraph.check(eq(expr3).to(expr2))


def test_not_equals():
    egraph = EGraph()
    egraph.check(ne(i64(10)).to(i64(2)))


def test_custom_equality():
    egraph = EGraph()

    class Boolean(Expr):
        def __init__(self, value: BoolLike) -> None: ...

        def __eq__(self, other: Boolean) -> Boolean:  # type: ignore[override]
            ...

        def __ne__(self, other: Boolean) -> Boolean:  # type: ignore[override]
            ...

    egraph.register(rewrite(Boolean(True) == Boolean(True)).to(Boolean(False)))
    egraph.register(rewrite(Boolean(True) != Boolean(True)).to(Boolean(True)))

    should_be_true = Boolean(True) == Boolean(True)
    should_be_false = Boolean(True) != Boolean(True)
    egraph.register(should_be_true, should_be_false)
    egraph.run(10)
    egraph.check(eq(should_be_true).to(Boolean(False)))
    egraph.check(eq(should_be_false).to(Boolean(True)))


class TestMutate:
    def test_setitem_defaults(self):
        EGraph()

        class Foo(Expr):
            def __init__(self) -> None: ...
            def __setitem__(self, key: i64Like, value: i64Like) -> None: ...

        foo = Foo()
        foo[10] = 20
        assert str(foo) == "_Foo_1 = Foo()\n_Foo_1[10] = 20\n_Foo_1"
        assert expr_parts(foo) == TypedExprDecl(
            JustTypeRef(Ident("Foo", __name__)),
            CallDecl(
                MethodRef(Ident("Foo", __name__), "__setitem__"),
                (expr_parts(Foo()), expr_parts(i64(10)), expr_parts(i64(20))),
            ),
        )

    def test_function(self):
        egraph = EGraph()

        class Math(Expr):
            def __init__(self, i: i64Like) -> None: ...

            def __add__(self, other: Math) -> Math: ...

        @function(mutates_first_arg=True)
        def incr(x: Math) -> None: ...

        x = Math(i64(10))
        x_copied = copy(x)
        incr(x)
        assert expr_parts(x_copied) == expr_parts(Math(i64(10)))
        assert expr_parts(x) == TypedExprDecl(
            JustTypeRef(Ident("Math", __name__)),
            CallDecl(FunctionRef(Ident("incr", __name__)), (expr_parts(x_copied),)),
        )
        assert str(x) == "_Math_1 = Math(10)\nincr(_Math_1)\n_Math_1"
        assert str(x + Math(10)) == "_Math_1 = Math(10)\nincr(_Math_1)\n_Math_1 + Math(10)"

        i, _j = vars_("i j", Math)
        incr_i = copy(i)
        incr(incr_i)
        egraph.register(rewrite(incr_i).to(i + Math(1)), x)
        egraph.run(10)
        egraph.check(eq(x).to(Math(10) + Math(1)))

        x_incr_copy = copy(x)
        incr(x)
        assert (
            str(x_incr_copy + x)
            == "_Math_1 = Math(10)\nincr(_Math_1)\n_Math_2 = copy(_Math_1)\nincr(_Math_2)\n_Math_1 + _Math_2"
        ), "only copy when re-used later"


def test_builtin_reflected():
    assert expr_parts(5 + i64(10)) == expr_parts(i64(5) + i64(10))


def test_reflected_binary_method():
    # If we have a reflected binary method, it should be converted into the non-reflected version
    EGraph()

    class Math(Expr):
        def __init__(self, value: i64Like) -> None: ...

        def __add__(self, other: Math) -> Math: ...

        def __radd__(self, other: Math) -> Math: ...

    converter(i64, Math, Math)

    expr = 10 + Math(5)  # type: ignore[operator]
    assert str(expr) == "Math(10) + Math(5)"
    assert expr_parts(expr) == TypedExprDecl(
        JustTypeRef(Ident("Math", __name__)),
        CallDecl(MethodRef(Ident("Math", __name__), "__add__"), (expr_parts(Math(i64(10))), expr_parts(Math(i64(5))))),
    )


def test_rewrite_upcasts():
    class X(Expr):
        def __init__(self, value: i64Like) -> None: ...

    converter(i64, X, X)
    rewrite(X(1)).to(0)  # type: ignore[arg-type]


def test_function_default_upcasts():
    @function
    def f(x: i64Like) -> i64: ...

    assert expr_parts(f(1)) == expr_parts(f(i64(1)))


def test_upcast_self_lower_cost():
    # Verifies that self will be upcasted, if that upcast has a lower cast than converting the other arg
    # i.e. Int(x) + NDArray(y) -> NDArray(Int(x)) + NDArray(y) instead of Int(x) + NDArray(y).to_int()

    class Int(Expr):
        def __init__(self, name: StringLike) -> None: ...

        def __add__(self, other: Int) -> Int: ...

    class NDArray(Expr):
        def __init__(self, name: StringLike) -> None: ...

        def __add__(self, other: NDArrayLike) -> NDArray: ...

        def __radd__(self, other: NDArrayLike) -> NDArray: ...

        def to_int(self) -> Int: ...

        @classmethod
        def from_int(cls, other: Int) -> NDArray: ...

    NDArrayLike: TypeAlias = NDArray | Int

    converter(Int, NDArray, NDArray.from_int)
    converter(NDArray, Int, lambda a: a.to_int(), 100)

    r = Int("x") + NDArray("y")
    assert expr_parts(r) == expr_parts(NDArray.from_int(Int("x")) + NDArray("y"))


class TestEval:
    def test_string(self):
        assert String("hi").value == "hi"

    def test_bool(self):
        assert Bool(True).value is True
        assert bool(Bool(True)) is True

    def test_i64(self):
        assert i64(10).value == 10
        assert int(i64(10)) == 10
        assert [10][i64(0)] == 10

    def test_f64(self):
        assert f64(10.0).value == 10.0
        assert int(f64(10.0)) == 10
        assert float(f64(10.0)) == 10.0

    def test_map(self):
        assert Map[String, i64].empty().value == {}
        m = Map[String, i64].empty().insert(String("a"), i64(1)).insert(String("b"), i64(2))
        assert m.value == {String("a"): i64(1), String("b"): i64(2)}

        assert set(m) == {String("a"), String("b")}
        assert len(m) == 2
        assert String("a") in m
        assert String("c") not in m

    def test_set(self):
        assert EGraph().extract(Set[i64].empty()).value == set()
        s = Set(i64(1), i64(2))
        assert s.value == {i64(1), i64(2)}

        assert set(s) == {i64(1), i64(2)}
        assert len(s) == 2
        assert i64(1) in s
        assert i64(3) not in s

    def test_rational(self):
        assert Rational(1, 2).value == Fraction(1, 2)
        assert float(Rational(1, 2)) == 0.5
        assert int(Rational(1, 1)) == 1

    def test_vec(self):
        assert Vec[i64].empty().value == ()
        s = Vec(i64(1), i64(2))
        assert s.value == (i64(1), i64(2))

        assert list(s) == [i64(1), i64(2)]
        assert len(s) == 2
        assert i64(1) in s
        assert i64(3) not in s

    def test_py_object(self):
        assert PyObject(10).value == 10
        o = (1, 2, 3)
        assert PyObject(o).value == o

    def test_big_int(self):
        assert int(EGraph().extract(BigInt(10))) == 10

    def test_big_rat(self):
        br = EGraph().extract(BigRat(1, 2))
        assert float(br) == 1 / 2
        assert br.value == Fraction(1, 2)

    def test_extract_big_rat_from_f64(self):
        br = EGraph().extract(BigRat.from_f64(2.0))
        assert br.value == Fraction(2, 1)

    def test_extract_nested_maps_preserves_empty_map_type_params(self):
        inner = Map[String, BigRat].empty().insert("x", BigRat.from_f64(2.0))
        expr = Map[Map[String, BigRat], f64].empty().insert(inner, 1.0)

        extracted = EGraph().extract(expr)

        assert "Map[String, BigRat].empty().insert" in str(extracted)
        assert 'Map[Map[String, BigRat], f64].empty().insert(String("x")' not in str(extracted)

    def test_multiset(self):
        assert list(MultiSet(i64(1), i64(1))) == [i64(1), i64(1)]

    def test_unstable_fn(self):
        class Math(Expr):
            def __init__(self) -> None: ...

        @function
        def f(x: Math) -> Math: ...

        u_f = UnstableFn(f)
        assert u_f.value == f
        p_u_f = UnstableFn(f, Math())
        value = p_u_f.value
        assert isinstance(value, partial)
        assert value.func == f
        assert value.args == (Math(),)


# def test_egglog_string():
#     egraph = EGraph(save_egglog_string=True)
#     egraph.register((i64(1)))
#     assert egraph.as_egglog_string

# def test_no_egglog_string():
#     egraph = EGraph()
#     egraph.register((i64(1)))
#     with pytest.raises(ValueError):
#         egraph.as_egglog_string


def test_eval_fn():
    assert EGraph().extract(PyObject(lambda x: (x,))(PyObject.from_int(1))).value == (1,)


def _global_make_tuple(x):
    return (x,)


def test_eval_fn_globals():
    assert EGraph().extract(PyObject(lambda x: _global_make_tuple(x))(PyObject.from_int(1))).value == (1,)


def test_eval_fn_locals():
    def _locals_make_tuple(x):
        return (x,)

    assert EGraph().extract(PyObject(lambda x: _locals_make_tuple(x))(PyObject.from_int(1))).value == (1,)


def test_lazy_types():
    class A(Expr):
        def __init__(self) -> None: ...

        def b(self) -> B: ...

    class B(Expr): ...

    EGraph().register(A().b())


# https://github.com/egraphs-good/egglog-python/issues/100
def test_functions_seperate_pop():
    egraph = EGraph()

    class T(Expr):
        def __init__(self, x: i64Like) -> None: ...

    with egraph:

        @function
        def f(x: T) -> T: ...

        egraph.register(f(T(1)))

    with egraph:

        @function
        def f(x: T, y: T) -> T: ...  # type: ignore[misc]

        egraph.register(f(T(1), T(2)))  # type: ignore[call-arg]


# https://github.com/egraphs-good/egglog/issues/113
def test_multiple_generics():
    @function
    def f() -> Vec[i64]: ...

    @function
    def g() -> Vec[String]: ...

    egraph = EGraph()

    egraph.register(
        set_(f()).to(Vec[i64]()),
        set_(g()).to(Vec[String]()),
    )

    assert str(egraph.extract(f())) == "Vec[i64].empty()"
    assert str(egraph.extract(g())) == "Vec[String].empty()"


def test_deferred_ruleset():
    @ruleset
    def rules(x: AA):
        yield rewrite(first(x)).to(second(x))

    class AA(Expr):
        def __init__(self) -> None: ...

    @function
    def first(x: AA) -> AA: ...

    @function
    def second(x: AA) -> AA: ...

    check(
        eq(first(AA())).to(second(AA())),
        rules,
        first(AA()),
    )


def test_access_method_on_class():
    class A(Expr):
        def __init__(self) -> None: ...

        def b(self, x: i64Like) -> A: ...

    assert expr_parts(A.b(A(), 1)) == expr_parts(A().b(1))


def test_access_property_on_class():
    class A(Expr):
        def __init__(self) -> None: ...

        @property
        def b(self) -> i64: ...

    assert expr_parts(A.b(A())) == expr_parts(A().b)


class A(Expr):
    def __init__(self) -> None: ...


class TestDefaultReplacements:
    def test_builtin_function_without_body(self):
        @function(builtin=True)
        def f(x: i64Like) -> i64: ...

        assert expr_parts(f(1)) == expr_parts(f(i64(1)))

    def test_eqsort_merge_function_without_body(self):
        @function(merge=lambda old, new: old)
        def f() -> A: ...

        egraph = EGraph()
        egraph.register(set_(f()).to(A()))
        egraph.check(eq(f()).to(A()))

    def test_primitive_constant_with_merge(self):
        best = constant("best", i64, merge=lambda old, new: old.max(new))

        egraph = EGraph()
        egraph.register(set_(best).to(i64(1)), set_(best).to(i64(2)))

        egraph.check(eq(best).to(i64(2)))

    def test_eqsort_constant_with_merge(self):
        merged = constant("merged", A, merge=lambda old, _new: old)

        egraph = EGraph()
        egraph.register(set_(merged).to(A()))

        egraph.check(eq(merged).to(A()))
        assert egraph.function_values(merged) == {merged: A()}
        assert "set_(merged).to(A())" in str(egraph.freeze())

    def test_function(self):
        @function
        def f() -> A:
            return A()

        check_eq(f(), A())

    def test_function_ruleset(self):
        r = ruleset()

        @function(ruleset=r)
        def f() -> A:
            return A()

        check_eq(f(), A(), r)

    def test_function_ruleset_with_subsume(self):
        r = ruleset()

        @function(ruleset=r, subsume=True)
        def f() -> A:
            return A()

        check_eq(f(), A(), r)

    def test_function_ruleset_can_run_after_materialization_without_registration(self):
        r = ruleset()

        @function(ruleset=r)
        def f() -> A:
            return A()

        # Materialize the function once so its default rewrite is added to the ruleset,
        # but do not register any expression that would separately add `f` to the egraph.
        f()
        egraph = EGraph()
        assert not egraph.run(r).updated

    def test_constant(self):
        a = constant("a", A, A())
        check_eq(a, A())

    def test_constant_ruleset(self):
        r = ruleset()
        a = constant("a", A, A(), ruleset=r)

        check_eq(a, A(), r)

    def test_method(self):
        class B(Expr):
            def __init__(self) -> None: ...
            def f(self) -> A:
                return A()

        check_eq(B().f(), A())

    def test_method_ruleset(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            def __init__(self) -> None: ...
            def f(self) -> A:
                return A()

        check_eq(B().f(), A(), r)

    def test_classmethod(self):
        class B(Expr):
            @classmethod
            def f(cls) -> A:
                return A()

        check_eq(B.f(), A())

    def test_property(self):
        class B(Expr):
            def __init__(self, value: i64Like) -> None: ...

            @property
            def a(self) -> A:
                return A()

        check_eq(B(i64(1)).a, A())

    def test_init(self):
        class B(Expr):
            def __init__(self, value: i64Like) -> None:
                return B.wrap(value)

            @classmethod
            def wrap(cls, value: i64Like) -> B: ...

        check_eq(B(i64(1)), B.wrap(i64(1)))

    def test_classmethod_ruleset(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            @classmethod
            def f(cls) -> A:
                return A()

        check_eq(B.f(), A(), r)

    def test_classvar(self):
        class B(Expr):
            a: ClassVar[A] = A()

        check_eq(B.a, A())

    def test_classvar_ruleset(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            a: ClassVar[A] = A()

        check_eq(B.a, A(), r)

    def test_constructor_unextractable(self):
        class B(Expr):
            def __init__(self, value: i64Like) -> None: ...

            @method(unextractable=True)
            def opaque(self) -> B: ...

            def __add__(self, other: B) -> B: ...

        egraph = EGraph()
        opaque = egraph.let("opaque", B(i64(1)).opaque())
        egraph.register(union(opaque).with_(B(i64(1)) + B(i64(1))))
        assert expr_parts(egraph.extract(opaque)) == expr_parts(B(i64(1)) + B(i64(1)))

    def test_method_refer_to_later(self):
        """
        Verify that an earlier method body can refer to values defined in later ones
        """

        class B(Expr):
            def __init__(self) -> None: ...
            def f(self) -> A:
                return self.g()

            def g(self) -> A: ...

        B()
        left = B().f()
        right = B().g()
        check_eq(left, right)

    def test_classmethod_own_class(self):
        class B(Expr):
            def __init__(self) -> None: ...
            @classmethod
            def f(cls) -> B:
                return B()

        check_eq(B.f(), B())


class TestIssue166:
    """
    Raised by @cgyurgyik in https://github.com/egraphs-good/egglog-python/issues/166
    """

    def test_inserting_map(self):
        egraph = EGraph()
        m = egraph.let("map", Map[String, i64].empty().insert(String("a"), i64(42)))
        egraph.run(5)
        egraph.extract(m)

    def test_creating_map(self):
        m = Map[String, i64].empty()
        egraph = EGraph()
        egraph.register(m)
        egraph.extract(m)


def test_helpful_error_function_class():
    class E(Expr):
        @function(cost=10)
        def __init__(self) -> None: ...

    match = "Inside of classes, wrap methods with the `method` decorator, not `function`"
    with pytest.raises(ValueError, match=match):
        E()


class TestCallableValidation:
    def test_primitive_function_ruleset_subsume_rejected(self):
        r = ruleset()

        @function(ruleset=r, subsume=True)
        def f() -> i64:
            return i64(1)

        with pytest.raises(ValueError, match="Primitive-returning callables cannot use subsume"):
            f()

    def test_no_body_function_cannot_use_explicit_ruleset(self):
        r = ruleset()

        @function(ruleset=r)
        def f() -> A: ...

        with pytest.raises(ValueError, match="Explicit rulesets require a body"):
            f()

    def test_constant_without_default_cannot_use_explicit_ruleset(self):
        r = ruleset()

        with pytest.raises(ValueError, match="Explicit rulesets require a default"):
            EGraph().register(constant("no_default", A, ruleset=r))

    def test_primitive_constant_default_cannot_use_explicit_ruleset(self):
        r = ruleset()

        with pytest.raises(ValueError, match="Primitive-returning defaults cannot use an explicit ruleset"):
            EGraph().register(constant("primitive_default", i64, i64(1), ruleset=r))

    def test_eqsort_constant_default_cannot_use_merge(self):
        with pytest.raises(ValueError, match="Eqsort-returning callables with bodies cannot use merge"):
            EGraph().register(constant("default_merge", A, A(), merge=lambda old, _new: old))

    def test_primitive_constant_default_cannot_use_merge(self):
        with pytest.raises(ValueError, match="Primitive-returning callables with bodies cannot use merge"):
            EGraph().register(constant("primitive_default_merge", i64, i64(1), merge=lambda old, new: old.max(new)))

    def test_unit_constant_cannot_use_merge(self):
        with pytest.raises(ValueError, match="Functions that return Unit cannot use merge"):
            EGraph().register(constant("unit_merge", Unit, merge=lambda old, _new: old))

    def test_eqsort_eager_body_cannot_use_merge(self):
        @function(merge=lambda old, new: old)
        def f() -> A:
            return A()

        with pytest.raises(ValueError, match="Eqsort-returning callables with bodies cannot use merge"):
            f()

    def test_primitive_returning_functions_cannot_use_cost(self):
        @function(cost=1)
        def f() -> i64: ...

        with pytest.raises(ValueError, match="Primitive-returning callables cannot use cost"):
            f()

    def test_primitive_returning_functions_cannot_be_unextractable(self):
        @function(unextractable=True)
        def f() -> i64: ...

        with pytest.raises(ValueError, match="Primitive-returning callables cannot be unextractable"):
            f()

    def test_builtin_callables_cannot_use_merge(self):
        @function(builtin=True, merge=lambda old, new: old)
        def f() -> i64: ...

        with pytest.raises(ValueError, match="Builtin callables cannot use merge"):
            f()

    def test_primitive_body_cannot_use_builtin(self):
        @function(builtin=True)
        def f() -> i64:
            return i64(1)

        with pytest.raises(ValueError, match="Builtin callables cannot have a body"):
            f()

    def test_primitive_body_cannot_use_merge(self):
        @function(merge=lambda old, new: old)
        def f() -> i64:
            return i64(1)

        with pytest.raises(ValueError, match="Primitive-returning callables with bodies cannot use merge"):
            f()

    def test_primitive_body_cannot_use_explicit_ruleset(self):
        r = ruleset()

        @function(ruleset=r)
        def f() -> i64:
            return i64(1)

        with pytest.raises(
            ValueError, match="Primitive-returning callables with bodies cannot use an explicit ruleset"
        ):
            f()

    def test_eqsort_body_cannot_use_merge(self):
        r = ruleset()

        @function(ruleset=r, merge=lambda old, new: old)
        def f() -> A:
            return A()

        with pytest.raises(ValueError, match="Eqsort-returning callables with bodies cannot use merge"):
            f()

    def test_eqsort_eager_body_cannot_use_cost(self):
        @function(cost=1)
        def f() -> A:
            return A()

        with pytest.raises(ValueError, match="Eqsort-returning eager bodies cannot use cost"):
            f()

    def test_eqsort_eager_body_cannot_be_unextractable(self):
        @function(unextractable=True)
        def f() -> A:
            return A()

        with pytest.raises(ValueError, match="Eqsort-returning eager bodies cannot be unextractable"):
            f()

    def test_no_body_function_cannot_use_subsume(self):
        @function(subsume=True)
        def f() -> A: ...

        with pytest.raises(ValueError, match="subsume requires an explicit ruleset"):
            f()

    def test_primitive_method_subsume_rejected(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            def __init__(self, value: i64Like) -> None: ...

            @method(subsume=True)
            def f(self) -> i64:
                return i64(1)

        with pytest.raises(ValueError, match="Primitive-returning callables cannot use subsume"):
            B(i64(0)).f()

    def test_primitive_classvar_default_cannot_use_explicit_ruleset(self):
        r = ruleset()

        class B(Expr, ruleset=r):
            a: ClassVar[i64] = i64(1)

            def __init__(self) -> None: ...

        with pytest.raises(ValueError, match="Primitive-returning defaults cannot use an explicit ruleset"):
            _ = B.a


def test_vec_like_conversion():
    """
    Test that we can use a generic type alias for conversion
    """

    @function
    def my_fn(xs: VecLike[i64, i64Like]) -> Unit: ...

    assert expr_parts(my_fn((1, 2))) == expr_parts(my_fn(Vec(i64(1), i64(2))))
    assert expr_parts(my_fn([])) == expr_parts(my_fn(Vec[i64].empty()))


def test_set_like_conversion():
    @function
    def my_fn(xs: SetLike[i64, i64Like]) -> Unit: ...

    assert expr_parts(my_fn({1, 2})) == expr_parts(my_fn(Set(i64(1), i64(2))))
    assert expr_parts(my_fn(set())) == expr_parts(my_fn(Set[i64].empty()))


def test_map_like_conversion():
    @function
    def my_fn(xs: MapLike[i64, String, i64Like, StringLike]) -> Unit: ...

    assert expr_parts(my_fn({1: "hi"})) == expr_parts(my_fn(Map[i64, String].empty().insert(i64(1), String("hi"))))
    assert expr_parts(my_fn({})) == expr_parts(my_fn(Map[i64, String].empty()))


def test_maybe_builtin_surface():
    none_expr = EGraph().extract(Maybe[f64].none())
    assert none_expr.value is None

    some_expr = EGraph().extract(Maybe[f64].some(1.0))
    assert some_expr.value is not None
    assert some_expr.value.value == 1.0

    assert EGraph().extract(Maybe[f64].some(1.0).unwrap()).value == 1.0
    assert EGraph().extract(Maybe[f64].none().unwrap_or(2.5)).value == 2.5


def test_higher_order_maybe_pair_and_catch_builtins():
    assert EGraph().extract(Maybe[i64].some(2).match(lambda x: x + 3, i64(0))).value == 5
    assert EGraph().extract(Maybe[i64].none().match(lambda x: x + 3, i64(7))).value == 7

    matched = EGraph().extract(Pair(i64(2), i64(3)).match(lambda left, right: left + right))
    assert matched.value == 5

    mapped_left = EGraph().extract(Pair(i64(2), i64(3)).map_left(lambda left: left + 10))
    left, right = mapped_left.value
    assert left.value == 12
    assert right.value == 3

    mapped_right = EGraph().extract(Pair(i64(2), i64(3)).map_right(lambda right: right + 10))
    left, right = mapped_right.value
    assert left.value == 2
    assert right.value == 13

    caught = EGraph().extract(catch(lambda: Maybe[i64].some(4).unwrap()))
    assert caught.value is not None
    assert caught.value.value == 4

    failed = EGraph().extract(catch(lambda: Maybe[i64].none().unwrap()))
    assert failed.value is None


@pytest.mark.xfail(
    reason=(
        "egg-smol type inference for unstable-fn on synthesized primitive lambdas is too weak "
        "under overloaded unstable-catch, so nested catch/match fails when the inner lambda "
        "returns a different sort from the outer one"
    ),
    strict=True,
)
def test_nested_catch_match_with_different_inner_lambda_result_sort() -> None:
    expr = catch(lambda: i64(1)).match(
        lambda _: catch(lambda: f64(2.0)).match(lambda v: v, f64(0.0)),
        f64(9.0),
    )

    assert EGraph().extract(expr).value == 2.0


def test_maybe_conversion_and_merge_builtin():
    @function
    def maybe_identity(x: Maybe[i64]) -> Maybe[i64]: ...

    assert expr_parts(maybe_identity(None)) == expr_parts(maybe_identity(Maybe[i64].none()))

    merged = EGraph().extract(maybe_f64_merge_with_tol(Maybe[f64].some(1.0), Maybe[f64].some(1.0000005), 1e-6))
    assert merged.value is not None
    assert merged.value.value == 1.0

    merged_none = EGraph().extract(maybe_f64_merge_with_tol(Maybe[f64].none(), Maybe[f64].some(1.0), 1e-6))
    assert merged_none.value is None


class TestEqNE:
    def test_eq(self):
        assert i64(3) == i64(3)

    def test_ne(self):
        EGraph().check(i64(3) != i64(4))

    def test_eq_false(self):
        assert not (i64(3) == 4)  # noqa: SIM201


def test_no_upcast_eq():
    """
    Verifies that if two items can be upcast to something, calling == on them won't use
    equality
    """

    class A(Expr):
        def __init__(self) -> None: ...

    class B(Expr):
        def __init__(self) -> None: ...
        def __eq__(self, other: B) -> B: ...  # type: ignore[override]

    converter(A, B, lambda a: B())

    assert isinstance(A() == A(), Fact)
    assert not isinstance(B() == B(), Fact)


def test_isinstance_expr():
    """
    Verifies that isinstance() works on Exprs, and returns a Fact
    """

    class A(Expr):
        def __init__(self) -> None: ...

    class B(Expr):
        def __init__(self) -> None: ...

    assert isinstance(A(), A)
    assert not isinstance(A(), B)


class TestMatch:
    def test_class(self):
        """
        Verify that we can pattern match on expressions
        """

        class A(Expr):
            def __init__(self) -> None: ...

        class B(Expr):
            def __init__(self) -> None: ...

        a = A()
        match a:
            case B():
                msg = "Should not have matched B"
                raise ValueError(msg)
            case A():
                pass
            case _:
                msg = "Should have matched A"
                raise ValueError(msg)

    def test_literal(self):
        match i64(10):
            case i64(i):
                assert i == 10
            case _:
                msg = "Should have matched i64(10)"
                raise ValueError(msg)

    def test_literal_fail(self):
        """
        Verify that matching on a literal that does not match raises an error
        """
        match i64(10) + i64(10):
            case i64(_i):
                msg = "Should not have matched i64(20)"
                raise ValueError(msg)

    def test_custom_args(self):
        class A(Expr):
            def __init__(self) -> None: ...

            __match_args__ = ("a", "b")

            @method(preserve=True)  # type: ignore[prop-decorator]
            @property
            def a(self) -> int:
                return 1

            @method(preserve=True)  # type: ignore[prop-decorator]
            @property
            def b(self) -> str:
                return "hi"

        match A():
            case A(a, b):
                assert a == 1
                assert b == "hi"
            case _:
                msg = "Should have matched A"
                raise ValueError(msg)

    def test_custom_args_fail(self):
        """
        Verify that matching on a custom match that does not match raises an error
        """

        class A(Expr):
            def __init__(self) -> None: ...

            __match_args__ = ("a",)

            @method(preserve=True)  # type: ignore[prop-decorator]
            @property
            def a(self) -> int:
                raise AttributeError

        match A():
            case A(_a):
                msg = "Should not have matched A"
                raise ValueError(msg)


T = TypeVar("T")


def test_type_param_sub():
    """
    Verify that type substituion works properly, by comparing string version.

    Comparing actual versions is always false if they are no the same object for unions
    """
    V = Vec[T] | int
    assert str(V[Unit]) == str(Vec[Unit] | int)  # type: ignore[misc]


def test_override_hash():
    class A(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __hash__(self) -> int:
            return 42

    assert hash(A()) == 42


def test_serialize_warning_max_functions():
    class A(Expr):
        def __init__(self) -> None: ...

    egraph = EGraph()
    egraph.register(A())
    with pytest.warns(UserWarning, match="A"):
        egraph._serialize(max_functions=0)


def test_serialize_warning_max_calls():
    class A(Expr): ...

    @function
    def f(x: StringLike) -> A: ...

    egraph = EGraph()
    egraph.register(f("a"), f("b"))
    with pytest.warns(UserWarning, match="f"):
        egraph._serialize(max_calls_per_function=1)


EXAMPLE_FILES = list((pathlib.Path(__file__).parent / "../egglog/examples").glob("*.py"))


# Test all files in the `examples` directory by importing them in this parametrized test
@pytest.mark.parametrize("name", [f.stem for f in EXAMPLE_FILES if f.stem != "__init__"])
def test_example(name):
    importlib.import_module(f"egglog.examples.{name}")


@function
def f() -> i64: ...


class E(Expr):
    X: ClassVar[i64]

    def __init__(self) -> None: ...
    def m(self) -> i64: ...

    @property
    def p(self) -> i64: ...

    @classmethod
    def cm(cls) -> i64: ...


egraph = EGraph()

C = constant("C", i64)

zero = i64(0)
egraph.register(
    set_(f()).to(zero),
    set_(E().m()).to(zero),
    set_(E.X).to(zero),
    set_(E().p).to(zero),
    set_(C).to(zero),
    set_(E.cm()).to(zero),
)


@pytest.mark.parametrize(
    "c",
    [
        pytest.param(E, id="init"),
        pytest.param(f, id="function"),
        pytest.param(E.m, id="method"),
        pytest.param(E.X, id="class var"),
        pytest.param(E.p, id="property"),
        pytest.param(C, id="constant"),
        pytest.param(E.cm, id="class method"),
    ],
)
def test_function_size(c):
    assert egraph.function_size(c) == 1


def test_all_function_size():
    res = egraph.all_function_sizes()
    assert set(res) == {
        (E, 1),
        (f, 1),
        (E.m, 1),
        (E.X, 1),
        (E.p, 1),
        (C, 1),
        (E.cm, 1),
    }


def test_overall_run_report():
    assert EGraph().stats()


def test_function_values():
    egraph = EGraph()

    @function
    def f(x: i64Like) -> i64: ...

    egraph.register(set_(f(i64(1))).to(i64(2)))
    values = egraph.function_values(f)
    assert values == {f(i64(1)): i64(2)}


def test_dynamic_cost():
    """
    https://github.com/egraphs-good/egglog-experimental/blob/6d07a34ac76deec751f86f70d9b9358cd3e236ca/tests/integration_test.rs#L5-L35
    """

    class E(Expr):
        def __init__(self, x: i64Like) -> None: ...
        def __add__(self, other: E) -> E: ...
        @method(cost=200)
        def __sub__(self, other: E) -> E: ...

    egraph = EGraph()
    egraph.register(
        union(E(2)).with_(E(1) + E(1)),
        set_cost(E(2), 1000),
        set_cost(E(1), 100),
    )
    assert egraph.extract(E(2), include_cost=True) == (E(1) + E(1), 203)
    with egraph:
        egraph.register(set_cost(E(1) + E(1), 800))
        assert egraph.extract(E(2), include_cost=True) == (E(2), 1001)
    with egraph:
        egraph.register(set_cost(E(1) + E(1), 798))
        assert egraph.extract(E(2), include_cost=True) == (E(1) + E(1), 1000)
    egraph.register(union(E(2)).with_(E(5) - E(3)))
    assert egraph.extract(E(2), include_cost=True) == (E(1) + E(1), 203)
    egraph.register(set_cost(E(5) - E(3), 198))
    assert egraph.extract(E(2), include_cost=True) == (E(5) - E(3), 202)


class TestScheduler:
    def test_seq_schedule_decls_track_ruleset_updates(self):
        egraph = EGraph()

        rel = relation("rel_live", i64)
        live_rules = ruleset(name="live-rules")
        schedule = seq(live_rules, run()).saturate()
        _ = schedule.__egg_decls__

        live_rules.register(rule(rel(i64(0))).then(rel(i64(1))))

        egraph.register(rel(i64(0)))
        egraph.run(schedule)
        egraph.check(rel(i64(1)))

    def test_sequence_repeat_saturate(self):
        """
        Mirrors the scheduling example: alternate step-right and step-left,
        saturating each, repeated 10 times. Verifies final facts.
        """
        egraph = EGraph()

        left = relation("left", i64)
        right = relation("right", i64)

        x, y = vars_("x y", i64)

        # Name rulesets to make schedule translation stable and explicit
        step_left = ruleset(
            rule(
                left(x),
                right(x),
            ).then(left(x + 1)),
            name="step-left",
        )
        step_right = ruleset(
            rule(
                left(x),
                right(y),
                eq(x).to(y + 1),
            ).then(right(x)),
            name="step-right",
        )

        # Initial facts
        egraph.register(left(i64(0)), right(i64(0)))

        # (repeat 10 (seq (run step-right) (saturate step-left)))
        egraph.run(seq(step_right, step_left) * 10)

        # We took 10 left steps, but only 9 right steps (first can't move)
        egraph.check(left(i64(10)), right(i64(9)))
        egraph.check_fail(left(i64(11)), right(i64(10)))

    def test_backoff_scheduler(self):
        """
        Passing `scheduler=...` to run(...) hoists the scheduler to the
        outer scope. This is equivalent to an explicit outer `bo.scope(...)`
        around the whole repeated schedule.

        https://egraphs.zulipchat.com/#narrow/channel/375765-egg.2Fegglog/topic/.E2.9C.94.20Backoff.20Scheduler.20Example/with/538745863
        """
        includes = relation("includes", i64)
        x = var("x", i64)
        grow = ruleset(rule(includes(x)).then(includes(x + 1)))
        shrink = ruleset(rule(includes(x)).then(includes(x - 1)))

        bo = back_off(match_limit=1)

        def _run_and_collect(schedule: Schedule) -> set[int]:
            egraph = EGraph()
            egraph.register(includes(i64(0)))
            with egraph:
                egraph.run(schedule)
                values = set()
                for i in range(-3, 4):
                    try:
                        egraph.check(includes(i64(i)))
                        values.add(i)
                    except EggSmolError:
                        pass
            return values

        default_values = _run_and_collect((grow + shrink) * 3)
        assert default_values == {-3, -2, -1, 0, 1, 2, 3}

        implicit_values = _run_and_collect((run(grow, scheduler=bo) + shrink) * 3)
        explicit_values = _run_and_collect(bo.scope((run(grow, scheduler=bo) + shrink) * 3))

        assert implicit_values == explicit_values == {-3, -2, -1, 0, 1, 2}

    def test_backoff_scheduler_fresh_rematch_flag_changes_match_frontier(self):
        r = relation("R", i64)
        s = relation("S", i64)
        seed = relation("Seed")
        x = var("x", i64)

        copy = ruleset(rule(r(x)).then(s(x)), name="copy")
        grow = ruleset(rule(seed()).then(r(i64(3))), name="grow")

        def _run_and_collect(*, fresh_rematch: bool) -> set[int]:
            egraph = EGraph()
            egraph.register(r(i64(0)), r(i64(1)), r(i64(2)), seed())
            bo = back_off(match_limit=2, ban_length=2, fresh_rematch=fresh_rematch)
            schedule = bo.scope(seq(run(copy, scheduler=bo), run(grow), run(copy, scheduler=bo)))
            egraph.run(schedule)
            values = set()
            for i in range(4):
                try:
                    egraph.check(s(i64(i)))
                    values.add(i)
                except EggSmolError:
                    pass
            return values

        assert _run_and_collect(fresh_rematch=False) == {0, 1, 2}
        assert _run_and_collect(fresh_rematch=True) == {0, 1, 2, 3}

    def test_persistent_scheduler_reuses_state_across_runs(self):
        r = relation("R", i64)
        s = relation("S", i64)
        seed = relation("Seed")
        x = var("x", i64)

        copy = ruleset(rule(r(x)).then(s(x)), name="copy")
        grow = ruleset(rule(seed()).then(r(i64(3))), name="grow")

        def _run_and_collect(*, fresh_rematch: bool) -> set[int]:
            egraph = EGraph()
            egraph.register(r(i64(0)), r(i64(1)), r(i64(2)), seed())
            scheduler = back_off(match_limit=2, ban_length=2, fresh_rematch=fresh_rematch).persistent()
            egraph.run(run(copy, scheduler=scheduler))
            egraph.run(run(grow))
            egraph.run(run(copy, scheduler=scheduler))
            values = set()
            for i in range(4):
                try:
                    egraph.check(s(i64(i)))
                    values.add(i)
                except EggSmolError:
                    pass
            return values

        assert _run_and_collect(fresh_rematch=False) == {0, 1, 2}
        assert _run_and_collect(fresh_rematch=True) == {0, 1, 2, 3}

    def test_persistent_scheduler_is_saved_once_across_runs(self):
        r = relation("R_saved_scheduler", i64)
        s = relation("S_saved_scheduler", i64)
        x = var("x", i64)
        copy = ruleset(rule(r(x)).then(s(x)), name="copy-saved-scheduler")

        egraph = EGraph(save_egglog_string=True)
        egraph.register(r(i64(0)), r(i64(1)))
        scheduler = back_off(match_limit=2, ban_length=2, fresh_rematch=True).persistent()

        egraph.run(run(copy, scheduler=scheduler))
        egraph.run(run(copy, scheduler=scheduler))

        scheduler_lines = [line for line in egraph.as_egglog_string.splitlines() if line.startswith("(let-scheduler ")]
        run_with_lines = [line for line in egraph.as_egglog_string.splitlines() if "(run-with " in line]

        assert len(scheduler_lines) == 1
        assert len(run_with_lines) == 2

    def test_custom_scheduler_invalid_until(self):
        """
        Custom schedulers do not support equality facts in :until,
        and only allow a single non-equality fact.
        """
        egraph = EGraph()

        rel = relation("rel", i64)
        x = var("x", i64)
        r = ruleset(name="r")
        bo = back_off(match_limit=1)

        # Equality in until should error via high-level run
        with pytest.raises(ValueError, match="Cannot use equality fact with custom scheduler"):
            egraph.run(run(r, eq(x).to(i64(1)), scheduler=bo))

        # Multiple until facts should error via high-level run
        with pytest.raises(ValueError, match="Can only have one until fact with custom scheduler"):
            egraph.run(run(r, rel(i64(0)), rel(i64(1)), scheduler=bo))


@function
def ff(x: i64Like, y: i64Like) -> E: ...


@function
def gg() -> E: ...


class TestCustomExtract:
    @staticmethod
    def _capture_container_children_costs(
        expr: BaseExpr,
        *,
        leaf_cost: Callable[[BaseExpr], int],
        should_capture: Callable[[BaseExpr, list[int]], bool],
    ) -> tuple[BaseExpr, BaseExpr, list[int]]:
        seen: dict[str, object] = {}

        def my_cost_model(egraph: EGraph, candidate: BaseExpr, children_costs: list[int]) -> int:
            match candidate:
                case i64() | String():
                    return leaf_cost(candidate)
                case _:
                    if should_capture(candidate, children_costs):
                        seen["expr"] = candidate
                        seen["children_costs"] = children_costs.copy()
            return default_cost_model(egraph, candidate, children_costs)

        extracted, _cost = EGraph().extract(expr, include_cost=True, cost_model=my_cost_model)

        assert "expr" in seen
        assert "children_costs" in seen

        return extracted, cast("BaseExpr", seen["expr"]), cast("list[int]", seen["children_costs"])

    @staticmethod
    def _small_leaf_cost(expr: BaseExpr) -> int:
        match expr:
            case i64():
                return {1: 11, 2: 22, 3: 33, 4: 44}[expr.value]
            case String():
                return {"a": 101, "b": 202}[expr.value]
            case _:
                msg = f"Unexpected leaf {expr!r}"
                raise AssertionError(msg)

    @pytest.mark.parametrize(
        "expr",
        [
            pytest.param(i64(10), id="i64"),
            pytest.param(f64(10.0), id="f64"),
            pytest.param(String("hi"), id="String"),
            pytest.param(Bool(True), id="Bool"),
            pytest.param(Rational(1, 2), id="Rational"),
            pytest.param(BigInt(10), id="BigInt"),
            pytest.param(BigRat(1, 2), id="BigRat"),
            pytest.param(PyObject("hi"), id="PyObject"),
            pytest.param(Vec(i64(1), i64(2)), id="Vec"),
            pytest.param(Set(i64(1), i64(2)), id="Set"),
            pytest.param(Map[i64, String].empty().insert(i64(1), String("hi")), id="Map"),
            pytest.param(MultiSet(i64(1), i64(1)), id="MultiSet"),
            pytest.param(Unit(), id="Unit"),
            pytest.param(UnstableFn[E, i64, i64](ff), id="fn"),
            pytest.param(UnstableFn[E, i64](ff, i64(1)), id="fn partial"),
        ],
    )
    def test_to_from_value(self, expr):
        egraph = EGraph()
        expr = egraph.extract(expr)
        assert expr == self._to_from_value(egraph, expr)

    def _to_from_value(self, egraph: EGraph, expr: RuntimeExpr):
        typed_expr = expr.__egg_typed_expr__
        value = egraph._state.typed_expr_to_value(typed_expr)
        res_val = egraph._state.value_to_expr(typed_expr.tp, value)
        return expr.__with_expr__(TypedExprDecl(typed_expr.tp, res_val))

    def test_compare_values(self):
        egraph = EGraph()
        egraph.register(E(), gg())
        e_value = self._to_from_value(egraph, cast("RuntimeExpr", E()))
        gg_value = self._to_from_value(egraph, cast("RuntimeExpr", gg()))
        assert e_value != gg_value
        assert hash(e_value) != hash(gg_value)
        assert str(e_value) != str(gg_value)

    def test_no_changes(self):
        egraph = EGraph()
        assert egraph.extract(E(), include_cost=True) == egraph.extract(
            E(), include_cost=True, cost_model=default_cost_model
        )

    def test_calls_methods(self):
        @function
        def my_f(xs: Vec[i64]) -> E: ...

        # cost = 2
        x = i64(10)
        # cost = 3 + 2 = 5
        xs = Vec(x)
        # cost = 100
        res = E()
        # cost = 1 + 5  = 6
        called = my_f(xs)
        egraph = EGraph()
        egraph.register(union(called).with_(res))

        def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if get_callable_fn(expr) == E:
                return 100
            match expr:
                case i64():
                    return 2
                case Vec():
                    return 3 + sum(children_costs)
            return default_cost_model(egraph, expr, children_costs)

        my_cost_model = MagicMock(side_effect=my_cost_model)
        assert egraph.extract(called, include_cost=True, cost_model=my_cost_model) == (called, 6)

        my_cost_model.assert_any_call(egraph, res, [])
        my_cost_model.assert_any_call(egraph, xs, [2])
        my_cost_model.assert_any_call(egraph, x, [])
        my_cost_model.assert_any_call(egraph, called, [5])

    def test_map_container_children_costs_match_python_items_order(self):
        expr = Map[i64, String].empty().insert(i64(2), String("b")).insert(i64(1), String("a"))
        extracted, seen_expr, seen_children_costs = self._capture_container_children_costs(
            expr,
            leaf_cost=self._small_leaf_cost,
            should_capture=lambda candidate, children_costs: isinstance(candidate, Map) and len(children_costs) == 4,
        )

        map_expr = cast("Map[i64, String]", seen_expr)
        flattened_item_costs = [
            self._small_leaf_cost(item) for key, value in map_expr.value.items() for item in (key, value)
        ]

        assert flattened_item_costs == seen_children_costs
        assert flattened_item_costs == [11, 101, 22, 202]
        assert list(cast("Map[i64, String]", extracted).value.items()) == list(map_expr.value.items())

    def test_vec_container_children_costs_match_python_iteration_order(self):
        expr = Vec(i64(2), i64(1), i64(3))
        extracted, _seen_expr, seen_children_costs = self._capture_container_children_costs(
            expr,
            leaf_cost=self._small_leaf_cost,
            should_capture=lambda candidate, children_costs: isinstance(candidate, Vec) and len(children_costs) == 3,
        )

        extracted_vec = cast("Vec[i64]", extracted)
        iter_costs = [self._small_leaf_cost(item) for item in extracted_vec]

        assert iter_costs == seen_children_costs
        assert list(extracted_vec) == [i64(2), i64(1), i64(3)]

    def test_set_container_children_costs_match_python_iteration_order(self):
        expr = Set(i64(2), i64(1), i64(3))
        extracted, _seen_expr, seen_children_costs = self._capture_container_children_costs(
            expr,
            leaf_cost=self._small_leaf_cost,
            should_capture=lambda candidate, children_costs: isinstance(candidate, Set) and len(children_costs) == 3,
        )

        extracted_set = cast("Set[i64]", extracted)
        iter_costs = [self._small_leaf_cost(item) for item in extracted_set]

        assert iter_costs == seen_children_costs
        assert list(extracted_set) == [i64(1), i64(2), i64(3)]

    def test_multiset_container_children_costs_match_python_iteration_order(self):
        expr = MultiSet(i64(2), i64(1), i64(2), i64(3))
        extracted, _seen_expr, seen_children_costs = self._capture_container_children_costs(
            expr,
            leaf_cost=self._small_leaf_cost,
            should_capture=lambda candidate, children_costs: isinstance(candidate, MultiSet)
            and len(children_costs) == 4,
        )

        extracted_multiset = cast("MultiSet[i64]", extracted)
        iter_costs = [self._small_leaf_cost(item) for item in extracted_multiset]

        assert iter_costs == seen_children_costs
        assert list(extracted_multiset) == [i64(1), i64(2), i64(2), i64(3)]

    def test_pair_container_children_costs_match_python_value_order(self):
        expr = Pair(i64(2), i64(1))
        extracted, _seen_expr, seen_children_costs = self._capture_container_children_costs(
            expr,
            leaf_cost=self._small_leaf_cost,
            should_capture=lambda candidate, children_costs: isinstance(candidate, Pair) and len(children_costs) == 2,
        )

        pair_value = cast("Pair[i64, i64]", extracted).value
        value_costs = [self._small_leaf_cost(item) for item in pair_value]

        assert value_costs == seen_children_costs
        assert pair_value == (i64(2), i64(1))

    def test_maybe_container_children_costs_match_python_value_order(self):
        some_expr = Maybe[i64].some(i64(3))
        extracted_some, _seen_some, seen_some_children_costs = self._capture_container_children_costs(
            some_expr,
            leaf_cost=self._small_leaf_cost,
            should_capture=lambda candidate, children_costs: isinstance(candidate, Maybe) and len(children_costs) == 1,
        )

        some_value = cast("Maybe[i64]", extracted_some).value
        assert some_value is not None
        assert [self._small_leaf_cost(some_value)] == seen_some_children_costs
        assert some_value == i64(3)

        none_expr = Maybe[i64].none()
        extracted_none, _seen_none, seen_none_children_costs = self._capture_container_children_costs(
            none_expr,
            leaf_cost=self._small_leaf_cost,
            should_capture=lambda candidate, children_costs: isinstance(candidate, Maybe) and len(children_costs) == 0,
        )

        assert cast("Maybe[i64]", extracted_none).value is None
        assert seen_none_children_costs == []
        assert cast("Maybe[i64]", extracted_none).value is None

    @pytest.mark.xfail(reason="Errors dont bubble, just panic")
    def test_errors_bubble(self):
        def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            msg = "bad"
            raise ValueError(msg)

        egraph = EGraph()

        with pytest.raises(ValueError, match="bad"):
            egraph.extract(i64(10), cost_model=my_cost_model)

    def test_dag_cost_model(self):
        egraph = EGraph()
        expr = ff(1, 2)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=greedy_dag_cost_model())
        assert cost.total == 3
        assert expr == res

        expr = ff(1, 1)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=greedy_dag_cost_model())
        assert cost.total == 2
        assert expr == res

        @function
        def bin(l: E, r: E) -> E: ...

        x = constant("x", E)
        y = constant("y", E)
        expr = bin(x, bin(x, y))
        egraph.register(expr)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=greedy_dag_cost_model())
        assert cost.total == 4
        assert expr == res


def test_class_module():
    class A(Expr):
        def __init__(self) -> None: ...

    assert A.__module__ == __name__


def test_function_module():
    @function
    def f() -> i64: ...

    assert f.__module__ == __name__


def test_method_module():
    class A(Expr):
        def m(self) -> i64: ...

    assert A.m.__module__ == __name__


def test_class_doc():
    class A(Expr):
        """Docstring for A"""

    assert A.__doc__ == "Docstring for A"


def test_constructor_doc():
    class A(Expr):
        @classmethod
        def create(cls) -> A:
            """Docstring for A.create"""

    assert A.create.__doc__ == "Docstring for A.create"


def test_function_doc():
    @function
    def f() -> i64:
        """Docstring for f"""

    assert f.__doc__ == "Docstring for f"


def test_get_class_method():
    class A(Expr):
        def __init__(self) -> None: ...

        def __eq__(self, other: A) -> A: ...  # type: ignore[override]

    assert eq(A() == A()).to(A.__eq__(A(), A()))


def test_match_none_pyobject():
    x = PyObject(None)
    match x:
        case PyObject(None):
            pass
        case _:
            raise AssertionError


class MyInt(int):
    pass


def test_binary_convert_parent():
    """
    Verify that a binary method will convert an arg based on the parent type
    """

    class Math(Expr):
        def __init__(self, value: i64Like) -> None: ...

        def __add__(self, other: Math | MyInt) -> Math: ...

    converter(int, Math, lambda i: Math(int(i)))
    assert Math(5) + MyInt(10) == Math(5) + Math(10)


def test_py_eval_fn_no_globals():
    """
    Verify that PyObject without globals still works
    """
    assert not hasattr(int, "__globals__")
    assert EGraph().extract(PyObject(int)(PyObject.from_int(10))).value == 10


class MathPrim(Expr):
    def __init__(self) -> None: ...

    def __bytes__(self) -> bytes:
        return b"hi"

    def __str__(self) -> str:
        return "hi"

    def __repr__(self) -> str:
        return "hi"

    # def __format__(self, format_spec: str) -> str:
    #     return "hi"

    def __hash__(self) -> int:
        return 42

    def __bool__(self) -> bool:
        return False

    def __int__(self) -> int:
        return 1000

    def __float__(self) -> float:
        return 100.0

    def __complex__(self) -> complex:
        return 1 + 0j

    def __index__(self) -> int:
        return 20

    def __len__(self) -> int:
        return 10

    def __length_hint__(self) -> int:
        return 5

    def __iter__(self) -> Iterator[int]:
        yield 1

    def __reversed__(self) -> Iterator[int]:
        yield 10

    def __contains__(self, item: int) -> bool:
        return True


m = MathPrim()


@pytest.mark.parametrize(
    ("expr", "res"),
    [
        pytest.param(lambda: bytes(m), b"hi", id="bytes"),
        pytest.param(lambda: str(m), "hi", id="str"),
        pytest.param(lambda: repr(m), "hi", id="repr"),
        pytest.param(lambda: format(m, ""), "hi", id="format"),
        pytest.param(lambda: hash(m), 42, id="hash"),
        pytest.param(lambda: bool(m), False, id="bool"),
        pytest.param(lambda: int(m), 1000, id="int"),
        pytest.param(lambda: float(m), 100.0, id="float"),
        pytest.param(lambda: complex(m), 1 + 0j, id="complex"),
        pytest.param(lambda: m.__index__(), 20, id="index"),
        pytest.param(lambda: len(m), 10, id="len"),
        pytest.param(lambda: m.__length_hint__(), 5, id="length_hint"),
        pytest.param(lambda: list(m), [1], id="iter"),
        pytest.param(lambda: list(reversed(m)), [10], id="reversed"),
        pytest.param(lambda: 1 in m, True, id="contains"),
    ],
)
def test_always_preserved(expr, res):
    assert expr() == res


def test_class_lookup_method():
    class A(Expr):
        def __init__(self) -> None: ...

        def m(self) -> i64: ...
        def __eq__(self, other: A) -> A: ...  # type: ignore[override]
        def __add__(self, other: A) -> A: ...

        def __str__(self) -> str:
            """Hi"""
            return "hi"

    assert A.m(A()) == A().m()
    assert isinstance(A.m, RuntimeFunction)
    assert eq(A.__eq__(A(), A())).to(A() == A())
    assert isinstance(A.__eq__, RuntimeFunction)
    assert eq(A.__add__(A(), A())).to(A() + A())
    assert isinstance(A.__add__, RuntimeFunction)
    assert A.__str__(A()) == "hi"
    assert A.__str__.__doc__ == "Hi"


def test_py_object_raise_exception():
    """
    Verify that PyObject can raise exceptions properly
    """
    msg = "bad"

    def raises(_val):
        raise ValueError(msg)

    egraph = EGraph()
    with pytest.raises(ValueError, match=msg):
        egraph.extract(PyObject(raises)(PyObject(None)))


def test_mutates_self_rewrite():
    mutates_ruleset = ruleset()

    class MutateMethod(Expr, ruleset=mutates_ruleset):
        def __init__(self) -> None: ...
        def incr(self) -> MutateMethod: ...

        @method(mutates_self=True)
        def mutates(self) -> None:
            self.__replace_expr__(self.incr())

    x = MutateMethod()
    x.mutates()
    egraph = EGraph()
    egraph.register(x)
    egraph.run(mutates_ruleset)
    egraph.check(x == MutateMethod().incr())


def test_mutates_self_preserved():
    class MutatePreserved(Expr):
        def __init__(self) -> None: ...
        def incr(self) -> MutatePreserved: ...

        @method(preserve=True)
        def mutates(self) -> None:
            self.__replace_expr__(self.incr())

    x = MutatePreserved()
    x.mutates()
    assert x == MutatePreserved().incr()


def test_binary_conversion_lookup_parent_class():
    class X(Expr):
        def __init__(self, value: object) -> None: ...

        def __add__(self, other: XLike) -> X: ...
        def __radd__(self, other: XLike) -> X: ...

    XLike: TypeAlias = X | object
    converter(object, X, lambda x: X(PyObject(x)))

    assert X(1) + 2 == X(1) + X(2)
    assert 2 + X(1) == X(2) + X(1)


def test_binary_preserved():
    class X(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(preserve=True)
        def __add__(self, other: T) -> tuple[X, T]:
            return (self, other)

        def __radd__(self, other: object) -> tuple[X, X]: ...

    converter(i64, X, X)

    assert X(1) + 10 == (X(1), 10)
    assert 10 + X(1) == (X(10), X(1))


def test_custom_cost_model_size():
    """
    https://egraphs.zulipchat.com/#narrow/channel/375765-egg.2Fegglog/topic/Cost.20function.3A.20using.20function.20values.20of.20subtrees/near/577062352
    """

    class KAT(Expr):
        @classmethod
        def eps(cls) -> KAT: ...

        @classmethod
        def emp(cls) -> KAT: ...

        def func(self, other: KAT) -> KAT: ...

        def size(self) -> i64: ...

    eps, emp = KAT.eps(), KAT.emp()

    eg = EGraph()
    q0 = eg.let("q0", KAT.func(eps, emp))

    eg.register(set_(eps.size()).to(i64(1)))
    eg.register(set_(emp.size()).to(i64(0)))

    def conv_cost(eg, expr, child_costs):
        if isinstance(expr, KAT):
            args = get_callable_args(expr)
            return sum(int(eg.lookup_function_value(cast("KAT", a).size())) for a in args)

        return 2

    assert eg.extract(q0, include_cost=True, cost_model=conv_cost) == (KAT.eps().func(KAT.emp()), 1)
