# mypy: disable-error-code="empty-body"
from __future__ import annotations

import importlib
import math
import pathlib
from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import ClassVar, TypeAlias, TypeVar, cast
from unittest.mock import MagicMock

import pytest

import egglog.bindings as egg_bindings
import egglog.builtins as egg_builtins
from egglog import *
from egglog.declarations import (
    BiRewriteDecl,
    CallDecl,
    FunctionRef,
    Ident,
    JustTypeRef,
    MethodRef,
    RewriteDecl,
    TypedExprDecl,
)
from egglog.runtime import RuntimeExpr, RuntimeFunction

_BuiltinExprT = TypeVar("_BuiltinExprT", bound=BaseExpr)


class TestExprStr:
    def test_unwrap_lit(self):
        assert str(i64(1) + 1) == "i64(1) + 1"
        assert str(i64(1).max(2)) == "i64(1).max(2)"

    def test_ne(self):
        assert str(ne(i64(1)).to(i64(2))) == "ne(i64(1)).to(i64(2))"


@pytest.mark.parametrize(
    "eval_mode",
    ["seminaive", "naive", "unsafe-seminaive"],
)
def test_rule_eval_mode(eval_mode: RuleEvalMode) -> None:
    rel = relation(f"eval_mode_{eval_mode}", i64)
    x = var("x", i64)
    egraph = EGraph(rel(i64(0)))

    egraph.register(rule(rel(x), eval_mode=eval_mode).then(rel(x + 1)))
    egraph.run(1)
    egraph.check(rel(i64(1)))


@pytest.mark.parametrize("seminaive", [False, True])
def test_rule_eval_mode_rejects_unknown_value(seminaive: bool) -> None:
    rel = relation(f"invalid_eval_mode_{seminaive}", i64)
    x = var("x", i64)
    invalid = cast("RuleEvalMode", "unknown")

    with pytest.raises(ValueError, match="Unknown rule evaluation mode"):
        EGraph(seminaive=seminaive).register(rule(rel(x), eval_mode=invalid).then(rel(x)))


def test_per_egraph_configuration() -> None:
    egraph = EGraph(num_threads=2, no_decomp=True)

    assert egraph.num_threads() == 2
    assert egraph.no_decomp()
    egraph.set_num_threads(1)
    egraph.set_no_decomp(False)
    assert egraph.num_threads() == 1
    assert not egraph.no_decomp()


@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_zero_threads_uses_available_parallelism(*, use_setter: bool) -> None:
    egraph = EGraph(num_threads=1 if use_setter else 0)
    if use_setter:
        egraph.set_num_threads(0)

    assert egraph.num_threads() >= 1


@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_egraph_no_decomp_reaches_saved_rules(*, use_setter: bool) -> None:
    rel = relation(f"no_decomp_rel_{use_setter}", i64)
    x = var("x", i64)
    egraph = EGraph(save_egglog_string=True, no_decomp=not use_setter)
    if use_setter:
        egraph.set_no_decomp(True)

    egraph.register(rule(rel(x)).then(rel(x + 1)))

    assert ":no-decomp" in egraph.as_egglog_string


def test_global_naive_mode_reaches_saved_rules() -> None:
    rel = relation("global_naive_saved_rule", i64)
    x = var("x", i64)
    egraph = EGraph(save_egglog_string=True, seminaive=False)

    egraph.register(rule(rel(x)).then(rel(x + 1)))

    assert ":naive" in egraph.as_egglog_string


@pytest.mark.parametrize("bidirectional", [False, True], ids=["rewrite", "birewrite"])
def test_global_rule_options_reach_saved_rewrites(*, bidirectional: bool) -> None:
    class RewriteConfigExpr(Expr):
        def __init__(self, value: i64Like) -> None: ...

    @function
    def rewrite_config_lookup(value: i64) -> i64: ...

    @function
    def rewrite_config_read(value: i64) -> i64:
        return rewrite_config_lookup(value)

    grounded = relation("rewrite_config_grounded", i64)
    x = var("x", i64, egg_name="__expr_0")
    rewrite_builder = birewrite if bidirectional else rewrite
    registered = rewrite_builder(RewriteConfigExpr(x)).to(RewriteConfigExpr(rewrite_config_read(x)), grounded(x))
    egraph = EGraph(
        set_(rewrite_config_lookup(i64(1))).to(i64(2)),
        grounded(i64(1)),
        RewriteConfigExpr(i64(1)),
        save_egglog_string=True,
        seminaive=False,
        no_decomp=True,
    )

    egraph.register(registered)
    report = egraph.run(1)
    egraph.check(eq(RewriteConfigExpr(i64(1))).to(RewriteConfigExpr(i64(2))))

    transcript = egraph.as_egglog_string
    expected_rule_count = 2 if bidirectional else 1
    expected_decl_type = BiRewriteDecl if bidirectional else RewriteDecl
    assert report.num_matches_per_rule
    assert all(isinstance(decl, expected_decl_type) for decl in report.num_matches_per_rule)
    assert transcript.count(":naive") == expected_rule_count
    assert transcript.count(":no-decomp") == expected_rule_count
    egg_bindings.EGraph().parse_and_run_program(transcript)


@pytest.mark.parametrize("egg_name", ["check", "true", "_", "@internal"])
def test_saved_transcript_rejects_parser_sensitive_explicit_callable_name(egg_name: str) -> None:
    rel = relation(f"python_name_{egg_name}", egg_fn=egg_name)
    egraph = EGraph(save_egglog_string=True)

    with pytest.raises(ValueError, match="Explicit Egglog callable name"):
        egraph.register(rel())

    assert not egraph.as_egglog_string


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_empty_explicit_backend_names_are_treated_as_unspecified(*, save_egglog_string: bool) -> None:
    class EmptyNamed(Expr, egg_sort=""):
        def __init__(self) -> None: ...

    marker = relation("empty_explicit_backend_name", EmptyNamed, egg_fn="")
    egraph = EGraph(marker(EmptyNamed()), save_egglog_string=save_egglog_string)

    egraph.check(marker(EmptyNamed()))
    if save_egglog_string:
        egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        pytest.param("a\rb", "a\nb", id="cr"),
        pytest.param("a\r\nb", "a\nb", id="crlf"),
    ],
)
def test_saved_transcript_preserves_string_newlines(value: str, normalized: str) -> None:
    marker = relation(f"transcript_newline_{value.count(chr(13))}_{len(value)}", String)
    egraph = EGraph(marker(String(value)), save_egglog_string=True)

    with pytest.raises(EggSmolError):
        egraph.check(marker(String(normalized)))

    transcript = egraph.as_egglog_string
    assert value in transcript
    egg_bindings.EGraph().parse_and_run_program(transcript)


@pytest.mark.parametrize("message", ['bad "quote"', "bad \\ slash"])
def test_saved_transcript_escapes_panic_message(message: str) -> None:
    trigger = relation("escaped_panic_message_trigger")
    panic_rules = ruleset(rule(trigger()).then(panic(message)), name="escaped-panic-message")
    egraph = EGraph(trigger(), save_egglog_string=True)

    with pytest.raises(EggSmolError) as exc_info:
        egraph.run(panic_rules)

    assert message in str(exc_info.value)
    egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


def test_saved_transcript_allows_reserved_callable_name_in_nested_expression() -> None:
    @function(egg_fn="check")
    def nested_check(value: i64Like) -> i64: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(set_(nested_check(1)).to(i64(7)))

    assert egraph.lookup_function_value(nested_check(1)) == i64(7)
    egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string + "\n(check (= (check 1) 7))")


def test_saved_transcript_allows_command_head_name_in_rule_action() -> None:
    source = relation("rule_action_command_head_source")
    target = relation("rule_action_command_head_target", egg_fn="check")
    action_rules = ruleset(rule(source()).then(target()), name="rule-action-command-head")
    egraph = EGraph(source(), save_egglog_string=True)

    egraph.run(action_rules)

    egraph.check(target())
    egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


def test_saved_transcript_rejects_equality_head_name_in_rule_fact() -> None:
    source = relation("equality_head_source", i64, i64)
    claimed = relation("equality_head_claimed", i64, i64, egg_fn="=")  # type: ignore[call-overload]
    target = relation("equality_head_target")
    x, y = vars_("x y", i64)
    equality_head_rules = ruleset(rule(source(x, y), claimed(x, y)).then(target()), name="equality-head")

    direct = EGraph(source(i64(1), i64(2)))
    direct.run(equality_head_rules)
    assert not direct.check_bool(target())

    recorded = EGraph(source(i64(1), i64(2)), save_egglog_string=True)
    with pytest.raises(ValueError, match="cannot be used as a fact"):
        recorded.run(equality_head_rules)


def test_saved_transcript_allows_equality_head_name_as_standalone_action() -> None:
    claimed = relation("standalone_equality_head", i64, i64, egg_fn="=")  # type: ignore[call-overload]
    egraph = EGraph(save_egglog_string=True)

    egraph.register(claimed(i64(1), i64(2)))

    assert egraph.function_size(claimed) == 1
    egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


def test_saved_transcript_rejects_literal_explicit_sort_name() -> None:
    class LiteralNamed(Expr, egg_sort="true"):
        def __init__(self) -> None: ...

    egraph = EGraph(save_egglog_string=True)

    with pytest.raises(ValueError, match="Explicit Egglog sort name"):
        egraph.register(LiteralNamed())

    assert not egraph.as_egglog_string


def test_saved_transcript_rejects_unparseable_explicit_sort_name() -> None:
    class UnparseableNamed(Expr, egg_sort="has space"):
        def __init__(self) -> None: ...

    egraph = EGraph(save_egglog_string=True)

    with pytest.raises(ValueError, match="Explicit Egglog sort name"):
        egraph.register(UnparseableNamed())

    assert not egraph.as_egglog_string


@pytest.mark.parametrize("egg_name", ["_", "@internal"])
def test_saved_transcript_rejects_context_sensitive_explicit_sort_name(egg_name: str) -> None:
    class ContextSensitiveNamed(Expr, egg_sort=egg_name):
        def __init__(self) -> None: ...

    egraph = EGraph(save_egglog_string=True)

    with pytest.raises(ValueError, match="Explicit Egglog sort name"):
        egraph.register(Map[ContextSensitiveNamed, i64].empty())

    assert not egraph.as_egglog_string


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("has space", id="space"),
        pytest.param("x)", id="parenthesis"),
        pytest.param("x;y", id="comment"),
        pytest.param("x\ny", id="newline"),
    ],
)
def test_saved_transcript_rejects_unparseable_let_name(name: str) -> None:
    egraph = EGraph(save_egglog_string=True)

    with pytest.raises(ValueError, match="does not serialize as one Egglog symbol"):
        egraph.let(name, i64(1))

    assert not egraph.as_egglog_string


def test_direct_egraph_still_accepts_non_source_let_name() -> None:
    egraph = EGraph()

    value = egraph.let("has space", i64(1))

    egraph.check(eq(value).to(i64(1)))


@pytest.mark.parametrize("name", ["has space", "x;y", "x\ny"])
def test_saved_transcript_rejects_unparseable_ruleset_name(name: str) -> None:
    named_ruleset = ruleset(name=name)

    EGraph().run(named_ruleset)

    recorded = EGraph(save_egglog_string=True)
    with pytest.raises(ValueError, match="Ruleset name"):
        recorded.run(named_ruleset)
    assert not recorded.as_egglog_string


def test_saved_transcript_rejects_literal_explicit_variable_name() -> None:
    source = relation("literal_variable_source", i64)
    destination = relation("literal_variable_destination", i64)
    x = var("x", i64, egg_name="0")

    direct = EGraph(source(i64(7)))
    direct.register(rule(source(x)).then(destination(x)))
    direct.run(1)
    direct.check(destination(i64(7)))

    recorded = EGraph(source(i64(7)), save_egglog_string=True)
    with pytest.raises(ValueError, match="Egglog variable name"):
        recorded.register(rule(source(x)).then(destination(x)))


@pytest.mark.parametrize(
    ("python_name", "egg_name"),
    [
        pytest.param("", None, id="default-wildcard"),
        pytest.param("has space", None, id="default-space"),
        pytest.param("x", "@x", id="internal"),
    ],
)
def test_saved_transcript_rejects_non_source_variable_name(python_name: str, egg_name: str | None) -> None:
    source = relation("non_source_variable_source", i64)
    destination = relation("non_source_variable_destination", i64)
    x = var(python_name, i64, egg_name=egg_name)

    direct = EGraph(source(i64(7)))
    direct.register(rule(source(x)).then(destination(x)))
    direct.run(1)
    direct.check(destination(i64(7)))

    recorded = EGraph(source(i64(7)), save_egglog_string=True)
    with pytest.raises(ValueError, match="Egglog variable name"):
        recorded.register(rule(source(x)).then(destination(x)))


def test_saved_transcript_rejects_variable_wildcard_semantic_drift() -> None:
    source = relation("wildcard_variable_source", i64, i64)
    destination = relation("wildcard_variable_destination")
    x = var("x", i64, egg_name="_")
    repeated_variable_rule = rule(source(x, x)).then(destination())

    direct = EGraph(source(i64(1), i64(2)))
    direct.register(repeated_variable_rule)
    direct.run(1)
    assert not direct.check_bool(destination())

    recorded = EGraph(source(i64(1), i64(2)), save_egglog_string=True)
    with pytest.raises(ValueError, match="Egglog variable name"):
        recorded.register(repeated_variable_rule)


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


def test_lookup_function_value_constructor_row() -> None:
    class A(Expr):
        def __init__(self, value: i64Like) -> None: ...

        def score(self) -> i64: ...

    egraph = EGraph(A(1))
    egraph.register(set_(A(1).score()).to(i64(7)))

    value = egraph.lookup_function_value(A(1))
    assert value is not None
    score = egraph.lookup_function_value(value.score())
    assert score is not None
    assert int(score) == 7
    assert egraph.lookup_function_value(A(2)) is None


def test_let_auto_prefixes_global_names(capfd: pytest.CaptureFixture[str]):
    egraph = EGraph(save_egglog_string=True)

    x = egraph.let("x", i64(1))
    egraph.check(eq(x).to(i64(1)))

    captured = capfd.readouterr()
    assert "should start with `$`" not in captured.err
    assert "(let $x " in egraph.as_egglog_string


def test_failed_check_does_not_materialize_shared_constructor_expressions() -> None:
    class CheckEdge(Expr):
        @classmethod
        def leaf(cls, value: StringLike) -> CheckEdge: ...

        @classmethod
        def pair(cls, left: CheckEdge, right: CheckEdge) -> CheckEdge: ...

    pair_rel = relation("check_pair_rel", CheckEdge, CheckEdge)
    leaf = CheckEdge.leaf("shared")
    pair = CheckEdge.pair(leaf, leaf)
    egraph = EGraph()

    with pytest.raises(EggSmolError, match="Check failed"):
        egraph.check(pair_rel(pair, pair))

    assert egraph.function_size(CheckEdge.leaf) == 0
    assert egraph.function_size(CheckEdge.pair) == 0


def test_synthetic_lets_skip_explicit_let_conflicts() -> None:
    class LetConflictNum(Expr):
        @classmethod
        def var(cls, v: StringLike) -> LetConflictNum: ...

        @classmethod
        def pair(cls, left: LetConflictNum, right: LetConflictNum) -> LetConflictNum: ...

    egraph = EGraph()
    explicit = egraph.let("__expr_0", LetConflictNum.var("explicit"))
    shared = LetConflictNum.var("shared")
    pair = LetConflictNum.pair(shared, shared)
    egraph.register(pair)

    egraph.check(eq(explicit).to(LetConflictNum.var("explicit")))
    assert egraph.function_size(LetConflictNum.var) == 2
    assert egraph.function_size(LetConflictNum.pair) == 1


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
@pytest.mark.parametrize("explicit_first", [False, True], ids=["generated-first", "explicit-first"])
def test_synthetic_let_names_reserve_explicit_lets_in_a_batch(
    *, save_egglog_string: bool, explicit_first: bool
) -> None:
    class BatchedLetConflict(Expr):
        @classmethod
        def leaf(cls, value: i64Like) -> BatchedLetConflict: ...

        @classmethod
        def pair(cls, left: BatchedLetConflict, right: BatchedLetConflict) -> BatchedLetConflict: ...

    explicit = let("__expr_0", BatchedLetConflict.leaf(2))
    shared = BatchedLetConflict.leaf(1)
    pair = BatchedLetConflict.pair(shared, shared)
    actions = (explicit, pair) if explicit_first else (pair, explicit)
    egraph = EGraph(save_egglog_string=save_egglog_string)

    egraph.register(*actions)

    assert egraph.function_size(BatchedLetConflict.leaf) == 2
    assert egraph.function_size(BatchedLetConflict.pair) == 1
    if save_egglog_string:
        egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


def test_synthetic_let_names_do_not_shadow_default_rewrite_variables() -> None:
    default_ruleset = ruleset(name="synthetic-let-shadow-default-rewrite")

    class LetShadowDefaultNum(Expr, ruleset=default_ruleset):
        def __init__(self, value: i64Like) -> None: ...

        @classmethod
        def make(cls, value: i64Like) -> LetShadowDefaultNum:
            return LetShadowDefaultNum(value)

        @classmethod
        def pair(cls, left: LetShadowDefaultNum, right: LetShadowDefaultNum) -> LetShadowDefaultNum: ...

    egraph = EGraph()
    shared = LetShadowDefaultNum(3)
    egraph.register(LetShadowDefaultNum.pair(shared, shared))
    egraph.register(LetShadowDefaultNum.make(i64(1)))
    egraph.run(run(default_ruleset))
    egraph.check(eq(LetShadowDefaultNum.make(i64(1))).to(LetShadowDefaultNum(i64(1))))


def test_shared_expression_discovery_handles_deep_dags() -> None:
    class DeepDag(Expr):
        @classmethod
        def leaf(cls) -> DeepDag: ...

        @classmethod
        def wrap(cls, value: DeepDag) -> DeepDag: ...

        @classmethod
        def pair(cls, left: DeepDag, right: DeepDag) -> DeepDag: ...

    expr = DeepDag.leaf()
    for index in range(1_050):
        expr = DeepDag.wrap(expr)
        if index % 100 == 99:
            expr = DeepDag.pair(expr, expr)

    egraph = EGraph(expr)

    assert egraph.function_size(DeepDag.wrap) == 1_050


def test_save_egglog_string_defaults_to_disabled() -> None:
    egraph = EGraph()

    with pytest.raises(ValueError, match="save_egglog_string=True"):
        _ = egraph.as_egglog_string


@pytest.mark.parametrize(
    "name",
    [
        "",
        "true",
        "false",
        "123",
        "-1",
        "1e3",
        "NaN",
        "inf",
        "-inf",
        "=",
        "sort",
        "datatype",
        "datatype*",
        "function",
        "constructor",
        "relation",
        "ruleset",
        "unstable-combined-ruleset",
        "rule",
        "rewrite",
        "birewrite",
        "run",
        "run-schedule",
        "extract",
        "check",
        "prove",
        "prove-exists",
        "push",
        "pop",
        "print-stats",
        "print-function",
        "print-size",
        "input",
        "output",
        "include",
        "fail",
        "let",
        "set",
        "delete",
        "subsume",
        "union",
        "panic",
        "for",
        "with-ruleset",
        "with-dynamic-cost",
        "set-cost",
        "let-scheduler",
        "multi-extract",
        "keep-best",
        "print-table-stats",
        "primitive",
    ],
)
def test_generated_relation_names_avoid_egglog_parser_tokens(name: str) -> None:
    relation_with_parser_token_name = relation(name)
    egraph = EGraph(save_egglog_string=True)

    egraph.register(relation_with_parser_token_name())

    egraph.check(relation_with_parser_token_name())


@pytest.mark.parametrize("generated_first", [True, False], ids=["generated-first", "explicit-first"])
def test_generated_relation_names_reserve_explicit_backend_names(generated_first: bool) -> None:
    generated = relation("batch_generated_backend_name", i64)
    generated_candidate = f"{__name__.replace('.', '_')}_batch_generated_backend_name"
    explicit = relation("explicit_relation", i64, egg_fn=generated_candidate)
    generated_action = generated(i64(1))
    explicit_action = explicit(i64(2))
    actions = (generated_action, explicit_action) if generated_first else (explicit_action, generated_action)
    egraph = EGraph(save_egglog_string=True)

    egraph.register(*actions)

    egraph.check(generated(i64(1)), explicit(i64(2)))
    egraph.check_fail(generated(i64(2)))
    egraph.check_fail(explicit(i64(1)))


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
@pytest.mark.parametrize("explicit_first", [False, True], ids=["generated-first", "explicit-first"])
def test_synthetic_let_names_reserve_explicit_backend_names(*, save_egglog_string: bool, explicit_first: bool) -> None:
    class SyntheticLetNode(Expr):
        @classmethod
        def leaf(cls, value: i64Like) -> SyntheticLetNode: ...

        @classmethod
        def pair(cls, left: SyntheticLetNode, right: SyntheticLetNode) -> SyntheticLetNode: ...

    explicit = relation("synthetic_let_explicit", egg_fn="$__expr_0")
    shared = SyntheticLetNode.leaf(1)
    pair = SyntheticLetNode.pair(shared, shared)
    actions = (explicit(), pair) if explicit_first else (pair, explicit())
    egraph = EGraph(save_egglog_string=save_egglog_string)

    egraph.register(*actions)

    egraph.check(explicit())
    assert egraph.function_size(SyntheticLetNode.pair) == 1
    if save_egglog_string:
        egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_synthetic_lets_do_not_capture_unbound_variables(*, save_egglog_string: bool) -> None:
    class SyntheticLetVariable(Expr):
        @classmethod
        def leaf(cls, value: i64Like) -> SyntheticLetVariable: ...

        @classmethod
        def pair(cls, left: SyntheticLetVariable, right: SyntheticLetVariable) -> SyntheticLetVariable: ...

    x = var("synthetic_let_x", i64)
    shared = SyntheticLetVariable.leaf(x)
    egraph = EGraph(save_egglog_string=save_egglog_string)

    with pytest.raises(EggSmolError, match="Unbound symbol"):
        egraph.register(SyntheticLetVariable.pair(shared, shared))

    if save_egglog_string:
        with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
            egraph.register(SyntheticLetVariable.leaf(1))
    else:
        egraph.register(SyntheticLetVariable.leaf(1))


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_late_explicit_backend_name_cannot_replace_registered_generated_relation(*, save_egglog_string: bool) -> None:
    generated = relation("late_explicit_backend_name", i64)
    egraph = EGraph(save_egglog_string=save_egglog_string)
    egraph.register(generated(i64(1)))
    transcript = egraph.as_egglog_string if save_egglog_string else None

    generated_name = f"{__name__.replace('.', '_')}_late_explicit_backend_name"
    explicit = relation("late_explicit_relation", i64, egg_fn=generated_name)
    with pytest.raises(ValueError, match="already registered"):
        egraph.register(explicit(i64(2)))

    egraph.check(generated(i64(1)))
    if transcript is not None:
        assert egraph.as_egglog_string.startswith(transcript)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_duplicate_explicit_backend_name_does_not_publish_a_failed_alias(*, save_egglog_string: bool) -> None:
    @function(egg_fn="duplicate_explicit_backend")
    def first(value: i64Like) -> i64: ...

    @function(egg_fn="duplicate_explicit_backend")
    def second(value: i64Like) -> i64: ...

    egraph = EGraph(set_(first(1)).to(i64(10)), save_egglog_string=save_egglog_string)
    transcript = egraph.as_egglog_string if save_egglog_string else None

    with pytest.raises(ValueError, match="already registered"):
        egraph.register(set_(second(2)).to(i64(20)))

    assert egraph.lookup_function_value(first(1)) == i64(10)
    assert "first" in str(egraph.freeze())
    if transcript is not None:
        assert egraph.as_egglog_string == transcript
        egg_bindings.EGraph().parse_and_run_program(transcript)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_late_explicit_backend_name_cannot_replace_registered_generated_sort(*, save_egglog_string: bool) -> None:
    class GeneratedLateSort(Expr):
        def __init__(self) -> None: ...

    egraph = EGraph(save_egglog_string=save_egglog_string)
    egraph.register(GeneratedLateSort())
    transcript = egraph.as_egglog_string if save_egglog_string else None

    class ExplicitLateSort(Expr, egg_sort=f"{__name__}.GeneratedLateSort"):
        def __init__(self) -> None: ...

    with pytest.raises(ValueError, match="already registered"):
        egraph.register(ExplicitLateSort())

    egraph.check(GeneratedLateSort())
    if transcript is not None:
        assert egraph.as_egglog_string.startswith(transcript)


def test_generated_relation_names_avoid_command_macro_heads() -> None:
    source = relation("command_macro_source")
    generated = relation("unstable-fresh!")
    copy = ruleset(rule(source()).then(generated()))
    egraph = EGraph(source(), save_egglog_string=True)

    egraph.run(copy)

    egraph.check(generated())


@pytest.mark.parametrize("name", ["1.0", "-1.0", "+.5", "1.", "1E+3"])
def test_generated_sort_names_avoid_egglog_float_literals(name: str) -> None:
    def init(_self: object) -> None: ...

    numeric_name_expr = type(name, (Expr,), {"__module__": __name__, "__init__": init})
    expr = numeric_name_expr()
    egraph = EGraph(save_egglog_string=True)

    egraph.register(expr)

    egraph.check(expr)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("has space", id="space"),
        pytest.param("x;y", id="comment"),
        pytest.param("x(y)", id="parentheses"),
        pytest.param("x\ny", id="newline"),
    ],
)
def test_generated_sort_names_are_safe_in_saved_source(name: str) -> None:
    def init(_self: object) -> None: ...

    generated_expr = type(name, (Expr,), {"__module__": __name__, "__init__": init})
    expr = generated_expr()
    egraph = EGraph(expr, save_egglog_string=True)

    egraph.check(expr)
    egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


def test_saved_egglog_transcript_close_is_idempotent() -> None:
    egraph = EGraph(save_egglog_string=True)
    egraph.let("x", i64(1))
    assert egraph.as_egglog_string

    egraph.close()
    egraph.close()

    with pytest.raises(ValueError, match="has been closed"):
        _ = egraph.as_egglog_string


def test_saved_egglog_transcript_close_rejects_commands_before_mutation() -> None:
    class ClosedExpr(Expr):
        def __init__(self, value: i64Like) -> None: ...

    egraph = EGraph(ClosedExpr(1), save_egglog_string=True)
    before = egraph.freeze()
    egraph.close()

    with pytest.raises(ValueError, match="transcript has been closed"):
        egraph.register(ClosedExpr(2))

    assert egraph.freeze() == before


def test_close_without_a_saved_transcript_is_a_noop() -> None:
    egraph = EGraph()

    egraph.close()
    value = egraph.let("after_close", i64(1))

    egraph.check(eq(value).to(i64(1)))


def test_saved_egglog_transcript_is_shared_across_push_and_pop() -> None:
    egraph = EGraph(save_egglog_string=True)
    egraph.let("outer", i64(1))
    outer_transcript = egraph.as_egglog_string

    egraph.push()
    egraph.let("inner", i64(2))
    egraph.pop()
    assert egraph.as_egglog_string.startswith(outer_transcript)
    assert "(let $inner 2)" in egraph.as_egglog_string


def test_saved_egglog_string_uses_qualified_generated_sort_and_function_names() -> None:
    class Num(Expr):
        @classmethod
        def var(cls, v: StringLike) -> Num: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(Num.var("x"))
    egglog_string = egraph.as_egglog_string

    qualified_sort = f"{__name__}.Num"
    qualified_var = f"{__name__.replace('.', '_')}_Num_var"
    assert f"(sort {qualified_sort})" in egglog_string
    assert f"(constructor {qualified_var} (String) {qualified_sort})" in egglog_string


def test_generated_names_do_not_collide_with_lazy_builtin_declarations() -> None:
    class String(Expr):
        def __init__(self, value: i64Like) -> None: ...

    @function
    def log(value: i64Like) -> i64: ...

    egraph = EGraph()
    user_string = String(1)
    egraph.register(user_string, set_(log(1)).to(i64(2)))

    egraph.check(eq(user_string).to(String(1)), eq(log(1)).to(i64(2)))


def test_parameterized_sort_names_use_allocated_argument_names() -> None:
    egraph = EGraph(save_egglog_string=True)

    egraph.register(Map[i64, BigRat].empty())

    assert "(sort Map[i64,BigRat] (Map i64 BigRat))" in egraph.as_egglog_string


def test_registering_bare_variable_expression_raises() -> None:
    egraph = EGraph()

    with pytest.raises(ValueError, match="must be calls"):
        egraph.register(var("x", i64))


def test_registering_let_reference_expression_raises() -> None:
    egraph = EGraph()
    x = egraph.let("x", i64(1))

    with pytest.raises(ValueError, match="must be calls"):
        egraph.register(x)


def test_nested_rule_lowering_does_not_reuse_top_level_synthetic_lets() -> None:
    class NestedRuleEdge(Expr):
        @classmethod
        def leaf(cls, value: StringLike) -> NestedRuleEdge: ...

        @classmethod
        def pair(cls, left: NestedRuleEdge, right: NestedRuleEdge) -> NestedRuleEdge: ...

    egraph = EGraph()
    done_rel = relation("done_rel_ctx")
    shared = NestedRuleEdge.leaf("shared")
    pair = NestedRuleEdge.pair(shared, shared)

    egraph.register(pair)
    egraph.register(rule(eq(var("x", NestedRuleEdge)).to(pair)).then(done_rel()))
    egraph.register(subsume(shared))

    egraph.run(1)
    egraph.check_fail(done_rel())


def test_top_level_action_factors_duplicate_sibling_edges() -> None:
    class DuplicateEdge(Expr):
        @classmethod
        def leaf(cls, value: StringLike) -> DuplicateEdge: ...

        @classmethod
        def pair(cls, left: DuplicateEdge, right: DuplicateEdge) -> DuplicateEdge: ...

    leaf = DuplicateEdge.leaf("shared")
    first_pair = DuplicateEdge.pair(leaf, leaf)
    second_pair = DuplicateEdge.pair(first_pair, first_pair)
    egraph = EGraph(save_egglog_string=True)

    egraph.register(DuplicateEdge.pair(second_pair, second_pair))

    lines = egraph.as_egglog_string.splitlines()
    let_lines = [line for line in lines if line.startswith("(let $__expr_")]
    assert len(let_lines) == 3
    assert let_lines[0].startswith("(let $__expr_0 (")
    assert let_lines[0].endswith(' "shared"))')
    assert let_lines[1].startswith("(let $__expr_1 (")
    assert let_lines[1].endswith(" $__expr_0 $__expr_0))")
    assert let_lines[2].startswith("(let $__expr_2 (")
    assert let_lines[2].endswith(" $__expr_1 $__expr_1))")
    assert sum(line.endswith(" $__expr_2 $__expr_2)") for line in lines) == 1


def test_freeze_omits_synthetic_let_bindings() -> None:
    class FreezeLetNum(Expr):
        @classmethod
        def var(cls, value: StringLike) -> FreezeLetNum: ...

        @classmethod
        def pair(cls, left: FreezeLetNum, right: FreezeLetNum) -> FreezeLetNum: ...

    shared = FreezeLetNum.var("shared")
    egraph = EGraph(save_egglog_string=True)
    egraph.register(FreezeLetNum.pair(shared, shared))

    assert "$__expr_" in egraph.as_egglog_string
    frozen = str(egraph.freeze())
    assert "FreezeLetNum.pair" in frozen
    assert "$__expr_" not in frozen


def test_integer_run_accepts_a_combined_ruleset() -> None:
    seen = relation("combined_seen", i64)
    copied = relation("combined_copied", i64)
    x = var("x", i64)
    copy_rules = ruleset(rule(seen(x)).then(copied(x)))
    empty_rules = ruleset()
    egraph = EGraph(seen(i64(1)))

    egraph.run(1, ruleset=copy_rules | empty_rules)

    egraph.check(copied(i64(1)))


def test_higher_order_builtin_callback_materializes_parameterized_builtin_dummy_args() -> None:
    input_map = Map[i64, Maybe[f64]].empty().insert(i64(1), Maybe[f64].some(f64(2.5)))
    initial: Map[i64, f64] = Map[i64, f64].empty()
    expected = Map[i64, f64].empty().insert(i64(1), f64(2.5))
    check_eq(map_fold_kv(lambda result, key, value: result.insert(key, value.unwrap()), initial, input_map), expected)


def test_higher_order_builtin_callback_materializes_rational_builtin_dummy_args() -> None:
    input_map = Map[i64, Rational].empty().insert(i64(1), Rational(1, 2))
    initial: Map[i64, f64] = Map[i64, f64].empty()
    expected = Map[i64, f64].empty().insert(i64(1), f64(1.5))
    check_eq(
        map_fold_kv(lambda result, key, value: result.insert(key, value.to_f64() + 1.0), initial, input_map),
        expected,
    )


def test_map_fold_generic_negation_callback_is_concretized() -> None:
    input_map = Map[i64, f64].empty().insert(i64(1), f64(2.5)).insert(i64(2), f64(-4.0))
    initial: Map[i64, f64] = Map[i64, f64].empty()
    expected = Map[i64, f64].empty().insert(i64(1), f64(-2.5)).insert(i64(2), f64(4.0))

    check_eq(map_fold_kv(lambda result, key, value: result.insert(key, -value), initial, input_map), expected)


def test_map_and_set_length_primitives() -> None:
    check_eq(Map[i64, f64].empty().insert(i64(1), f64(2.0)).insert(i64(2), f64(3.0)).length(), i64(2))
    check_eq(Set(i64(1), i64(2)).length(), i64(2))


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

    initial: Map[i64, Box] = Map[i64, Box].empty()
    expected = Map[i64, Box].empty().insert(i64(1), Box(i64(2)))
    actual = map_fold_kv(
        lambda result, key, value: result.insert(key, Box(value)),
        initial,
        Map[i64, i64].empty().insert(i64(1), i64(2)),
    )
    check_eq(cast("BaseExpr", actual), cast("BaseExpr", expected))


def test_named_builtin_return_function_is_eager() -> None:
    @function
    def add1(x: i64Like) -> i64:
        return cast("i64", x) + 1

    check_eq(add1(i64(2)), i64(3))


def test_named_container_return_function_is_eager() -> None:
    @function
    def singleton_map(x: i64Like) -> Map[i64, i64]:
        runtime_x = cast("i64", x)
        return Map[i64, i64].empty().insert(runtime_x, runtime_x + 1)

    expected = Map[i64, i64].empty().insert(i64(2), i64(3))
    check_eq(singleton_map(i64(2)), expected)


def test_reverse_args_eager_method_preserves_python_argument_order() -> None:
    class ReverseBody(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def ordered(self, other: i64Like) -> Pair[ReverseBody, i64]:
            return Pair(self, cast("i64", other))

    extracted = EGraph().extract(ReverseBody(1).ordered(i64(2)))

    assert extracted.value == (ReverseBody(1), i64(2))


def test_reverse_args_rewrite_method_preserves_python_argument_order() -> None:
    reverse_ruleset = ruleset(name="reverse-args-rewrite")

    class ReverseRewrite(Expr, ruleset=reverse_ruleset):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def select_self(self, other: ReverseRewrite) -> ReverseRewrite:
            return self

    expr = ReverseRewrite(1).select_self(ReverseRewrite(2))
    egraph = EGraph(expr)

    egraph.run(reverse_ruleset)
    egraph.check(eq(expr).to(ReverseRewrite(1)))


def test_reverse_args_bodyless_method_uses_backend_argument_order() -> None:
    class ReverseLookup(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def lookup(self, key: i64Like) -> String:
            return None  # type: ignore[return-value]  # None denotes a bodyless symbolic function.

    value = ReverseLookup(1).lookup(i64(2))
    egraph = EGraph()
    egraph.register(set_(value).to(String("found")), set_cost(value, 7))

    egraph.check(eq(value).to(String("found")), eq(get_cost(value)).to(i64(7)))
    assert egraph.function_values(ReverseLookup.lookup) == {value: String("found")}


def test_reverse_args_bodyless_constructor_round_trips_backend_argument_order() -> None:
    class ReverseResult(Expr): ...

    class ReverseSource(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def make(self, key: i64Like) -> ReverseResult: ...

    expr = ReverseSource(1).make(i64(2))

    assert expr_parts(EGraph(expr).extract(expr)) == expr_parts(expr)


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
        return None  # type: ignore[return-value]  # None denotes a bodyless symbolic function.

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


def test_mutating_eqsort_function_body_can_construct_a_bound_result() -> None:
    class IntBox(Expr):
        def __init__(self, value: i64Like) -> None: ...

        def __add__(self, other: IntBox) -> IntBox: ...

    @function(mutates_first_arg=True)
    def increment(box: IntBox) -> None:
        box.__replace_expr__(box + IntBox(1))

    box = IntBox(10)
    increment(box)
    egraph = EGraph()
    incremented = egraph.let("incremented", box)
    egraph.check(eq(incremented).to(IntBox(10) + IntBox(1)))


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
    assert egraph.extract(f64(2.0) + 1).value == 3.0


def test_f64_math_primitives() -> None:
    egraph = EGraph()
    assert egraph.extract(f64(1.0).exp()).value == pytest.approx(math.e)
    assert egraph.extract(f64(math.e).log()).value == pytest.approx(1.0)
    assert egraph.extract(f64(4.0).sqrt()).value == pytest.approx(2.0)


def test_f64_is_finite_predicate() -> None:
    egraph = EGraph()
    egraph.check(f64(1.0).is_finite())
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(EggSmolError):
            egraph.check(f64(value).is_finite())


def test_rational_like_operations() -> None:
    assert "RationalLike" in egg_builtins.__all__
    assert EGraph().extract(Rational(1, 2) + Fraction(1, 3)).value == Fraction(5, 6)
    assert EGraph().extract(Fraction(1, 3) + Rational(1, 2)).value == Fraction(5, 6)
    assert EGraph().extract(Rational(1, 2) - 1).value == Fraction(-1, 2)
    assert EGraph().extract(1 - Rational(1, 2)).value == Fraction(1, 2)
    assert EGraph().extract(Rational(2, 3) * Fraction(3, 4)).value == Fraction(1, 2)
    assert EGraph().extract(Fraction(3, 4) * Rational(2, 3)).value == Fraction(1, 2)
    assert EGraph().extract(Rational(1, 2) / 2).value == Fraction(1, 4)
    assert EGraph().extract(1 / Rational(1, 2)).value == Fraction(2, 1)
    assert EGraph().extract(Rational(2, 1) ** 3).value == Fraction(8, 1)
    assert EGraph().extract(2 ** Rational(3, 1)).value == Fraction(8, 1)
    assert EGraph().extract(Rational(1, 2).min(Fraction(1, 3))).value == Fraction(1, 3)
    assert EGraph().extract(Rational(1, 2).max(1)).value == Fraction(1, 1)

    egraph = EGraph()
    egraph.check(Rational(1, 2) < Fraction(2, 3))
    egraph.check(Rational(2, 3) > Fraction(1, 2))
    egraph.check(Rational(1, 2) <= Fraction(1, 2))
    egraph.check(Rational(1, 2) >= Fraction(1, 2))


def test_rational_partial_operations_remain_undefined() -> None:
    with pytest.raises(EggSmolError):
        EGraph().extract(Rational(1, 2) / 0)
    with pytest.raises(EggSmolError):
        EGraph().extract(Rational(2, 1) ** -1)
    with pytest.raises(EggSmolError):
        EGraph().check(Rational(2, 3) < Fraction(1, 2))


def test_bigrat_to_i64_is_exact_and_bounded() -> None:
    assert EGraph().extract(BigRat(4, 2).to_i64()) == i64(2)
    assert EGraph().extract(BigRat(1, 2) + i64(1)).value == Fraction(3, 2)

    for numerator, denominator in [(1, 2), (2**63, 1), (-(2**63) - 1, 1)]:
        value = BigRat(BigInt.from_string(str(numerator)), BigInt(denominator))
        with pytest.raises(EggSmolError, match="primitive to-i64 failed"):
            EGraph().extract(value.to_i64())


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

    def test_map_duplicate_key_uses_latest_value(self):
        m = Map[String, i64].empty().insert(String("a"), i64(1)).insert(String("a"), i64(2))

        assert EGraph().extract(m).value == {String("a"): i64(2)}

    def test_set(self):
        assert EGraph().extract(Set[i64].empty()).value == set()
        s = Set(i64(1), i64(2))
        assert s.value == {i64(1), i64(2)}

        assert set(s) == {i64(1), i64(2)}
        assert len(s) == 2
        assert i64(1) in s
        assert i64(3) not in s
        assert list(Set(i64(1), i64(1))) == [i64(1)]

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

    def test_extract_nested_maps_preserves_empty_map_type_params(self):
        inner = Map[String, BigRat].empty().insert(String("x"), BigRat(2, 1))
        expr = Map[Map[String, BigRat], f64].empty().insert(inner, f64(1.0))

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

    def test_none_is_a_primitive_constant_default(self):
        missing = constant("missing_default", Maybe[i64], None)

        check_eq(missing, Maybe[i64].none())

    def test_none_is_a_primitive_class_variable_default(self):
        class Defaults(Expr):
            missing: ClassVar[Maybe[i64] | None] = None

        check_eq(cast("Maybe[i64]", Defaults.missing), Maybe[i64].none())

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
                return B.wrap(value)  # type: ignore[return-value]  # noqa: PLE0101 - symbolic constructor body

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
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            pytest.param({"cost": 1}, "Cost can only be set for constructors", id="cost"),
            pytest.param({"unextractable": True}, "Unextractable can only be set for constructors", id="unextractable"),
            pytest.param(
                {"cost": 1, "unextractable": True},
                "Cost can only be set for constructors",
                id="cost-before-unextractable",
            ),
        ],
    )
    def test_eqsort_merge_rejects_constructor_options_before_body(
        self, options: dict[str, object], message: str
    ) -> None:
        body_calls = 0
        merge_calls = 0

        def merge(old: A, _new: A) -> A:
            nonlocal merge_calls
            merge_calls += 1
            return old

        @function(merge=merge, **options)  # type: ignore[call-overload]
        def f() -> A:
            nonlocal body_calls
            body_calls += 1
            return A()

        with pytest.raises(ValueError, match=message):
            f()

        assert body_calls == 0
        assert merge_calls == 0

    def test_primitive_function_ruleset_subsume_rejected(self):
        r = ruleset()

        @function(ruleset=r, subsume=True)  # type: ignore[type-var]  # Deliberately invalid runtime API call.
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
            EGraph().register(constant("no_default", A, ruleset=r))  # type: ignore[call-overload]

    def test_primitive_constant_default_cannot_use_explicit_ruleset(self):
        r = ruleset()

        with pytest.raises(ValueError, match="Primitive-returning defaults cannot use an explicit ruleset"):
            EGraph().register(
                constant("primitive_default", i64, i64(1), ruleset=r)  # type: ignore[call-overload]
            )

    def test_eqsort_constant_default_cannot_use_merge(self):
        merge_calls = 0

        def merge(old: A, _new: A) -> A:
            nonlocal merge_calls
            merge_calls += 1
            return old

        with pytest.raises(ValueError, match="Eqsort-returning callables with bodies cannot use merge"):
            EGraph().register(
                constant("default_merge", A, A(), merge=merge)  # type: ignore[call-overload]
            )
        assert merge_calls == 0

    def test_primitive_constant_default_cannot_use_merge(self):
        with pytest.raises(ValueError, match="Primitive-returning callables with bodies cannot use merge"):
            EGraph().register(
                constant(  # type: ignore[call-overload]
                    "primitive_default_merge", i64, i64(1), merge=lambda old, new: old.max(new)
                )
            )

    def test_unit_constant_cannot_use_merge(self):
        with pytest.raises(ValueError, match="Functions that return Unit cannot use merge"):
            EGraph().register(constant("unit_merge", Unit, merge=lambda old, _new: old))

    def test_eqsort_eager_body_cannot_use_merge(self):
        body_calls = 0
        merge_calls = 0

        def merge(old: A, _new: A) -> A:
            nonlocal merge_calls
            merge_calls += 1
            return old

        @function(merge=merge)
        def f() -> A:
            nonlocal body_calls
            body_calls += 1
            return A()

        with pytest.raises(ValueError, match="Eqsort-returning callables with bodies cannot use merge"):
            f()
        assert body_calls == 0
        assert merge_calls == 0

    def test_primitive_returning_functions_cannot_use_cost(self):
        @function(cost=1)  # type: ignore[type-var]  # Deliberately invalid runtime API call.
        def f() -> i64: ...

        with pytest.raises(ValueError, match="Primitive-returning callables cannot use cost"):
            f()

    def test_primitive_returning_functions_cannot_be_unextractable(self):
        @function(unextractable=True)  # type: ignore[type-var]  # Deliberately invalid runtime API call.
        def f() -> i64: ...

        with pytest.raises(ValueError, match="Primitive-returning callables cannot be unextractable"):
            f()

    def test_builtin_callables_cannot_use_merge(self):
        @function(builtin=True, merge=lambda old, new: old)  # type: ignore[call-overload]
        def f() -> i64: ...

        with pytest.raises(ValueError, match="Builtin callables cannot use merge"):
            f()

    def test_primitive_body_cannot_use_builtin(self):
        body_calls = 0

        @function(builtin=True)
        def f() -> i64:
            nonlocal body_calls
            body_calls += 1
            return i64(1)

        with pytest.raises(ValueError, match="Builtin callables cannot have a body"):
            f()
        assert body_calls == 0

    def test_generic_primitive_body_cannot_use_builtin(self):
        body_calls = 0

        @function(builtin=True)
        def f(value: Vec[_BuiltinExprT]) -> i64:
            nonlocal body_calls
            body_calls += 1
            return value.length()

        with pytest.raises(ValueError, match="Builtin callables cannot have a body"):
            f(Vec(i64(1)))
        assert body_calls == 0

    def test_generic_builtin_cannot_use_explicit_ruleset(self):
        r = ruleset()

        @function(builtin=True, ruleset=r)  # type: ignore[call-overload]  # Deliberately invalid runtime API call.
        def f(value: Vec[_BuiltinExprT]) -> i64: ...

        with pytest.raises(ValueError, match="Builtin callables cannot use an explicit ruleset"):
            f(Vec(i64(1)))

    def test_primitive_body_cannot_use_merge(self):
        @function(merge=lambda old, new: old)
        def f() -> i64:
            return i64(1)

        with pytest.raises(ValueError, match="Primitive-returning callables with bodies cannot use merge"):
            f()

    def test_primitive_body_cannot_use_explicit_ruleset(self):
        r = ruleset()

        @function(ruleset=r)  # type: ignore[type-var]  # Deliberately invalid runtime API call.
        def f() -> i64:
            return i64(1)

        with pytest.raises(
            ValueError, match="Primitive-returning callables with bodies cannot use an explicit ruleset"
        ):
            f()

    def test_primitive_body_must_return_a_value(self):
        body_calls = 0

        @function
        def f() -> i64:
            nonlocal body_calls
            body_calls += 1
            return None  # type: ignore[return-value]  # Exercise a genuine body with a missing symbolic result.

        with pytest.raises(ValueError, match="Callable bodies must return a value"):
            f()
        assert body_calls == 1

    def test_eqsort_body_must_return_a_value(self):
        body_calls = 0

        @function
        def f() -> A:
            nonlocal body_calls
            body_calls += 1
            return None  # type: ignore[return-value]  # Exercise a genuine body with a missing symbolic result.

        with pytest.raises(ValueError, match="Callable bodies must return a value"):
            f()
        assert body_calls == 1

    def test_eqsort_body_cannot_use_merge(self):
        r = ruleset()

        @function(ruleset=r, merge=lambda old, new: old)  # type: ignore[call-overload]
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

            @method(subsume=True)  # type: ignore[type-var]  # Deliberately invalid runtime API call.
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

    some_expr = EGraph().extract(Maybe[f64].some(1.0))  # type: ignore[arg-type]  # Runtime conversion.
    assert some_expr.value is not None
    assert some_expr.value.value == 1.0

    assert EGraph().extract(Maybe[f64].some(1.0).unwrap()).value == 1.0  # type: ignore[arg-type]
    assert EGraph().extract(Maybe[f64].none().unwrap_or(2.5)).value == 2.5  # type: ignore[arg-type]


def test_higher_order_maybe_pair_and_catch_builtins():
    assert EGraph().extract(Maybe[i64].some(2).match(lambda x: x + 3, i64(0))).value == 5  # type: ignore[arg-type]
    assert EGraph().extract(Maybe[i64].none().match(lambda x: x + 3, i64(7))).value == 7

    pair = Pair(i64(2), i64(3))
    assert EGraph().extract(pair.left).value == 2
    assert EGraph().extract(pair.right).value == 3
    assert EGraph().extract(pair).value == (i64(2), i64(3))

    caught = EGraph().extract(catch(lambda: Maybe[i64].some(4).unwrap()))  # type: ignore[arg-type]
    assert caught.value is not None
    assert caught.value.value == 4

    failed = EGraph().extract(catch(lambda: Maybe[i64].none().unwrap()))
    assert failed.value is None


def test_nested_catch_match_with_different_inner_lambda_result_sort() -> None:
    expr = catch(lambda: i64(1)).match(
        lambda _: catch(lambda: f64(2.0)).match(lambda v: v, f64(0.0)),
        f64(9.0),
    )

    assert EGraph().extract(expr).value == 2.0


def test_maybe_conversion():
    @function
    def maybe_identity(x: Maybe[i64]) -> Maybe[i64]: ...

    assert expr_parts(maybe_identity(None)) == expr_parts(  # type: ignore[arg-type]  # Runtime conversion.
        maybe_identity(Maybe[i64].none())
    )


def test_pair_conversion() -> None:
    @function
    def pair_identity(pair: Pair[i64, i64]) -> Pair[i64, i64]: ...

    assert expr_parts(pair_identity((1, 2))) == expr_parts(  # type: ignore[arg-type]  # Runtime conversion.
        pair_identity(Pair(i64(1), i64(2)))
    )


@pytest.mark.parametrize("value", [(), (1,), (1, 2, 3)])
def test_pair_conversion_requires_exactly_two_items(value: tuple[int, ...]) -> None:
    @function
    def pair_identity(pair: Pair[i64, i64]) -> Pair[i64, i64]: ...

    with pytest.raises(ValueError, match=rf"tuple of length 2.*length {len(value)}"):
        pair_identity(value)  # type: ignore[arg-type]  # Deliberately malformed runtime conversion input.


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
                msg = "Should have matched A"  # type: ignore[unreachable]
                raise ValueError(msg)

    def test_literal(self):
        match i64(10):
            case i64(i):
                assert i == 10
            case _:
                msg = "Should have matched i64(10)"  # type: ignore[unreachable]
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
                msg = "Should have matched A"  # type: ignore[unreachable]
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


def test_lookup_function_value_records_materialized_constructor_arguments() -> None:
    class LookupReplayValue(Expr, egg_sort="LookupReplayValueSort"):
        @method(egg_fn="LookupReplayValue")
        def __init__(self, value: i64Like) -> None: ...

    @function(egg_fn="lookup-replay-result")
    def result(value: LookupReplayValue) -> i64: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(set_(result(LookupReplayValue(1))).to(i64(10)))

    assert egraph.lookup_function_value(result(LookupReplayValue(2))) is None
    assert egraph.function_size(LookupReplayValue) == 2

    replayed = egg_bindings.EGraph()
    outputs = replayed.parse_and_run_program(egraph.as_egglog_string + "\n(print-size)")
    sizes = outputs[-1]
    assert isinstance(sizes, egg_bindings.PrintAllFunctionsSize)
    assert dict(sizes.sizes)["LookupReplayValue"] == 2


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_lookup_function_value_with_unstable_fn_argument(*, save_egglog_string: bool) -> None:
    @function
    def add_one(value: i64) -> i64: ...

    @function
    def score(fn: UnstableFn[i64, i64]) -> i64: ...

    fn = UnstableFn[i64, i64](add_one)
    egraph = EGraph(set_(score(fn)).to(i64(7)), save_egglog_string=save_egglog_string)

    assert egraph.lookup_function_value(score(fn)) == i64(7)


def test_table_inspection_rejects_eager_primitives() -> None:
    @function
    def eager_plus_one(x: i64Like) -> i64:
        return cast("i64", x) + 1

    egraph = EGraph()
    assert egraph.extract(eager_plus_one(i64(1))) == i64(2)

    with pytest.raises(ValueError, match="table-backed"):
        egraph.function_size(eager_plus_one)
    with pytest.raises(ValueError, match="table-backed"):
        egraph.function_values(eager_plus_one)
    with pytest.raises(ValueError, match="table-backed"):
        egraph.lookup_function_value(eager_plus_one(i64(1)))

    @function
    def eager_text() -> String:
        return String("value")

    with pytest.raises(ValueError, match="table-backed"):
        egraph.input(eager_text, "unused.csv")


@pytest.mark.parametrize(
    "filename",
    ['input\r"\\.csv', "input\u0300.csv"],
    ids=["control-and-escapes", "combining-unicode"],
)
def test_saved_transcript_preserves_input_path(tmp_path: pathlib.Path, filename: str) -> None:
    @function
    def loaded() -> String: ...

    path = tmp_path / filename
    path.write_text("value\n")
    recorded = EGraph(save_egglog_string=True)

    recorded.input(loaded, str(path))
    recorded.check(eq(loaded()).to(String("value")))
    egg_bindings.EGraph().parse_and_run_program(recorded.as_egglog_string)


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
    assert egraph.extract(E(2), include_cost=True, extractor="greedy-dag") == (E(1) + E(1), 102)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_dynamic_cost_materializes_constructor_before_storing_cost(*, save_egglog_string: bool) -> None:
    class MaterializedCost(Expr):
        def __init__(self, value: i64Like) -> None: ...

    expression = MaterializedCost(7)
    egraph = EGraph(save_egglog_string=save_egglog_string)

    egraph.register(set_cost(expression, 9))

    assert egraph.function_size(MaterializedCost) == 1
    assert egraph.lookup_function_value(get_cost(expression)) == i64(9)
    if save_egglog_string:
        egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_dynamic_cost_rule_materializes_constructor_without_shadowing(*, save_egglog_string: bool) -> None:
    class RuleMaterializedCost(Expr):
        def __init__(self, value: i64Like) -> None: ...

    source = relation("dynamic_cost_materialize_source", i64)
    # Exercise a name used by compiler-generated rule lets: lowering must
    # allocate a distinct binding instead of shadowing this variable.
    value = var("value", i64, egg_name="__expr_0")
    materialize = ruleset(
        rule(source(value)).then(set_cost(RuleMaterializedCost(value), 11)),
        name="dynamic-cost-materialize",
    )
    expression = RuleMaterializedCost(8)
    egraph = EGraph(source(i64(8)), save_egglog_string=save_egglog_string)

    egraph.run(materialize)

    assert egraph.function_size(RuleMaterializedCost) == 1
    assert egraph.lookup_function_value(get_cost(expression)) == i64(11)
    if save_egglog_string:
        egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_dynamic_cost_does_not_create_a_missing_function_row(*, save_egglog_string: bool) -> None:
    @function
    def missing_target(value: i64Like) -> i64: ...

    expression = missing_target(1)
    egraph = EGraph(save_egglog_string=save_egglog_string)

    with pytest.raises(EggSmolError, match=r"lookup .* failed"):
        egraph.register(set_cost(expression, 7))

    assert egraph.function_size(missing_target) == 0
    assert egraph.lookup_function_value(get_cost(expression)) is None
    if save_egglog_string:
        egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)


def test_dynamic_cost_evaluates_each_argument_once(tmp_path: pathlib.Path) -> None:
    marker = tmp_path / "set-cost-calls"

    def record_call(value: int) -> int:
        marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
        return value

    class SingleEvaluationCost(Expr):
        def __init__(self, value: PyObject) -> None: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(set_cost(SingleEvaluationCost(PyObject(record_call)(PyObject(7))), 9))

    assert marker.read_text() == "x"
    egg_bindings.EGraph().parse_and_run_program(egraph.as_egglog_string)
    assert marker.read_text() == "xx"


@pytest.mark.parametrize("cost_first", [False, True], ids=["raw-table-first", "cost-first"])
def test_dynamic_cost_reuses_a_compatible_canonical_table(cost_first: bool) -> None:
    @function(egg_fn="canonical_cost_target")
    def target(x: i64Like) -> i64: ...

    @function(egg_fn="cost_table_canonical_cost_target")
    def raw_cost(x: i64Like) -> i64: ...

    egraph = EGraph()
    cost_action = set_cost(target(1), 7)
    raw_table_action = set_(raw_cost(2)).to(i64(5))
    cost_actions = (cost_action, raw_table_action) if cost_first else (raw_table_action, cost_action)
    egraph.register(
        set_(target(1)).to(i64(2)),
        *cost_actions,
    )

    assert egraph.lookup_function_value(raw_cost(1)) == i64(7)
    assert egraph.lookup_function_value(raw_cost(2)) == i64(5)
    assert egraph.has_custom_cost(target)


@pytest.mark.parametrize("cost_first", [False, True], ids=["raw-table-first", "cost-first"])
def test_dynamic_cost_rejects_a_merged_canonical_table(cost_first: bool) -> None:
    @function(egg_fn="merged_cost_target")
    def target(x: i64Like) -> i64: ...

    @function(egg_fn="cost_table_merged_cost_target", merge=lambda old, _new: old)
    def raw_cost(x: i64Like) -> i64: ...

    cost_action = set_cost(target(1), 7)
    raw_table_action = set_(raw_cost(2)).to(i64(5))
    actions = (cost_action, raw_table_action) if cost_first else (raw_table_action, cost_action)

    with pytest.raises(ValueError, match="already used by an incompatible callable"):
        EGraph().register(*actions)


def test_dynamic_cost_rejects_multiple_raw_canonical_table_aliases() -> None:
    @function(egg_fn="aliased_cost_target")
    def target(x: i64Like) -> i64: ...

    @function(egg_fn="cost_table_aliased_cost_target")
    def first_raw_cost(x: i64Like) -> i64: ...

    @function(egg_fn="cost_table_aliased_cost_target")
    def second_raw_cost(x: i64Like) -> i64: ...

    with pytest.raises(ValueError, match="already has a raw callable alias"):
        EGraph().register(
            set_cost(target(1), 7),
            set_(first_raw_cost(2)).to(i64(5)),
            set_(second_raw_cost(3)).to(i64(6)),
        )


@pytest.mark.parametrize("cost_first", [False, True], ids=["raw-table-first", "cost-first"])
def test_dynamic_cost_rejects_a_raw_table_alias_for_a_primitive(cost_first: bool) -> None:
    @function(egg_fn="cost_table_/")
    def raw_cost(left: i64Like, right: i64Like) -> i64: ...

    cost_action = set_cost(i64(4) / i64(2), 7)
    raw_table_action = set_(raw_cost(1, 0)).to(i64(5))
    actions = (cost_action, raw_table_action) if cost_first else (raw_table_action, cost_action)

    with pytest.raises(ValueError, match="eager or builtin primitive"):
        EGraph(*actions)


def test_freeze_preserves_a_reused_canonical_cost_table_as_raw_rows_and_costs() -> None:
    @function(egg_fn="freeze_cost_target")
    def target(x: i64Like) -> i64: ...

    @function(egg_fn="cost_table_freeze_cost_target")
    def raw_cost(x: i64Like) -> i64: ...

    egraph = EGraph(
        set_(raw_cost(2)).to(i64(5)),
        set_(target(1)).to(i64(2)),
        set_cost(target(1), 7),
    )

    rendered = str(egraph.freeze())
    assert "set_(raw_cost(2)).to(i64(5))" in rendered
    assert "set_(raw_cost(1)).to(i64(7))" in rendered
    assert "set_cost(target(2), 5)" not in rendered
    assert "set_cost(target(1), 7)" in rendered

    replayed = eval(rendered.removesuffix(".freeze()"), globals(), locals())
    assert isinstance(replayed, EGraph)
    assert replayed.lookup_function_value(raw_cost(1)) == i64(7)
    assert replayed.lookup_function_value(raw_cost(2)) == i64(5)
    assert replayed.lookup_function_value(target(1)) == i64(2)
    assert replayed.has_custom_cost(target)


def test_freeze_preserves_every_callable_alias_for_a_shared_cost_table() -> None:
    @function(egg_fn="+", builtin=True)
    def plus_alias(left: i64Like, right: i64Like) -> i64: ...

    egraph = EGraph(
        set_cost(i64(1) + i64(2), 5),
        set_cost(plus_alias(3, 4), 6),
    )

    rendered = str(egraph.freeze())
    replayed = eval(rendered.removesuffix(".freeze()"), globals(), locals())
    assert isinstance(replayed, EGraph)
    assert replayed.has_custom_cost(i64.__add__)
    assert replayed.has_custom_cost(plus_alias)
    assert replayed.lookup_function_value(get_cost(i64(1) + i64(2))) == i64(5)
    assert replayed.lookup_function_value(get_cost(plus_alias(3, 4))) == i64(6)


def test_freeze_preserves_reverse_argument_order_for_rows_and_costs() -> None:
    class ReverseFreeze(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def label(self, value: StringLike) -> i64: ...

    expression = ReverseFreeze(1).label("key")
    egraph = EGraph(set_(expression).to(i64(3)), set_cost(expression, 7))

    replayed = eval(str(egraph.freeze()).removesuffix(".freeze()"), globals(), locals())

    assert isinstance(replayed, EGraph)
    assert replayed.lookup_function_value(expression) == i64(3)
    assert replayed.lookup_function_value(get_cost(expression)) == i64(7)


def test_dynamic_cost_rejects_an_incompatible_overload_without_recording_it() -> None:
    egraph = EGraph()
    egraph.register(set_cost(i64(1) + i64(2), 5))

    with pytest.raises(ValueError, match="already serves a callable with an incompatible schema"):
        egraph.register(set_cost(Rational(1, 2) + Rational(1, 3), 6))

    assert egraph.has_custom_cost(i64.__add__)
    assert not egraph.has_custom_cost(Rational.__add__)


def test_dynamic_cost_rejects_a_negative_literal() -> None:
    class Costed(Expr):
        def __init__(self, value: i64Like) -> None: ...

    with pytest.raises(ValueError, match="must be nonnegative"):
        set_cost(Costed(1), -1)


class TestScheduler:
    def test_seq_schedule_decls_track_ruleset_updates(self):
        egraph = EGraph()

        rel = relation("rel_live", i64)
        live_rules = ruleset(name="live-rules")
        schedule = seq(live_rules, run()).saturate()
        _ = str(schedule)

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
        around the whole repeated schedule. Scoping only one repetition creates
        fresh scheduler state for each repeat.

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
        explicit_outer_values = _run_and_collect(bo.scope((run(grow, scheduler=bo) + shrink) * 3))
        explicit_inner_values = _run_and_collect(bo.scope(run(grow, scheduler=bo) + shrink) * 3)

        assert implicit_values == explicit_outer_values == {-3, -2, -1, 0, 1, 2}
        assert explicit_inner_values == {-3, -2, -1, 0, 1}

    def test_persistent_scheduler_reuses_state_across_runs(self):
        r = relation("R", i64)
        s = relation("S", i64)
        seed = relation("Seed")
        x = var("x", i64)

        copy = ruleset(rule(r(x)).then(s(x)), name="copy")
        grow = ruleset(rule(seed()).then(r(i64(3))), name="grow")

        def _run_and_collect() -> set[int]:
            egraph = EGraph()
            egraph.register(r(i64(0)), r(i64(1)), r(i64(2)), seed())
            scheduler = back_off(match_limit=2, ban_length=2).persistent()
            egraph.run(run(copy, scheduler=scheduler))
            egraph.push()
            egraph.pop()
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

        assert _run_and_collect() == {0, 1, 2}

    def test_saturate_waits_for_deferred_persistent_scheduler_work(self):
        source = relation("saturate_source", i64)
        copied = relation("saturate_copied", i64)
        x = var("x", i64)
        copy = ruleset(rule(source(x)).then(copied(x)), name="saturate-copy")

        egraph = EGraph()
        egraph.register(*(source(i64(i)) for i in range(3)))
        scheduler = back_off(match_limit=1, ban_length=1).persistent()

        egraph.saturate(run(copy, scheduler=scheduler), max=4, visualize=False)

        egraph.check(*(copied(i64(i)) for i in range(3)))

    def test_persistent_scheduler_is_saved_once_across_runs(self):
        r = relation("R_saved_scheduler", i64)
        s = relation("S_saved_scheduler", i64)
        x = var("x", i64)
        copy = ruleset(rule(r(x)).then(s(x)), name="copy-saved-scheduler")

        egraph = EGraph(save_egglog_string=True)
        egraph.register(r(i64(0)), r(i64(1)))
        scheduler = back_off(match_limit=2, ban_length=2).persistent()

        egraph.run(run(copy, scheduler=scheduler))
        egraph.run(run(copy, scheduler=scheduler))

        scheduler_lines = [line for line in egraph.as_egglog_string.splitlines() if line.startswith("(let-scheduler ")]
        run_with_lines = [line for line in egraph.as_egglog_string.splitlines() if "(run-with " in line]

        assert len(scheduler_lines) == 1
        assert len(run_with_lines) == 2

    def test_scheduler_scope_does_not_leak_to_sequence_sibling(self):
        r = ruleset(name="scheduler-lexical-scope")
        scheduler = back_off(match_limit=2, ban_length=2)
        egraph = EGraph(save_egglog_string=True)

        egraph.run(seq(scheduler.scope(run(r, scheduler=scheduler)), run(r, scheduler=scheduler)))

        run_schedule = next(line for line in egraph.as_egglog_string.splitlines() if line.startswith("(run-schedule "))
        assert run_schedule.count("(let-scheduler ") == 2
        ruleset_name = f"{__name__}.scheduler-lexical-scope"
        assert f"(run-with _scheduler_0 {ruleset_name})" in run_schedule
        assert f"(run-with _scheduler_1 {ruleset_name})" in run_schedule

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

        egraph.run(run(r, rel(i64(0)), scheduler=bo))


@function
def ff(x: i64Like, y: i64Like) -> E: ...


@function
def gg() -> E: ...


@pytest.mark.parametrize("extractor", ["tree", "greedy-dag"])
def test_extract_multiple_sequence_preserves_heterogeneous_roots(extractor: ExtractionMode) -> None:
    class MultiRoot(Expr):
        def __init__(self, value: i64Like) -> None: ...

        def __add__(self, other: MultiRoot) -> MultiRoot: ...

        @method(unextractable=True)
        def opaque(self) -> MultiRoot: ...

    egraph = EGraph()
    opaque = egraph.let("opaque_multi_root", MultiRoot(1).opaque())
    repeated = MultiRoot(0) + MultiRoot(0)
    egraph.register(
        union(MultiRoot(2)).with_(repeated),
        set_cost(MultiRoot(2), 100),
        set_cost(MultiRoot(0), 1),
    )

    extracted = egraph.extract_multiple([String("first"), opaque, MultiRoot(2)], 1, extractor=extractor)

    assert extracted == [[String("first")], [], [repeated]]
    assert egraph.extract_multiple([String("first"), opaque, MultiRoot(2)], 2, extractor=extractor) == [
        [String("first")],
        [],
        [repeated, MultiRoot(2)],
    ]
    assert egraph.extract_multiple(i64(4), 1, extractor=extractor) == [i64(4)]
    homogeneous: list[list[MultiRoot]] = egraph.extract_multiple([MultiRoot(2)], 1, extractor=extractor)
    assert homogeneous == [[repeated]]


def test_extract_multiple_validates_batch_arguments() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        EGraph().extract_multiple(i64(1), 0)
    with pytest.raises(ValueError, match="at least one expression"):
        EGraph().extract_multiple([], 1)
    with pytest.raises(ValueError, match="Unknown extractor"):
        EGraph().extract_multiple(i64(1), 1, extractor="unknown")  # type: ignore[call-overload]


@pytest.mark.parametrize("extractor", ["tree", "greedy-dag"])
def test_keep_best_compacts_and_allows_continued_iteration(extractor: ExtractionMode) -> None:
    class CompactExpr(Expr):
        def __init__(self, value: i64Like) -> None: ...

        def __add__(self, other: CompactExpr) -> CompactExpr: ...

    @function(merge=lambda old, new: new)
    def target(key: i64Like) -> CompactExpr: ...

    @function(merge=lambda old, new: new)
    def discarded(key: i64Like) -> CompactExpr: ...

    direct = CompactExpr(2)
    shared = CompactExpr(1)
    repeated = shared + shared
    egraph = EGraph(save_egglog_string=True)
    egraph.register(
        union(direct).with_(repeated),
        set_(target(0)).to(direct),
        set_(discarded(0)).to(CompactExpr(9)),
        set_cost(direct, 100),
        set_cost(CompactExpr(1), 4),
    )

    egraph.keep_best(target, extractor=extractor)

    assert egraph.function_size(target) == 1
    assert egraph.function_size(discarded) == 0
    assert egraph.has_custom_cost(CompactExpr)
    assert egraph.extract(target(0)) == repeated

    # Reusing an expression that was factored through a synthetic let must not
    # reference the row that keep-best cleared.
    egraph.register(repeated, set_(target(1)).to(repeated))
    assert egraph.function_size(target) == 2


def test_keep_best_rejects_non_table_callable_before_compaction() -> None:
    @function
    def eager(value: i64Like) -> i64:
        return cast("i64", value) + 1

    with pytest.raises(ValueError, match="table-backed"):
        EGraph().keep_best(eager)


def test_keep_best_invalidates_opaque_values_until_a_parent_scope_is_restored() -> None:
    class CompactKey(Expr):
        def __init__(self, value: i64Like) -> None: ...

    @function
    def score(key: CompactKey) -> i64: ...

    egraph = EGraph(CompactKey(1), set_(score(CompactKey(1))).to(i64(11)))
    parent_key = egraph.lookup_function_value(CompactKey(1))
    assert parent_key is not None

    egraph.push()
    egraph.keep_best(score)
    fresh_key = egraph.lookup_function_value(CompactKey(1))
    assert fresh_key is not None
    assert egraph.lookup_function_value(score(fresh_key)) == i64(11)
    with pytest.raises(ValueError, match="inactive push scope"):
        egraph.lookup_function_value(score(parent_key))

    egraph.pop()
    assert egraph.lookup_function_value(score(parent_key)) == i64(11)

    egraph.keep_best(score)
    with pytest.raises(ValueError, match="inactive push scope"):
        egraph.lookup_function_value(score(parent_key))
    root_fresh_key = egraph.lookup_function_value(CompactKey(1))
    assert root_fresh_key is not None
    assert egraph.lookup_function_value(score(root_fresh_key)) == i64(11)


class TestCustomExtract:
    def test_literal_root(self) -> None:
        def is_even_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            del egraph, children_costs
            return int(isinstance(expr, i64) and expr.value % 2 == 0)

        assert EGraph().extract(i64(10), include_cost=True, cost_model=is_even_cost_model) == (i64(10), 1)
        assert EGraph().extract(i64(5), include_cost=True, cost_model=is_even_cost_model) == (i64(5), 0)

    def test_lookup_value_root(self) -> None:
        class LookupExpr(Expr):
            def __init__(self, value: i64Like) -> None: ...

        egraph = EGraph(LookupExpr(1))
        value = egraph.lookup_function_value(LookupExpr(1))

        assert value is not None
        assert egraph.extract(value, include_cost=True, cost_model=default_cost_model) == (LookupExpr(1), 2)

    @staticmethod
    def _capture_container_children_costs(
        root_expr: BaseExpr,
        *,
        leaf_cost: Callable[[BaseExpr], int],
        should_capture: Callable[[BaseExpr, list[int]], bool],
    ) -> tuple[BaseExpr, BaseExpr, list[int]]:
        seen: dict[str, object] = {}

        def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            match expr:
                case i64() | String():
                    return leaf_cost(expr)
                case _:
                    if should_capture(expr, children_costs):
                        seen["expr"] = expr
                        seen["children_costs"] = children_costs.copy()
            return default_cost_model(egraph, expr, children_costs)

        extracted, _cost = EGraph().extract(root_expr, include_cost=True, cost_model=my_cost_model)

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
            pytest.param(Pair(i64(1), String("hi")), id="Pair"),
            pytest.param(Maybe[i64].some(i64(1)), id="Maybe some"),
            pytest.param(Maybe[i64].none(), id="Maybe none"),
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
        expr = Set(i64(2), i64(1), i64(2), i64(3))
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

    def test_errors_bubble(self):
        def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            msg = "bad"
            raise ValueError(msg)

        egraph = EGraph()

        with pytest.raises(ValueError, match="bad"):
            egraph.extract(i64(10), cost_model=my_cost_model)

    def test_dag_cost_model(self):
        model = DagCostModel(
            marginal_cost=lambda egraph, expr: default_cost_model(egraph, expr, []),
            identity=0,
        )
        egraph = EGraph()
        expr = ff(1, 2)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=model, extractor="greedy-dag")
        assert cost == 3
        assert expr == res

        expr = ff(1, 1)
        _, tree_cost = egraph.extract(expr, include_cost=True, cost_model=model)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=model, extractor="greedy-dag")
        assert tree_cost == 3
        assert cost == 2
        assert expr == res

        @function
        def bin(l: E, r: E) -> E: ...

        x = constant("x", E)
        y = constant("y", E)
        expr = bin(x, bin(x, y))
        egraph.register(expr)
        res, cost = egraph.extract(expr, include_cost=True, cost_model=model, extractor="greedy-dag")
        assert cost == 4
        assert expr == res

    def test_dag_cost_model_supports_nonzero_typed_identity(self) -> None:
        @dataclass(frozen=True, order=True)
        class OffsetCost:
            value: int

            def __add__(self, other: OffsetCost) -> OffsetCost:
                return OffsetCost(self.value + other.value - 10)

        model = DagCostModel(
            marginal_cost=lambda _egraph, _expr: OffsetCost(11),
            identity=OffsetCost(10),
        )
        egraph = EGraph()
        expr = ff(1, 1)

        assert egraph.extract(expr, include_cost=True, cost_model=model) == (expr, OffsetCost(13))
        assert egraph.extract(expr, include_cost=True, cost_model=model, extractor="greedy-dag") == (
            expr,
            OffsetCost(12),
        )

    @pytest.mark.parametrize(
        ("model_kind", "extractor", "expected_cost"),
        [
            pytest.param("tree", "tree", 7, id="tree-model"),
            pytest.param("dag", "tree", 7, id="dag-model-tree-extractor"),
            pytest.param("dag", "greedy-dag", 4, id="dag-model-greedy-dag-extractor"),
        ],
    )
    def test_default_cost_model_reads_dynamic_costs_during_callbacks(
        self,
        model_kind: str,
        extractor: ExtractionMode,
        expected_cost: int,
    ) -> None:
        class DynamicCostExpr(Expr):
            def __init__(self, value: i64Like) -> None: ...

            def __add__(self, other: DynamicCostExpr) -> DynamicCostExpr: ...

        egraph = EGraph()
        egraph.register(
            union(DynamicCostExpr(2)).with_(DynamicCostExpr(1) + DynamicCostExpr(1)),
            set_cost(DynamicCostExpr(2), 50),
            set_cost(DynamicCostExpr(1), 2),
        )

        if model_kind == "tree":
            result, cost = egraph.extract(
                DynamicCostExpr(2), include_cost=True, cost_model=default_cost_model, extractor=extractor
            )
        else:
            model = DagCostModel(
                marginal_cost=lambda callback_egraph, node: default_cost_model(callback_egraph, node, []),
                identity=0,
            )
            result, cost = egraph.extract(DynamicCostExpr(2), include_cost=True, cost_model=model, extractor=extractor)

        assert result == DynamicCostExpr(1) + DynamicCostExpr(1)
        assert cost == expected_cost

    def test_tree_cost_model_can_lookup_table_with_a_primitive_callback_argument(self) -> None:
        class LookupByPrimitive(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function
        def score(value: i64Like) -> i64: ...

        egraph = EGraph()
        egraph.register(set_(score(3)).to(i64(17)))

        def lookup_cost(callback_egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if isinstance(expr, LookupByPrimitive):
                args = get_callable_args(expr)
                assert args is not None
                value = callback_egraph.lookup_function_value(score(cast("i64", args[0])))
                assert value is not None
                return int(value) + sum(children_costs)
            return sum(children_costs)

        assert egraph.extract(
            LookupByPrimitive(3), include_cost=True, cost_model=cast("TreeCostModel[int]", lookup_cost)
        ) == (
            LookupByPrimitive(3),
            17,
        )

    def test_cost_model_callback_rejects_evaluating_new_lookup_arguments(self) -> None:
        class LookupByPrimitive(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function
        def score(value: i64Like) -> i64: ...

        egraph = EGraph()
        egraph.register(set_(score(99)).to(i64(17)))

        def lookup_cost(callback_egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if isinstance(expr, LookupByPrimitive):
                callback_egraph.lookup_function_value(score(99))
            return sum(children_costs)

        with pytest.raises(ValueError, match="only look up tables using values supplied to the callback"):
            egraph.extract(LookupByPrimitive(3), cost_model=cast("TreeCostModel[int]", lookup_cost))

        assert egraph.lookup_function_value(score(99)) == i64(17)

    def test_cost_model_callback_requires_lookup_table_to_be_registered(self) -> None:
        class LookupByPrimitive(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function
        def never_registered(value: i64Like) -> i64: ...

        def lookup_cost(callback_egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if isinstance(expr, LookupByPrimitive):
                args = get_callable_args(expr)
                assert args is not None
                callback_egraph.lookup_function_value(never_registered(cast("i64", args[0])))
            return sum(children_costs)

        with pytest.raises(ValueError, match="must be registered before extraction starts"):
            EGraph().extract(LookupByPrimitive(3), cost_model=cast("TreeCostModel[int]", lookup_cost))

    def test_cost_model_callback_rejects_value_from_another_egraph(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        class Root(Expr):
            def __init__(self) -> None: ...

        @function
        def score(value: Key) -> i64: ...

        foreign_egraph = EGraph(Key(3))
        foreign_key = foreign_egraph.lookup_function_value(Key(3))
        assert foreign_key is not None

        egraph = EGraph()
        egraph.register(set_(score(Key(3))).to(i64(17)))

        def lookup_cost(callback_egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if isinstance(expr, Root):
                callback_egraph.lookup_function_value(score(foreign_key))
            return sum(children_costs)

        with pytest.raises(ValueError, match="only look up tables using values supplied to the callback"):
            egraph.extract(Root(), cost_model=cast("TreeCostModel[int]", lookup_cost))

    def test_lookup_rejects_value_from_another_egraph(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function
        def score(value: Key) -> i64: ...

        foreign_egraph = EGraph(Key(3))
        foreign_key = foreign_egraph.lookup_function_value(Key(3))
        assert foreign_key is not None

        egraph = EGraph()
        egraph.register(set_(score(Key(3))).to(i64(17)))

        with pytest.raises(ValueError, match="belongs to a different EGraph"):
            egraph.lookup_function_value(score(foreign_key))

    def test_lookup_values_follow_push_scope_lifetimes(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function
        def score(value: Key) -> i64: ...

        egraph = EGraph(Key(1), set_(score(Key(1))).to(i64(11)))
        parent_key = egraph.lookup_function_value(Key(1))
        assert parent_key is not None

        egraph.push()
        assert egraph.lookup_function_value(score(parent_key)) == i64(11)
        egraph.register(Key(2), set_(score(Key(2))).to(i64(22)))
        child_key = egraph.lookup_function_value(Key(2))
        assert child_key is not None
        egraph.pop()

        assert egraph.lookup_function_value(score(parent_key)) == i64(11)
        egraph.register(Key(3), set_(score(Key(3))).to(i64(33)))
        with pytest.raises(ValueError, match="inactive push scope"):
            egraph.lookup_function_value(score(child_key))

    def test_cost_model_callback_can_lookup_nested_value_from_container_argument(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        class Root(Expr):
            def __init__(self, values: Map[Key, i64]) -> None: ...

        @function
        def score(value: Key) -> i64: ...

        egraph = EGraph()
        egraph.register(set_(score(Key(3))).to(i64(17)))

        def lookup_cost(callback_egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if not isinstance(expr, Root):
                return sum(children_costs)
            args = get_callable_args(expr)
            assert args is not None
            (key,) = cast("Map[Key, i64]", args[0]).value
            value = callback_egraph.lookup_function_value(score(key))
            assert value is not None
            return int(value) + sum(children_costs)

        expr = Root(Map[Key, i64].empty().insert(Key(3), i64(0)))
        assert egraph.extract(expr, include_cost=True, cost_model=cast("TreeCostModel[int]", lookup_cost)) == (
            expr,
            17,
        )

    def test_cost_model_callback_values_are_scoped_to_their_egraph(self) -> None:
        class LookupByString(Expr):
            def __init__(self, value: StringLike) -> None: ...

        @function
        def score(value: StringLike) -> i64: ...

        lookup_egraph = EGraph()
        lookup_egraph.register(
            set_(score("padding")).to(i64(1)),
            set_(score("needle")).to(i64(17)),
        )

        def lookup_cost(_callback_egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if isinstance(expr, LookupByString):
                args = get_callable_args(expr)
                assert args is not None
                value = lookup_egraph.lookup_function_value(score(cast("String", args[0])))
                assert value is not None
                return int(value) + sum(children_costs)
            return sum(children_costs)

        source_egraph = EGraph()
        assert source_egraph.extract(
            LookupByString("needle"), include_cost=True, cost_model=cast("TreeCostModel[int]", lookup_cost)
        ) == (LookupByString("needle"), 17)

    def test_nested_cost_model_callback_preserves_outer_egraph_values(self) -> None:
        class Outer(Expr):
            def __init__(self, value: i64Like) -> None: ...

        class Inner(Expr):
            def __init__(self) -> None: ...

        @function
        def score(value: i64Like) -> i64: ...

        outer_egraph = EGraph()
        outer_egraph.register(set_(score(3)).to(i64(17)))
        inner_egraph = EGraph()

        def outer_cost(callback_egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if not isinstance(expr, Outer):
                return sum(children_costs)
            args = get_callable_args(expr)
            assert args is not None
            outer_value = cast("i64", args[0])

            def inner_cost(_inner_egraph: EGraph, inner_expr: BaseExpr, inner_children: list[int]) -> int:
                if not isinstance(inner_expr, Inner):
                    return sum(inner_children)
                value = callback_egraph.lookup_function_value(score(outer_value))
                assert value is not None
                return int(value)

            _, cost = inner_egraph.extract(
                Inner(), include_cost=True, cost_model=cast("TreeCostModel[int]", inner_cost)
            )
            return cost + sum(children_costs)

        assert outer_egraph.extract(Outer(3), include_cost=True, cost_model=cast("TreeCostModel[int]", outer_cost)) == (
            Outer(3),
            17,
        )

    @pytest.mark.parametrize(
        ("model_kind", "extractor"),
        [
            pytest.param("tree", "tree", id="tree-model"),
            pytest.param("dag", "greedy-dag", id="dag-model"),
        ],
    )
    def test_cost_model_callbacks_preserve_reverse_argument_order(
        self, model_kind: str, extractor: ExtractionMode
    ) -> None:
        class ReverseCostResult(Expr): ...

        class ReverseCostSource(Expr):
            def __init__(self, value: i64Like) -> None: ...

            @method(reverse_args=True)
            def make(self, label: StringLike) -> ReverseCostResult: ...

        source = ReverseCostSource(3)
        expr = source.make("label")
        egraph = EGraph()
        egraph.register(set_cost(expr, 9))
        seen_args: list[BaseExpr] = []

        def marginal_cost(callback_egraph: EGraph, node: BaseExpr) -> int:
            if get_callable_fn(node) != ReverseCostSource.make:
                return 0
            args = get_callable_args(node)
            assert args is not None
            seen_args.extend(args)
            return default_cost_model(callback_egraph, node, [])

        if model_kind == "tree":

            def tree_cost(callback_egraph: EGraph, node: BaseExpr, children_costs: list[int]) -> int:
                return marginal_cost(callback_egraph, node) + sum(children_costs)

            result, cost = egraph.extract(
                expr,
                include_cost=True,
                cost_model=cast("TreeCostModel[int]", tree_cost),
                extractor=extractor,
            )
        else:
            result, cost = egraph.extract(
                expr,
                include_cost=True,
                cost_model=DagCostModel(marginal_cost, 0),
                extractor=extractor,
            )

        assert result == expr
        assert cost == 9
        assert seen_args
        assert len(seen_args) % 2 == 0
        for source_arg, label_arg in zip(seen_args[::2], seen_args[1::2], strict=True):
            assert isinstance(source_arg, ReverseCostSource)
            assert isinstance(label_arg, String)
            assert label_arg.value == "label"

    def test_tree_cost_model_rejected_by_greedy_dag(self):
        with pytest.raises(TypeError, match="requires a DagCostModel"):
            EGraph().extract(i64(1), cost_model=default_cost_model, extractor="greedy-dag")

    def test_dag_marginal_error_bubbles(self):
        def marginal_cost(egraph: EGraph, expr: BaseExpr) -> int:
            del egraph, expr
            msg = "marginal failed"
            raise LookupError(msg)

        with pytest.raises(LookupError, match="marginal failed"):
            EGraph().extract(i64(1), cost_model=DagCostModel(marginal_cost, 0))

    def test_dag_add_error_bubbles(self):
        @dataclass(frozen=True)
        class AddErrorCost:
            value: int

            def __add__(self, other: AddErrorCost) -> AddErrorCost:
                del other
                msg = "addition failed"
                raise ArithmeticError(msg)

            def __lt__(self, other: AddErrorCost) -> bool:
                return self.value < other.value

            def __le__(self, other: AddErrorCost) -> bool:
                return self.value <= other.value

            def __gt__(self, other: AddErrorCost) -> bool:
                return self.value > other.value

            def __ge__(self, other: AddErrorCost) -> bool:
                return self.value >= other.value

        model = DagCostModel(lambda egraph, expr: AddErrorCost(1), AddErrorCost(0))
        with pytest.raises(ArithmeticError, match="addition failed"):
            EGraph().extract(ff(1, 2), cost_model=model)

    def test_dag_comparison_error_bubbles(self):
        @dataclass(frozen=True)
        class CompareErrorCost:
            value: int

            def __add__(self, other: CompareErrorCost) -> CompareErrorCost:
                return CompareErrorCost(self.value + other.value)

            def __eq__(self, other: object) -> bool:
                return False

            def __lt__(self, other: CompareErrorCost) -> bool:
                del other
                msg = "comparison failed"
                raise RuntimeError(msg)

            def __le__(self, other: CompareErrorCost) -> bool:
                del other
                msg = "comparison failed"
                raise RuntimeError(msg)

            def __gt__(self, other: CompareErrorCost) -> bool:
                del other
                msg = "comparison failed"
                raise RuntimeError(msg)

            def __ge__(self, other: CompareErrorCost) -> bool:
                del other
                msg = "comparison failed"
                raise RuntimeError(msg)

        model = DagCostModel(lambda egraph, expr: CompareErrorCost(1), CompareErrorCost(0))
        egraph = EGraph()
        egraph.register(union(ff(1, 2)).with_(gg()))
        with pytest.raises(RuntimeError, match="comparison failed"):
            egraph.extract(ff(1, 2), cost_model=model, extractor="greedy-dag")


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
    assert isinstance(cast("object", A.m), RuntimeFunction)
    assert eq(A.__eq__(A(), A())).to(A() == A())
    assert isinstance(cast("object", A.__eq__), RuntimeFunction)
    assert eq(A.__add__(A(), A())).to(A() + A())
    assert isinstance(cast("object", A.__add__), RuntimeFunction)
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
            assert args is not None
            values = [eg.lookup_function_value(cast("KAT", arg).size()) for arg in args]
            assert all(value is not None for value in values)
            return sum(int(value) for value in values if value is not None)

        return 2

    assert eg.extract(q0, include_cost=True, cost_model=conv_cost) == (KAT.eps().func(KAT.emp()), 1)
