# mypy: disable-error-code="empty-body"
from __future__ import annotations

import importlib
import math
import os
import pathlib
import runpy
import subprocess
import sys
from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import ClassVar, Literal, TypeAlias, TypeVar, Union, cast, get_args, get_origin
from unittest.mock import MagicMock

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

import egglog.bindings as egg_bindings
import egglog.builtins as egg_builtins
from egglog import *
from egglog._catalog import builtin_catalog
from egglog._presentation import CallKind, presentation
from egglog._program import Owner, Ref
from egglog.conversion import ConvertError
from egglog.egraph_state import _decode_run_frames
from egglog.runtime import RuntimeClass, RuntimeExpr, RuntimeFunction, resolve_callable
from egglog.type_constraint_solver import TypeConstraintError

_BuiltinExprT = TypeVar("_BuiltinExprT", bound=BaseExpr)


def _replay_recorded_source(text: str) -> list[pb.RunProgramOutput]:
    """Replay sequential Source phases exclusively through encoded engine operations."""
    source = egg_bindings._ProtoEngine.source(text)
    graph = EGraph()
    responses = []
    try:
        assert graph._state.handle is not None
        while (submission := source.next_submission()) is not None:
            kind, program, count = submission
            if kind == "program":
                assert type(program) is bytes
                encoded, host_error = graph._state.transport.run(
                    pb.RunProgramRequest(
                        egraph_id=graph._state.handle, program=pb.Program.from_binary(program)
                    ).to_binary()
                )
                if host_error is not None:
                    raise host_error
                for frame in _decode_run_frames(encoded):
                    assert frame.result is not None
                    if frame.result.field == "error":
                        error = EggSmolError(frame.result.value.message)
                        error.__dict__["_egglog_proto_error"] = frame.result.value
                        error.__dict__["_egglog_proto_frame"] = frame
                        raise error
                    if frame.result.value.outputs:
                        responses.append(frame.result.value)
            elif kind in {"push", "pop"}:
                assert program is None
                for _ in range(count):
                    {"push": graph.push, "pop": graph.pop}[kind]()
            else:
                raise AssertionError(f"Unknown Source submission: {kind}")
    finally:
        for state in [graph._state, *graph._state_stack]:
            state.destroy()
    return responses


def _restore_snapshot(snapshot: pb.EGraphSnapshot, *, environment: tuple[Owner, ...]) -> EGraph:
    """Restore the actual serialized logical program through the encoded adapter."""
    from egglog._program import Builder, pack  # noqa: PLC0415

    snapshot = pb.EGraphSnapshot.from_binary(snapshot.to_binary())
    assert snapshot.program is not None
    owner = Builder.from_program(snapshot.program, environment=environment).publish()
    restored = EGraph()
    restored._state.run_program(
        pack(
            roots=[
                owner.ref(role, index)
                for role in ("sorts", "nodes", "declarations")
                for index, record in enumerate(getattr(snapshot.program, role))
                if role == "declarations" or (record.kind is not None and record.kind.field in {"eq", "binding"})
            ],
            commands=[owner.ref("commands", index) for index in range(len(snapshot.program.commands))],
        )
    )
    return restored


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
    for removed in ("set_num_threads", "set_no_decomp", "set_report_level"):
        assert not hasattr(egraph, removed)


def test_no_decomp_uses_creation_settings_across_clone_return() -> None:
    egraph = EGraph(no_decomp=True)
    egraph.push()
    assert egraph.no_decomp()
    egraph.pop()
    assert egraph.no_decomp()


def test_reading_creation_settings_preserves_scope_return() -> None:
    egraph = EGraph()
    saved = egraph._state
    egraph.push()
    assert not saved.no_decomp
    assert egraph.num_threads() == saved.threads
    egraph.pop()
    assert egraph._state is saved


def test_independent_egraphs_run_concurrently_after_setup() -> None:
    # A fresh process verifies that the public setup below resolves lazy declarations.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from egglog import EGraph, function, i64, relation, rule, ruleset


def make_identity():
    local_i64 = i64

    @function
    def identity(value: "local_i64") -> "local_i64":
        return value

    return identity


identity = make_identity()
seen = relation("concurrent_seen", i64)

@ruleset
def advance(value: i64):
    yield rule(seen(value)).then(seen(value + 1))

# Definitions alone leave lazy declarations and rule generators unresolved.
setup_graph = EGraph()
assert setup_graph.extract(identity(i64(0)) + 1).value == 1
setup_graph.run(0, ruleset=advance)

workers = 32
barrier = Barrier(workers, timeout=30)

def extract(worker):
    barrier.wait()
    for iteration in range(4):
        number = worker * 100 + iteration
        value = identity(i64(number)) + 1
        egraph = EGraph(seen(number))
        egraph.run(2, ruleset=advance)
        egraph.check(seen(number + 2))
        assert egraph.extract(value).value == number + 1

with ThreadPoolExecutor(max_workers=workers) as executor:
    list(executor.map(extract, range(workers)))
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_zero_threads_uses_available_parallelism() -> None:
    assert EGraph(num_threads=0).num_threads() >= 1


def test_egraph_no_decomp_is_creation_policy_not_saved_rule_metadata() -> None:
    rel = relation("no_decomp_creation", i64)
    x = var("x", i64)
    egraph = EGraph(save_egglog_string=True, no_decomp=True)
    assert egraph.no_decomp()
    egraph.register(rel(i64(0)), rule(rel(x)).then(rel(x + 1)))
    egraph.run(run())
    egraph.check(rel(i64(1)))
    transcript = egraph.as_egglog_string
    assert ":no-decomp" not in transcript
    _replay_recorded_source(transcript)


def test_global_naive_source_error_is_deferred_until_export() -> None:
    graph = EGraph(save_egglog_string=True, seminaive=False)
    graph.register(i64(1))
    assert graph.history().commands
    with pytest.raises(NotImplementedError, match="initial seminaive=False"):
        _ = graph.as_egglog_string


@pytest.mark.parametrize("bidirectional", [False, True], ids=["rewrite", "birewrite"])
def test_global_rule_options_execute_without_fake_saved_rewrite_options(*, bidirectional: bool) -> None:

    class RewriteConfigExpr(Expr):
        def __init__(self, value: i64Like) -> None: ...

    @function
    def rewrite_config_lookup(value: i64) -> i64: ...

    @function
    def rewrite_config_read(value: i64) -> i64:
        return rewrite_config_lookup(value)

    grounded = relation("rewrite_config_grounded", i64)
    x = var("__expr_0", i64)
    rewrite_builder = birewrite if bidirectional else rewrite
    registered = rewrite_builder(RewriteConfigExpr(x)).to(RewriteConfigExpr(rewrite_config_read(x)), grounded(x))
    egraph = EGraph(
        set_(rewrite_config_lookup(i64(1))).to(i64(2)),
        grounded(i64(1)),
        RewriteConfigExpr(i64(1)),
        seminaive=False,
        no_decomp=True,
    )

    egraph.register(registered)
    report = egraph.run(1)
    egraph.check(eq(RewriteConfigExpr(i64(1))).to(RewriteConfigExpr(i64(2))))

    assert egraph.no_decomp()
    assert report.updated
    recorded = EGraph(save_egglog_string=True, seminaive=False, no_decomp=True)
    with pytest.raises(NotImplementedError, match="initial seminaive=False"):
        _ = recorded.as_egglog_string


@pytest.mark.parametrize("egg_name", ["check", "true", "_", "@internal"])
def test_saved_transcript_preserves_parser_sensitive_explicit_callable_name(egg_name: str) -> None:
    rel = relation(f"python_name_{egg_name}", egg_fn=egg_name)
    authored = rel()
    assert authored.__egg_ref__.read().kind.value.func == egg_name
    egraph = EGraph(save_egglog_string=True)

    egraph.register(authored)
    egraph.check(authored)
    _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_empty_explicit_backend_names_are_treated_as_unspecified(*, save_egglog_string: bool) -> None:
    class EmptyNamed(Expr, egg_sort=""):
        def __init__(self) -> None: ...

    marker = relation("empty_explicit_backend_name", EmptyNamed, egg_fn="")
    egraph = EGraph(marker(EmptyNamed()), save_egglog_string=save_egglog_string)

    egraph.check(marker(EmptyNamed()))
    if save_egglog_string:
        _replay_recorded_source(egraph.as_egglog_string)


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
    assert egraph.extract(String(value)).value == value
    responses = _replay_recorded_source(egraph.as_egglog_string)
    replayed = responses[-1]
    extraction_kind = replayed.outputs[0].kind
    assert extraction_kind is not None
    assert extraction_kind.field == "extraction"
    term = extraction_kind.value.roots[0].variants[0].term
    literal_kind = replayed.nodes[term].kind
    assert literal_kind is not None
    assert literal_kind.field == "primitive_value"
    string_literal = literal_kind.value.value
    assert string_literal is not None
    assert string_literal.field == "string"
    assert string_literal.value == value

    with pytest.raises(EggSmolError):
        egraph.check(marker(String(normalized)))

    source = egraph.as_egglog_string
    with pytest.raises(EggSmolError):
        _replay_recorded_source(source)


@pytest.mark.parametrize("message", ['bad "quote"', "bad \\ slash"])
def test_saved_transcript_escapes_panic_message(message: str) -> None:
    trigger = relation("escaped_panic_message_trigger")
    panic_rules = ruleset(rule(trigger()).then(panic(message)), name="escaped-panic-message")
    egraph = EGraph(trigger(), save_egglog_string=True)
    before_failure = egraph.as_egglog_string

    with pytest.raises(EggSmolError) as exc_info:
        egraph.run(panic_rules)

    assert message in str(exc_info.value)
    with pytest.raises(EggSmolError) as replayed:
        _replay_recorded_source(egraph.as_egglog_string)
    assert message in str(replayed.value)
    _replay_recorded_source(before_failure)


def test_saved_transcript_allows_reserved_callable_name_in_nested_expression() -> None:
    @function(egg_fn="check")
    def nested_check(value: i64Like) -> i64: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(set_(nested_check(1)).to(i64(7)))

    assert egraph.lookup_function_value(nested_check(1)) == i64(7)
    egraph.check(eq(nested_check(1)).to(i64(7)))
    _replay_recorded_source(egraph.as_egglog_string)


def test_saved_transcript_allows_command_head_name_in_rule_action() -> None:
    source = relation("rule_action_command_head_source")
    target = relation("rule_action_command_head_target", egg_fn="check")
    action_rules = ruleset(rule(source()).then(target()), name="rule-action-command-head")
    egraph = EGraph(source(), save_egglog_string=True)

    egraph.run(action_rules)

    egraph.check(target())
    _replay_recorded_source(egraph.as_egglog_string)


def test_saved_transcript_preserves_equality_head_name_in_rule_fact() -> None:
    source = relation("equality_head_source", i64, i64)
    claimed = relation("equality_head_claimed", i64, i64, egg_fn="=")  # type: ignore[call-overload]
    target = relation("equality_head_target")
    x, y = vars_("x y", i64)
    equality_head_rules = ruleset(rule(source(x, y), claimed(x, y)).then(target()), name="equality-head")

    direct = EGraph(source(i64(1), i64(2)))
    direct.run(equality_head_rules)
    assert not direct.check_bool(target())

    recorded = EGraph(source(i64(1), i64(2)), save_egglog_string=True)
    recorded.run(equality_head_rules)
    assert recorded.function_size(target) == 0
    recorded.register(claimed(i64(1), i64(2)))
    recorded.run(equality_head_rules)
    recorded.check(target())
    assert recorded.function_size(claimed) == 1
    snapshot = pb.EGraphSnapshot.from_binary(recorded.freeze().to_binary())
    assert snapshot.program is not None
    declaration = next(
        declaration
        for declaration in snapshot.program.declarations
        if declaration.kind is not None and declaration.kind.field == "relation" and declaration.kind.value.name == "="
    )
    assert declaration.kind is not None
    assert declaration.kind.value.name == resolve_callable(claimed).read().kind.value.name
    assert presentation(resolve_callable(claimed)).views[0].path == (*__name__.split("."), "equality_head_claimed")
    _replay_recorded_source(recorded.as_egglog_string)


def test_saved_transcript_allows_equality_head_name_as_standalone_action() -> None:
    claimed = relation("standalone_equality_head", i64, i64, egg_fn="=")  # type: ignore[call-overload]
    egraph = EGraph(save_egglog_string=True)

    egraph.register(claimed(i64(1), i64(2)))

    assert egraph.function_size(claimed) == 1
    _replay_recorded_source(egraph.as_egglog_string)


def test_saved_transcript_preserves_literal_explicit_sort_name() -> None:
    class LiteralNamed(Expr, egg_sort="true"):
        def __init__(self) -> None: ...

    value = LiteralNamed()
    mapping = Map[LiteralNamed, i64].empty().insert(value, i64(7))
    egraph = EGraph(value, save_egglog_string=True)
    egraph.check(eq(mapping[value]).to(i64(7)), eq(mapping.length()).to(i64(1)))
    snapshot = pb.EGraphSnapshot.from_binary(egraph.freeze().to_binary())
    assert snapshot.program is not None
    sort = next(sort for sort in snapshot.program.sorts if sort.kind == Oneof("eq", "true"))
    assert sort.kind == Oneof("eq", "true")
    sort_definition = cast("RuntimeClass", LiteralNamed).__egg_definition__
    assert isinstance(sort_definition, Ref)
    view = presentation(sort_definition).type_view
    assert view is not None
    assert view.path == (*__name__.split("."), "LiteralNamed")
    _replay_recorded_source(egraph.as_egglog_string)


def test_saved_transcript_preserves_unparseable_explicit_sort_name() -> None:
    class UnparseableNamed(Expr, egg_sort="has space"):
        def __init__(self) -> None: ...

    value = UnparseableNamed()
    mapping = Map[UnparseableNamed, i64].empty().insert(value, i64(7))
    egraph = EGraph(value, save_egglog_string=True)
    egraph.check(eq(mapping[value]).to(i64(7)), eq(mapping.length()).to(i64(1)))
    snapshot = pb.EGraphSnapshot.from_binary(egraph.freeze().to_binary())
    assert snapshot.program is not None
    sort = next(sort for sort in snapshot.program.sorts if sort.kind == Oneof("eq", "has space"))
    assert sort.kind == Oneof("eq", "has space")
    sort_definition = cast("RuntimeClass", UnparseableNamed).__egg_definition__
    assert isinstance(sort_definition, Ref)
    view = presentation(sort_definition).type_view
    assert view is not None
    assert view.path == (*__name__.split("."), "UnparseableNamed")
    _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("egg_name", ["_", "@internal"])
def test_saved_transcript_preserves_context_sensitive_explicit_sort_name(egg_name: str) -> None:
    class ContextSensitiveNamed(Expr, egg_sort=egg_name):
        def __init__(self) -> None: ...

    value = ContextSensitiveNamed()
    mapping = Map[ContextSensitiveNamed, i64].empty().insert(value, i64(7))
    egraph = EGraph(value, save_egglog_string=True)
    egraph.check(eq(mapping[value]).to(i64(7)), eq(mapping.length()).to(i64(1)))
    snapshot = pb.EGraphSnapshot.from_binary(egraph.freeze().to_binary())
    assert snapshot.program is not None
    sort = next(sort for sort in snapshot.program.sorts if sort.kind == Oneof("eq", egg_name))
    assert sort.kind == Oneof("eq", egg_name)
    sort_definition = cast("RuntimeClass", ContextSensitiveNamed).__egg_definition__
    assert isinstance(sort_definition, Ref)
    view = presentation(sort_definition).type_view
    assert view is not None
    assert view.path == (*__name__.split("."), "ContextSensitiveNamed")
    _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("has space", id="space"),
        pytest.param("x)", id="parenthesis"),
        pytest.param("x;y", id="comment"),
        pytest.param("x\ny", id="newline"),
    ],
)
def test_saved_transcript_preserves_unparseable_let_name(name: str) -> None:
    egraph = EGraph(save_egglog_string=True)
    safe_name = "".join(character if character.isalnum() else "_" for character in name)
    first = egraph.let(name, i64(1))
    second = egraph.let(safe_name, i64(2))
    egraph.check(eq(first).to(i64(1)), eq(second).to(i64(2)))
    snapshot = pb.EGraphSnapshot.from_binary(egraph.freeze().to_binary())
    assert snapshot.program is not None
    assert get_let_name(first) == name
    assert get_let_name(second) == safe_name
    bindings = {
        node.kind.value: index
        for index, node in enumerate(snapshot.program.nodes)
        if node.kind is not None and node.kind.field == "binding"
    }
    assert bindings.keys() == {name, safe_name}
    assert bindings[name] != bindings[safe_name]
    _replay_recorded_source(egraph.as_egglog_string)


def test_direct_egraph_still_accepts_non_source_let_name() -> None:
    egraph = EGraph()

    value = egraph.let("has space", i64(1))

    egraph.check(eq(value).to(i64(1)))


@pytest.mark.parametrize("name", ["has space", "x;y", "x\ny"])
def test_saved_transcript_preserves_unparseable_ruleset_name(name: str) -> None:
    source = relation("named_ruleset_source", i64)
    destination = relation("named_ruleset_destination", i64)
    x = var("x", i64)
    safe_name = "".join(character if character.isalnum() else "_" for character in name)
    named_ruleset = ruleset(rule(source(x)).then(destination(x)), name=name)
    other_ruleset = ruleset(rule(source(x)).then(destination(x + 1)), name=safe_name)

    direct = EGraph(source(i64(7)))
    direct.run(named_ruleset)
    direct.check(destination(i64(7)))

    recorded = EGraph(source(i64(7)), save_egglog_string=True)
    recorded.run(named_ruleset)
    recorded.check(destination(i64(7)))
    assert recorded.function_size(destination) == 1
    recorded.run(other_ruleset)
    recorded.check(destination(i64(7)), destination(i64(8)))
    assert recorded.function_size(destination) == 2
    snapshot = pb.EGraphSnapshot.from_binary(recorded.freeze().to_binary())
    assert snapshot.program is not None
    names = [binding.name for binding in snapshot.program.rulesets]
    assert len(names) >= 2
    assert len(set(names)) == len(names)
    _replay_recorded_source(recorded.as_egglog_string)


def test_saved_transcript_rejects_literal_explicit_variable_name() -> None:
    source = relation("literal_variable_source", i64)
    destination = relation("literal_variable_destination", i64)
    x = var("x", i64, egg_name="0")

    direct = EGraph(source(i64(7)))
    direct.register(rule(source(x)).then(destination(x)))
    direct.run(1)
    direct.check(destination(i64(7)))

    recorded = EGraph(source(i64(7)), save_egglog_string=True)
    recorded.register(rule(source(x)).then(destination(x)))
    recorded.run(1)
    recorded.check(destination(i64(7)))
    before = recorded.freeze()
    with pytest.raises(ValueError, match=r'^identifier "0" cannot be rendered as one native source atom$'):
        _ = recorded.as_egglog_string
    assert recorded.freeze() == before


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
    recorded.register(rule(source(x)).then(destination(x)))
    recorded.run(1)
    recorded.check(destination(i64(7)))
    before = recorded.freeze()
    with pytest.raises(ValueError, match="cannot be rendered as one native source atom") as caught:
        _ = recorded.as_egglog_string
    assert str(caught.value) == f'identifier "{get_var_name(x)}" cannot be rendered as one native source atom'
    assert recorded.freeze() == before


@pytest.mark.parametrize("shared", [False, True])
def test_saved_transcript_rejects_variable_wildcard_semantic_drift(*, shared: bool) -> None:
    source = relation("wildcard_variable_source", i64, i64)
    destination = relation("wildcard_variable_destination")
    x = var("x", i64, egg_name="_")
    other = x if shared else var("other", i64, egg_name="_")
    repeated_variable_rule = rule(source(x, other)).then(destination())

    direct = EGraph(source(i64(1), i64(2)))
    direct.register(repeated_variable_rule)
    direct.run(1)
    assert not direct.check_bool(destination())
    direct.register(source(i64(2), i64(2)))
    direct.run(1)
    direct.check(destination())

    recorded = EGraph(source(i64(1), i64(2)), save_egglog_string=True)
    recorded.register(repeated_variable_rule)
    recorded.run(1)
    assert not recorded.check_bool(destination())
    recorded.register(source(i64(2), i64(2)))
    recorded.run(1)
    recorded.check(destination())
    before = recorded.freeze()
    with pytest.raises(ValueError, match=r'^identifier "_" cannot be rendered as one native source atom$'):
        _ = recorded.as_egglog_string
    assert recorded.freeze() == before


@pytest.mark.parametrize(("alias", "joined"), [("x", True), ("_x", False)])
def test_variable_aliases_select_canonical_query_identity(alias: str, *, joined: bool) -> None:
    source = relation("canonical_variable_pair", i64, i64)
    destination = relation("canonical_variable_match", i64, i64)
    left, right = var("x", i64), var("shown", i64, egg_name=alias)
    assert left.__egg_ref__ != right.__egg_ref__
    selected = ruleset(rule(source(left, right)).then(destination(left, right)))
    graph = EGraph(source(i64(1), i64(2)), save_egglog_string=True)
    graph.run(selected)
    # Unlike the historical implicit '_x' backend prefix, aliases now select
    # the exact canonical name: alias 'x' joins, alias '_x' stays distinct.
    assert graph.function_size(destination) == (0 if joined else 1)
    graph.register(source(i64(3), i64(3)))
    graph.run(selected)
    graph.check(destination(i64(3), i64(3)))
    assert graph.function_size(destination) == (1 if joined else 2)
    _replay_recorded_source(graph.as_egglog_string)


def test_variable_aliases_avoid_capture_and_keep_query_binders_independent() -> None:
    class AliasLeaf(Expr):
        def __init__(self, value: i64) -> None: ...

    class AliasPair(Expr):
        def __init__(self, left: AliasLeaf, right: AliasLeaf) -> None: ...

    source = relation("alias_capture_source", i64)
    other_source = relation("alias_other_source", i64)
    other_destination = relation("alias_other_destination", i64)
    value = var("shown", i64, egg_name="__expr_0")
    independent = var("__expr_0", i64)
    shared = AliasLeaf(value)
    selected = ruleset(
        rule(source(value)).then(AliasPair(shared, shared)),
        rule(other_source(independent)).then(other_destination(independent)),
    )
    graph = EGraph(source(i64(7)), other_source(i64(9)), save_egglog_string=True)
    graph.run(selected)
    graph.check(AliasPair(AliasLeaf(i64(7)), AliasLeaf(i64(7))), other_destination(i64(9)))
    assert graph.function_size(AliasLeaf) == 1
    assert graph.function_size(AliasPair) == 1
    _replay_recorded_source(graph.as_egglog_string)


@pytest.mark.parametrize("saved", [False, True])
def test_canonical_variable_name_rejects_conflicting_query_sorts(*, saved: bool) -> None:
    number, text = relation("alias_number", i64), relation("alias_text", String)
    left, right = var("x", i64), var("shown", String, egg_name="x")
    graph = EGraph(number(i64(1)), text(String("one")), save_egglog_string=saved)
    error = EggSmolError
    with pytest.raises(error, match="unbound or inconsistently typed variable x"):
        graph.register(rule(number(left), text(right)).then())


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

    observed = egraph.lookup_function_value(A(1))
    assert observed is not None
    value = egraph.let("lookup-value", A(1))
    score = egraph.lookup_function_value(value.score())
    assert score is not None
    assert int(score) == 7
    assert egraph.lookup_function_value(A(2)) is None
    assert egraph.function_size(A) == 1
    with pytest.raises(ValueError, match="observation scope"):
        egraph.lookup_function_value(observed.score())


def test_let_uses_global_binding_without_public_function(capfd: pytest.CaptureFixture[str]):
    egraph = EGraph(save_egglog_string=True)
    x = egraph.let("x", i64(1))
    egraph.check(eq(x).to(i64(1)))
    assert "should start with `$`" not in capfd.readouterr().err
    node = expr_parts(x).ref.read()
    assert node.kind == Oneof("binding", "x")
    snapshot = egraph.freeze()
    assert snapshot.program is not None
    assert snapshot.program.declarations == []
    assert any(node.kind == Oneof("binding", "x") for node in snapshot.program.nodes)
    assert sum(command.kind is not None and command.kind.field == "let" for command in snapshot.program.commands) == 1
    with pytest.raises(TypeError):
        resolve_callable(x)
    _replay_recorded_source(egraph.as_egglog_string)


def test_named_let_actions_reconcile_across_independent_owners() -> None:
    from egglog._program import pack  # noqa: PLC0415

    first = let("shared", i64(1))
    second = let("shared", i64(2))
    packed = pack(commands=[first.__egg_ref__, second.__egg_ref__])
    assert packed.program.declarations == []
    targets = []
    for command in packed.program.commands:
        assert command.kind is not None
        assert command.kind.field == "let"
        target = command.kind.value.binding
        assert packed.program.nodes[target].kind == Oneof("binding", "shared")
        targets.append(target)
    assert targets[0] == targets[1]
    graph = EGraph(save_egglog_string=True)
    graph.register(first, second)
    saved = graph.let("shared", i64(3))
    graph.check(eq(saved).to(i64(3)))
    with pytest.raises(ValueError, match="source rendering does not support global rebinding"):
        _ = graph.as_egglog_string


def test_named_lets_preserve_exact_names_without_sanitization_collisions() -> None:
    names = ["", "x", "$x", "has space", "has_space", "caf\u00e9", "cafe\u0301", "x;y", "x_y"]
    graph = EGraph(save_egglog_string=True)
    for index, name in enumerate(names):
        saved = graph.let(name, i64(index))
        assert expr_parts(saved).ref.read().kind == Oneof("binding", name)
        assert get_let_name(saved) == name
        graph.check(eq(saved).to(i64(index)))
    snapshot = graph.freeze()
    assert snapshot.program is not None
    bindings = [
        node.kind.value for node in snapshot.program.nodes if node.kind is not None and node.kind.field == "binding"
    ]
    assert set(bindings) == set(names)
    assert len(bindings) == len(names)
    assert snapshot.program.declarations == []
    _replay_recorded_source(graph.as_egglog_string)


def test_named_let_rebinding_reads_old_value_and_preserves_failed_initializer() -> None:
    graph = EGraph()
    saved = graph.let("same", i64(1))
    graph.let("same", saved + i64(1))
    graph.check(eq(saved).to(i64(2)))
    graph.register(set_(saved).to(saved + i64(1)))
    graph.check(eq(saved).to(i64(3)))
    with pytest.raises(EggSmolError) as exc_info:
        graph.let("same", i64(1) / i64(0))
    error = exc_info.value.__dict__["_egglog_proto_error"]
    assert error.code == pb.ErrorCode.RUNTIME_FAILURE
    assert error.location is not None
    assert error.location.path == [0]
    graph.check(eq(saved).to(i64(3)))


def test_named_let_loop_rebinding_and_rule_head_rejection() -> None:
    graph = EGraph()
    saved = graph.let("count", i64(0))
    increment = let("count", saved + i64(1))
    graph.run(seq(increment) * 0)
    graph.check(eq(saved).to(i64(0)))
    graph.run(seq(increment) * 3)
    graph.check(eq(saved).to(i64(3)))
    graph.run(seq(let("count", saved)).saturate())
    graph.check(eq(saved).to(i64(3)))
    lexical = var("count", i64)
    graph.check(eq(lexical).to(i64(99)))
    assert get_var_name(lexical) == get_let_name(saved) == "count"
    with pytest.raises(TypeError, match=r"Let.*rule heads"):
        rule(eq(saved).to(i64(3))).then(increment)


def test_named_let_incompatible_output_sort_rejects_before_effects() -> None:
    graph = EGraph(save_egglog_string=True)
    saved = graph.let("same", i64(1))
    before, history = graph.freeze(), graph.as_egglog_string

    with pytest.raises(ValueError, match=r"^conflicting declaration __egglog_python_let_73616d65$"):
        graph.let("same", String("different sort"))

    assert graph.freeze() == before
    assert graph.as_egglog_string == history
    graph.check(eq(saved).to(i64(1)))
    _replay_recorded_source(graph.as_egglog_string)


def test_named_let_resolves_recreated_names_after_pop_and_in_other_graphs() -> None:
    graph = EGraph()
    parent = graph.let("parent", i64(1))
    graph.push()
    graph.check(eq(parent).to(i64(1)))
    child = graph.let("child", i64(2))
    graph.pop()
    graph.check_fail(eq(child).to(i64(2)))
    graph.check(eq(parent).to(i64(1)))

    recreated = graph.let("child", i64(3))
    graph.check(eq(child).to(recreated), eq(child).to(i64(3)))
    independent = EGraph()
    independent.check_fail(eq(parent).to(i64(1)))
    independent.let("child", i64(7))
    independent.check(eq(child).to(i64(7)))


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
        _replay_recorded_source(egraph.as_egglog_string)


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
        "user_defined_primitive",
    ],
)
def test_generated_relation_names_avoid_egglog_parser_tokens(name: str) -> None:
    relation_with_parser_token_name = relation(name)
    egraph = EGraph(save_egglog_string=True)

    egraph.register(relation_with_parser_token_name())

    egraph.check(relation_with_parser_token_name())


@pytest.mark.parametrize("generated_first", [True, False], ids=["generated-first", "explicit-first"])
def test_compatible_relation_aliases_share_one_backend_name(generated_first: bool) -> None:
    generated = relation("batch_generated_backend_name", i64)
    generated_candidate = f"{__name__.replace('.', '_')}_batch_generated_backend_name"
    explicit = relation("explicit_relation", i64, egg_fn=generated_candidate)
    generated_action = generated(i64(1))
    explicit_action = explicit(i64(2))
    actions = (generated_action, explicit_action) if generated_first else (explicit_action, generated_action)
    marker = relation("batch_collision_marker", i64)
    egraph = EGraph(marker(i64(0)), save_egglog_string=True)
    egraph.register(marker(i64(9)), *actions)
    egraph.check(marker(i64(9)), generated(i64(1)), explicit(i64(2)))
    assert egraph.function_size(generated) == egraph.function_size(explicit) == 2
    assert str(generated_action) == "batch_generated_backend_name(1)"
    assert str(explicit_action) == "explicit_relation(2)"
    _replay_recorded_source(egraph.as_egglog_string)

    egraph = EGraph(save_egglog_string=True)
    distinct = relation("distinct_explicit_relation", i64, egg_fn=f"{generated_candidate}_distinct")
    distinct_actions = (generated_action, distinct(i64(2)))
    egraph.register(*(distinct_actions if generated_first else distinct_actions[::-1]))
    egraph.check(generated(i64(1)), distinct(i64(2)))
    egraph.check_fail(generated(i64(2)))
    egraph.check_fail(distinct(i64(1)))
    joined = relation("distinct_backend_join", i64)
    value = var("distinct_backend_join_value", i64)
    join_rules = ruleset(rule(generated(value), distinct(value)).then(joined(value)))
    egraph.run(join_rules)
    assert egraph.function_size(joined) == 0
    egraph.register(distinct(i64(1)))
    egraph.run(join_rules)
    egraph.check(joined(i64(1)))
    egraph.check_fail(joined(i64(2)))
    assert egraph.function_size(joined) == 1
    _replay_recorded_source(egraph.as_egglog_string)


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
        _replay_recorded_source(egraph.as_egglog_string)


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

    if save_egglog_string:
        before = egraph.freeze()
        with pytest.raises(ValueError, match=r"[Uu]nbound"):
            egraph.register(SyntheticLetVariable.pair(shared, shared))
        assert egraph.freeze() == before
        assert not egraph.as_egglog_string
        egraph.register(SyntheticLetVariable.leaf(1))
        egraph.check(SyntheticLetVariable.leaf(1))
        _replay_recorded_source(egraph.as_egglog_string)
    else:
        with pytest.raises(EggSmolError, match="Unbound symbol"):
            egraph.register(SyntheticLetVariable.pair(shared, shared))
        egraph.register(SyntheticLetVariable.leaf(1))


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_compatible_relation_aliases_share_semantics_and_keep_local_paths(*, save_egglog_string: bool) -> None:
    generated = relation("late_explicit_backend_name", i64)
    marker = relation("late_relation_collision_marker", i64)
    egraph = EGraph(generated(i64(1)), marker(i64(0)), save_egglog_string=save_egglog_string)
    generated_name = f"{__name__.replace('.', '_')}_late_explicit_backend_name"
    explicit = relation("late_explicit_relation", i64, egg_fn=generated_name)
    egraph.register(marker(i64(9)), explicit(i64(2)))
    egraph.check(generated(i64(1)))
    egraph.check(generated(i64(2)), marker(i64(9)))
    assert egraph.function_size(generated) == egraph.function_size(explicit) == 2
    assert str(generated(i64(1))) == "late_explicit_backend_name(1)"
    assert str(explicit(i64(1))) == "late_explicit_relation(1)"
    snapshot = egraph.freeze()
    assert snapshot.program is not None
    assert sum(d.kind is not None and d.kind.value.name == generated_name for d in snapshot.program.declarations) == 1
    if save_egglog_string:
        _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_compatible_function_aliases_keep_frontend_names_out_of_semantic_identity(*, save_egglog_string: bool) -> None:
    @function(egg_fn="duplicate_explicit_backend")
    def first(value: i64Like) -> i64: ...

    @function(egg_fn="duplicate_explicit_backend")
    def second(value: i64Like) -> i64: ...

    marker = relation("late_function_collision_marker", i64)
    egraph = EGraph(set_(first(1)).to(i64(10)), marker(i64(0)), save_egglog_string=save_egglog_string)
    egraph.register(marker(i64(9)), set_(second(2)).to(i64(20)))
    assert egraph.lookup_function_value(first(1)) == i64(10)
    assert egraph.lookup_function_value(first(2)) == i64(20)
    egraph.check(marker(i64(9)))
    assert egraph.function_size(first) == egraph.function_size(second) == 2
    assert str(first(1)) == "first(1)"
    assert str(second(1)) == "second(1)"
    if save_egglog_string:
        _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_compatible_sort_aliases_keep_local_classes_and_shared_semantic_sort(*, save_egglog_string: bool) -> None:
    class GeneratedLateSort(Expr):
        def __init__(self) -> None: ...

    marker = relation("late_sort_collision_marker", i64)
    egraph = EGraph(GeneratedLateSort(), marker(i64(0)), save_egglog_string=save_egglog_string)

    class ExplicitLateSort(Expr, egg_sort=f"{__name__}.GeneratedLateSort"):
        def __init__(self) -> None: ...

    egraph.register(marker(i64(9)), ExplicitLateSort())
    egraph.check(GeneratedLateSort(), ExplicitLateSort(), marker(i64(9)))
    assert egraph.function_size(GeneratedLateSort) == 1
    assert egraph.function_size(ExplicitLateSort) == 1
    assert str(GeneratedLateSort()) == "GeneratedLateSort()"
    assert str(ExplicitLateSort()) == "ExplicitLateSort()"
    snapshot = egraph.freeze()
    assert snapshot.program is not None
    assert sum(sort.kind is not None and sort.kind.field == "eq" for sort in snapshot.program.sorts) == 1
    if save_egglog_string:
        _replay_recorded_source(egraph.as_egglog_string)


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
    _replay_recorded_source(egraph.as_egglog_string)


def test_saved_source_close_is_an_idempotent_noop() -> None:
    egraph = EGraph(save_egglog_string=True)
    egraph.let("x", i64(1))
    before = egraph.as_egglog_string
    egraph.close()
    egraph.close()
    assert egraph.as_egglog_string == before
    value = egraph.let("after_close", i64(2))
    assert egraph.lookup_function_value(value) == i64(2)


def test_saved_source_close_keeps_graph_and_export_usable() -> None:
    class ClosedExpr(Expr):
        def __init__(self, value: i64Like) -> None: ...

    egraph = EGraph(ClosedExpr(1), save_egglog_string=True, no_decomp=True)
    before = egraph.freeze()
    egraph.close()
    assert egraph.no_decomp()
    egraph.register(ClosedExpr(2))
    assert egraph.freeze() != before
    assert egraph.function_size(ClosedExpr) == 2
    _replay_recorded_source(egraph.as_egglog_string)


def test_close_without_a_saved_transcript_is_a_noop() -> None:
    egraph = EGraph()

    egraph.close()
    value = egraph.let("after_close", i64(1))

    egraph.check(eq(value).to(i64(1)))


def test_saved_source_pop_restores_parent_history_prefix() -> None:
    egraph = EGraph(save_egglog_string=True)
    outer = egraph.let("outer", i64(1))
    outer_source, outer_history = egraph.as_egglog_string, egraph.history()
    egraph.push()
    egraph.let("inner", i64(2))
    assert len(egraph.history().commands) > len(outer_history.commands)
    egraph.pop()
    assert egraph.history() == outer_history
    assert egraph.as_egglog_string == outer_source
    egraph.check(eq(outer).to(i64(1)))
    _replay_recorded_source(egraph.as_egglog_string)


def test_saved_source_reports_unextractable_cost_only_at_export() -> None:
    class RecordedCost(Expr):
        def __init__(self) -> None: ...

        @method(unextractable=True)
        def opaque(self) -> RecordedCost: ...

    graph = EGraph(RecordedCost(), save_egglog_string=True)
    graph.register(set_cost(RecordedCost().opaque(), 7))
    assert graph.function_size(RecordedCost.opaque) == 1
    before = graph.freeze()
    with pytest.raises(ValueError, match="source dynamic-cost helpers do not exist for unextractable constructors"):
        _ = graph.as_egglog_string
    assert graph.freeze() == before
    graph.check(RecordedCost())


def test_saved_source_keeps_failed_inputs_and_allows_recovery() -> None:
    graph = EGraph(save_egglog_string=True)
    graph.let("before_failure", i64(1))
    with pytest.raises(EggSmolError, match="recorded failure") as caught:
        graph.register(panic("recorded failure"))
    assert caught.value.__dict__["_egglog_proto_error"].code == pb.ErrorCode.RUNTIME_FAILURE
    assert "recorded failure" in graph.as_egglog_string
    value = graph.let("after_failure", i64(2))
    assert graph.lookup_function_value(value) == i64(2)
    graph.close()
    graph.close()
    with pytest.raises(EggSmolError, match="recorded failure"):
        _replay_recorded_source(graph.as_egglog_string)


def test_saved_pyobject_payload_stays_inert_during_export_failure(tmp_path: pathlib.Path) -> None:
    marker = tmp_path / "unpickle-calls"

    def rebuild() -> int:
        marker.write_text(marker.read_text() + "u" if marker.exists() else "u")
        return 7

    class Payload:
        def __reduce__(self) -> tuple[Callable[[], int], tuple[()]]:
            return rebuild, ()

    class InertRecordedCost(Expr):
        def __init__(self, value: PyObject) -> None: ...

        @method(unextractable=True)
        def opaque(self) -> InertRecordedCost: ...

    payload = PyObject(Payload())
    graph = EGraph(InertRecordedCost(PyObject(0)), save_egglog_string=True)
    graph.register(set_cost(InertRecordedCost(payload).opaque(), 7))
    assert not marker.exists()
    assert graph.function_size(InertRecordedCost) == 2
    before = graph.freeze()
    with pytest.raises(ValueError, match="source dynamic-cost helpers do not exist for unextractable constructors"):
        _ = graph.as_egglog_string
    assert not marker.exists()
    assert graph.freeze() == before
    assert graph.extract(payload).value == 7
    assert marker.read_text() == "u"


def test_saved_pyobject_source_omits_popped_branch_and_close_is_noop(tmp_path: pathlib.Path) -> None:
    marker = tmp_path / "provider-calls"

    def record_call(value: int) -> int:
        marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
        return value

    seen = relation("recorded_provider_scope", PyObject)
    graph = EGraph(seen(PyObject(record_call)(PyObject(1))), save_egglog_string=True)
    assert marker.read_text() == "x"
    outer = graph.freeze()
    graph.push()
    try:
        graph.register(seen(PyObject(record_call)(PyObject(2))))
        assert graph.function_size(seen) == 2
        assert marker.read_text() == "xx"
    finally:
        graph.pop()
    assert graph.freeze() == outer
    assert graph.function_size(seen) == 1
    transcript = graph.as_egglog_string
    graph.close()
    graph.close()
    assert graph.as_egglog_string == transcript
    _replay_recorded_source(transcript)
    assert marker.read_text() == "xxx"
    graph.register(seen(PyObject(record_call)(PyObject(3))))
    assert marker.read_text() == "xxxx"
    assert graph.function_size(seen) == 2


def test_saved_pyobject_callback_error_preserves_identity_and_input_history(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "provider-errors"
    original = LookupError("recorded provider failure")
    error_name = "_recorded_provider_error"
    monkeypatch.setattr(sys.modules[__name__], error_name, original, raising=False)

    def fail(_value: object) -> None:
        marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
        raise getattr(sys.modules[__name__], error_name)

    graph = EGraph(save_egglog_string=True)
    graph.let("before_provider_failure", i64(1))
    history = graph.as_egglog_string
    with pytest.raises(LookupError, match="recorded provider failure") as caught:
        graph.register(PyObject(fail)(PyObject(None)))
    assert caught.value is original
    assert marker.read_text() == "x"
    assert original.__traceback__ is not None
    assert original.__notes__[-1] == "While calling primitive 'py-call'"
    assert graph.as_egglog_string != history
    before = graph.freeze()
    with pytest.raises(LookupError, match="recorded provider failure") as caught_again:
        graph.register(PyObject(fail)(PyObject(None)))
    assert caught_again.value is original
    assert graph.freeze() == before
    assert marker.read_text() == "xx"
    _replay_recorded_source(history)
    assert marker.read_text() == "xx"


def test_saved_egglog_string_uses_qualified_generated_sort_and_function_names() -> None:
    class Num(Expr):
        @classmethod
        def var(cls, v: StringLike) -> Num: ...

    egraph = EGraph(save_egglog_string=True)
    value = Num.var("x")
    egraph.register(value)
    snapshot = pb.EGraphSnapshot.from_binary(egraph.freeze().to_binary())
    assert snapshot.program is not None
    sort = next(sort for sort in snapshot.program.sorts if sort.kind is not None and sort.kind.field == "eq")
    assert sort.kind is not None
    sort_definition = cast("RuntimeClass", Num).__egg_definition__
    assert isinstance(sort_definition, Ref)
    type_view = presentation(sort_definition).type_view
    assert type_view is not None
    assert type_view.path == (*__name__.split("."), "Num")
    declaration = next(
        declaration
        for declaration in snapshot.program.declarations
        if declaration.kind is not None and declaration.kind.field == "constructor"
    )
    view = presentation(resolve_callable(Num.var)).views[0]
    assert view.kind == CallKind.CLASS_METHOD
    assert view.path == ("var",)
    constructor_kind = declaration.kind
    assert constructor_kind is not None
    assert constructor_kind.field == "constructor"
    constructor = constructor_kind.value
    assert view.owner is not None
    assert view.owner.read().kind.value == sort.kind.value
    output = snapshot.program.sorts[constructor.output]
    assert output.kind is not None
    assert output.kind.field == "eq"
    assert output.kind.value == sort.kind.value
    argument = snapshot.program.sorts[constructor.inputs[0].sort]
    assert argument.kind is not None
    assert argument.kind.field == "primitive"
    assert argument.kind.value.name == "String"
    assert cast("RuntimeExpr", value).__egg_ref__.read().kind.value.func == constructor.name
    assert egraph.function_size(Num.var) == 1
    egraph.check(value)
    _replay_recorded_source(egraph.as_egglog_string)


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
    from egglog._program import pack  # noqa: PLC0415

    egraph = EGraph(save_egglog_string=True)
    value = Map[i64, BigRat].empty()
    packed = pack([expr_parts(value).ref])
    root = packed.program.nodes[packed.indices[0]]
    sort = packed.program.sorts[root.sort_id]
    assert sort.kind is not None
    assert sort.kind.field == "primitive"
    assert sort.kind.value.name == "Map"
    arguments = [packed.program.sorts[index] for index in sort.kind.value.args]
    argument_names = []
    for argument in arguments:
        argument_kind = argument.kind
        assert argument_kind is not None
        assert argument_kind.field == "primitive"
        argument_names.append(argument_kind.value.name)
    assert argument_names == ["i64", "BigRat"]

    egraph.register(value)
    egraph.check(eq(value.length()).to(i64(0)))
    _replay_recorded_source(egraph.as_egglog_string)


def test_registering_bare_variable_expression_raises() -> None:
    marker = relation("bare_variable_marker", i64)
    egraph = EGraph(marker(i64(0)))
    before = egraph.freeze()

    with pytest.raises(EggSmolError, match="Unbound symbol x") as caught:
        egraph.register(marker(i64(9)), var("x", i64))

    error = caught.value.__dict__["_egglog_proto_error"]
    assert error.code == pb.ErrorCode.INVALID_PROGRAM
    assert error.location is None
    assert egraph.freeze() == before
    egraph.check(marker(i64(0)))
    egraph.check_fail(marker(i64(9)))
    assert egraph.function_size(marker) == 1


def test_registering_let_reference_preserves_bound_value() -> None:
    egraph = EGraph(save_egglog_string=True)
    x = egraph.let("x", i64(1))
    before = egraph.freeze()

    egraph.register(x)

    assert egraph.freeze() == before
    assert egraph.lookup_function_value(x) == i64(1)
    node = cast("RuntimeExpr", x).__egg_ref__.read()
    assert node.kind is not None
    assert node.kind == Oneof("binding", "x")
    egraph.check(eq(x).to(i64(1)))
    _replay_recorded_source(egraph.as_egglog_string)


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
    from egglog._program import pack  # noqa: PLC0415

    class DuplicateEdge(Expr):
        @classmethod
        def leaf(cls, value: StringLike) -> DuplicateEdge: ...

        @classmethod
        def pair(cls, left: DuplicateEdge, right: DuplicateEdge) -> DuplicateEdge: ...

    leaf = DuplicateEdge.leaf("shared")
    first_pair = DuplicateEdge.pair(leaf, leaf)
    second_pair = DuplicateEdge.pair(first_pair, first_pair)
    root = DuplicateEdge.pair(second_pair, second_pair)
    egraph = EGraph(save_egglog_string=True)
    packed = pack([expr_parts(root).ref])
    pair_name = cast("RuntimeExpr", root).__egg_ref__.read().kind.value.func
    leaf_name = cast("RuntimeExpr", leaf).__egg_ref__.read().kind.value.func
    index = packed.indices[0]
    for _ in range(3):
        node = packed.program.nodes[index]
        assert node.kind is not None
        assert node.kind.field == "call"
        assert node.kind.value.func == pair_name
        left, right = node.kind.value.args
        assert left == right
        index = left
    leaf_kind = packed.program.nodes[index].kind
    assert leaf_kind is not None
    assert leaf_kind.field == "call"
    assert leaf_kind.value.func == leaf_name
    calls = [node.kind.value for node in packed.program.nodes if node.kind is not None and node.kind.field == "call"]
    assert sum(call.func == leaf_name for call in calls) == 1
    assert sum(call.func == pair_name for call in calls) == 3

    egraph.register(root)
    assert egraph.function_size(DuplicateEdge.leaf) == 1
    assert egraph.function_size(DuplicateEdge.pair) == 3
    egraph.check(root)
    _replay_recorded_source(egraph.as_egglog_string)


_DETERMINISTIC_HOISTING_SCRIPT = """
from __future__ import annotations

from egglog import EGraph, Expr, StringLike


class B(Expr):
    @classmethod
    def var(cls, name: StringLike) -> B: ...

    def __and__(self, other: B) -> B: ...

    def __or__(self, other: B) -> B: ...


egraph = EGraph(save_egglog_string=True)
x, y = B.var("x"), B.var("y")
shared = x & y
expr = shared | y
for _ in range(24):
    expr = (shared & expr) | (expr & (shared | x))
egraph.let("$e", expr)
print(egraph.as_egglog_string, end="")
"""


def test_shared_subexpression_lowering_is_deterministic_across_processes() -> None:
    transcripts = [
        subprocess.run(
            [sys.executable, "-c", _DETERMINISTIC_HOISTING_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout
        for seed in (1, 2, 3)
    ]

    assert transcripts[1:] == transcripts[:-1]


def test_shared_subexpression_inside_cost_lookup_replays_its_value() -> None:
    class CostLeaf(Expr):
        def __init__(self, name: StringLike) -> None: ...

    class CostPair(Expr):
        def __init__(self, left: CostLeaf, right: CostLeaf) -> None: ...

    class CostRoot(Expr):
        def __init__(self, pair: CostPair) -> None: ...

    class CostBox(Expr):
        def __init__(self, cost: i64Like) -> None: ...

    shared = CostLeaf("shared")
    root = CostRoot(CostPair(shared, shared))
    egraph = EGraph(save_egglog_string=True)
    egraph.register(set_cost(root, i64(3)))
    egraph.register(CostBox(get_cost(root)))
    egraph.check(eq(get_cost(root)).to(i64(3)), eq(CostBox(get_cost(root))).to(CostBox(3)))
    assert egraph.function_size(CostLeaf) == 1
    assert egraph.function_size(CostPair) == 1
    assert egraph.function_size(CostRoot) == 1
    assert egraph.function_size(CostBox) == 1
    _replay_recorded_source(egraph.as_egglog_string)


def test_freeze_omits_synthetic_let_bindings() -> None:
    class FreezeLetNum(Expr):
        @classmethod
        def var(cls, value: StringLike) -> FreezeLetNum: ...

        @classmethod
        def pair(cls, left: FreezeLetNum, right: FreezeLetNum) -> FreezeLetNum: ...

    shared = FreezeLetNum.var("shared")
    root = FreezeLetNum.pair(shared, shared)
    egraph = EGraph(save_egglog_string=True)
    egraph.register(root)

    snapshot = pb.EGraphSnapshot.from_binary(egraph.freeze().to_binary())
    assert snapshot.program is not None
    assert not any(
        declaration.kind is not None and declaration.kind.field == "function"
        for declaration in snapshot.program.declarations
    )
    pair_name = cast("RuntimeExpr", root).__egg_ref__.read().kind.value.func
    pairs = [
        node.kind.value
        for node in snapshot.program.nodes
        if node.kind is not None and node.kind.field == "call" and node.kind.value.func == pair_name
    ]
    assert len(pairs) == 1
    assert pairs[0].args[0] == pairs[0].args[1]
    restored = _restore_snapshot(snapshot, environment=(builtin_catalog().owner,))
    assert restored.function_size(FreezeLetNum.var) == 1
    assert restored.function_size(FreezeLetNum.pair) == 1
    restored.check(root)
    _replay_recorded_source(egraph.as_egglog_string)


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


@pytest.mark.parametrize("callback_kind", ["plain", "partial", "result-context"])
def test_multiset_map_infers_partial_result_from_closed_callback(monkeypatch, callback_kind: str) -> None:
    engine = MagicMock(side_effect=AssertionError("Callback authoring must not create an engine"))
    monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)
    from egglog.exp.array_api import Value, get_monomial  # noqa: PLC0415

    mss = var("mss", MultiSet[MultiSet[Value]])
    expected = mss.map(UnstableFn[MultiSet[Value], MultiSet[Value]](multiset_flat_map, get_monomial))
    if callback_kind == "plain":
        result = mss.map(lambda xs: multiset_flat_map(get_monomial, xs))
    elif callback_kind == "partial":
        result = mss.map(partial(multiset_flat_map, get_monomial))
        assert expr_parts(result) == expr_parts(expected)
    else:
        result = expected
    assert convert(result, MultiSet[MultiSet[Value]]) is result

    wrong = var("wrong", MultiSet[MultiSet[i64]])
    with pytest.raises(ConvertError) as expected_error:
        wrong.map(UnstableFn[MultiSet[Value], MultiSet[Value]](multiset_flat_map, get_monomial))  # type: ignore[arg-type]
    with pytest.raises(ConvertError) as actual_error:
        wrong.map(partial(multiset_flat_map, get_monomial))
    assert str(actual_error.value) == str(expected_error.value)
    engine.assert_not_called()


@pytest.mark.parametrize("callback_kind", ["plain", "named", "result-context", "different-result"])
def test_generic_named_callback_uses_known_input_context(monkeypatch, callback_kind: str) -> None:
    engine = MagicMock(side_effect=AssertionError("Callback authoring must not create an engine"))
    monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)
    from egglog.exp.array_api import Value  # noqa: PLC0415

    mss = var("mss", MultiSet[MultiSet[Value]])
    expected = mss.map(UnstableFn[MultiSet[Value], MultiSet[Value]](MultiSet.reset_counts))
    if callback_kind == "plain":
        result = mss.map(lambda xs: xs.reset_counts())
    elif callback_kind == "named":
        result = mss.map(MultiSet.reset_counts)
        assert expr_parts(result) == expr_parts(expected)
    elif callback_kind == "different-result":
        lengths = mss.map(MultiSet.length)
        assert convert(lengths, MultiSet[i64]) is lengths
        engine.assert_not_called()
        return
    else:
        result = expected
    assert convert(result, MultiSet[MultiSet[Value]]) is result
    engine.assert_not_called()


def test_generic_named_callback_rejects_incompatible_input_context(monkeypatch) -> None:
    engine = MagicMock(side_effect=AssertionError("Invalid callbacks must not create an engine"))
    monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)
    with pytest.raises(TypeConstraintError, match="Expected Vec, got MultiSet"):
        var("mss", MultiSet[MultiSet[i64]]).map(Vec.length)  # type: ignore[arg-type]
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        UnstableFn(Vec.empty)
    assert UnstableFn[Vec[i64]](Vec.empty).value == Vec[i64].empty
    engine.assert_not_called()


def test_generic_callback_context_keeps_binders_closed_and_separate(monkeypatch) -> None:
    engine = MagicMock(side_effect=AssertionError("Contextual typing must not create an engine"))
    monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)
    from egglog._program import Ref  # noqa: PLC0415
    from egglog.runtime import _function_value, resolve_type_annotation  # noqa: PLC0415

    integer = resolve_type_annotation(i64)
    assert isinstance(integer, Ref)
    callback = cast("RuntimeFunction", MultiSet.pick)
    by_result = _function_value(callback, (), None, context=([None], integer))
    assert expr_parts(cast("BaseExpr", by_result)) == expr_parts(UnstableFn[i64, MultiSet[i64]](MultiSet.pick))
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        _function_value(callback, (), None, context=([None], None))

    declaration = callback.__egg_ref__
    record: pb.Primitive = declaration.read()
    assert record.signature is not None

    unbound = declaration.owner.ref("sorts", record.signature.output)
    assert unbound.read().kind.field == "var"
    with pytest.raises(TypeConstraintError, match="Actual argument sorts must be closed"):
        _function_value(callback, (), None, context=([unbound], integer))
    engine.assert_not_called()


@pytest.mark.parametrize("mode", ["contextual", "closed", "unresolved", "decode", "error"])
def test_generic_callback_prefix_conversion_is_not_speculative(monkeypatch, mode: str) -> None:
    engine = MagicMock(side_effect=AssertionError("Prefix authoring must not create an engine"))
    monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)
    from protobuf import Oneof  # noqa: PLC0415

    from egglog._program import Builder, Ref, pack  # noqa: PLC0415
    from egglog.runtime import resolve_type_annotation  # noqa: PLC0415

    class Host:
        pass

    calls: list[Host] = []
    supplied, bad = Host(), Host()
    failure = ValueError("capture converter failed")

    def convert_host(value: Host) -> i64:
        calls.append(value)
        if value is bad:
            raise failure
        return i64(7)

    converter(Host, i64, convert_host)
    integer = resolve_type_annotation(i64)
    assert isinstance(integer, Ref)
    builder = Builder()
    scalar = builder.import_ref(integer)
    variable = builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    index = builder.add(
        "primitives",
        pb.Primitive(
            name="test.generic.capture",
            signature=pb.GenericSignature(
                type_param_count=1,
                inputs=[pb.Arg(sort=scalar), pb.Arg(sort=variable)],
                output=variable,
            ),
        ),
    )
    reference = builder.publish().ref("primitives", index)
    function = cast("Callable[[Host | i64, i64], i64]", RuntimeFunction(reference))
    before = reference.read().to_binary()
    xs = var("xs", MultiSet[i64])
    if mode == "contextual":
        actual = xs.map(partial(function, supplied))
        assert calls == [supplied]
        expected = xs.map(UnstableFn[i64, i64](function, i64(7)))
        assert expr_parts(actual) == expr_parts(expected)
    elif mode == "closed":
        value = UnstableFn[i64, i64](function, supplied)
        assert calls == [supplied]
        assert isinstance(value.value, partial)
        assert calls == [supplied]
    elif mode == "unresolved":
        with pytest.raises(TypeConstraintError, match="Unresolved"):
            UnstableFn(function, supplied)
        assert calls == []
    elif mode == "decode":
        value = cast("RuntimeExpr", UnstableFn[i64, i64](function, i64(7)))
        packed = pack([value.__egg_ref__])
        decoded = Builder.from_program(
            pb.Program.from_binary(packed.program.to_binary()), environment=(reference.owner,)
        ).publish()
        restored = RuntimeExpr(decoded.ref("nodes", packed.indices[0]))
        assert isinstance(restored.value, partial)
        assert calls == []
    else:
        with pytest.raises(ValueError, match=r"^capture converter failed$") as caught:
            xs.map(partial(function, bad))
        assert caught.value is failure
        assert calls == [bad]
    assert reference.read().to_binary() == before
    engine.assert_not_called()


@pytest.mark.parametrize(
    ("name", "rule_count"),
    [("to_polynomial_ruleset", 5), ("factor_ruleset", 1), ("from_polynomial_ruleset", 7)],
)
def test_polynomial_theory_authoring_snapshots_without_engine(monkeypatch, name: str, rule_count: int) -> None:
    engine = MagicMock(side_effect=AssertionError("Theory authoring must not create an engine"))
    monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)
    from egglog._program import Builder, pack  # noqa: PLC0415
    from egglog.exp import array_api  # noqa: PLC0415

    # Recreate authoring without replacing globals used by retained expressions.
    retained = array_api.Boolean(True)
    namespace = runpy.run_path(array_api.__file__, run_name=array_api.__name__)
    snapshot = namespace[name]._snapshot()
    packed = pack([snapshot])
    decoded = pb.Program.from_binary(packed.program.to_binary())
    addition = decoded.rulesets[0].value
    assert addition is not None
    assert len(addition.rules) == rule_count
    assert decoded.nodes
    assert decoded.declarations
    restored = Builder.from_program(decoded).publish().ref("rulesets", 0).read()
    assert restored.value is not None
    assert len(restored.value.rules) == rule_count
    assert retained.value is True
    engine.assert_not_called()


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
    from egglog_proto.egglog.v1 import egglog_pb as pb  # noqa: PLC0415

    class ReverseLookup(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def lookup(self, key: i64Like) -> String:
            return None  # type: ignore[return-value]  # None denotes a bodyless symbolic function.

    value = ReverseLookup(1).lookup(i64(2))
    egraph = EGraph()
    egraph.register(set_(value).to(String("found")))

    egraph.check(eq(value).to(String("found")))
    assert egraph.function_values(ReverseLookup.lookup) == {value: String("found")}
    before = egraph.freeze().to_binary()
    with pytest.raises(EggSmolError, match="invalid action target subtype") as caught:
        egraph.register(set_cost(value, 7))
    error = caught.value.__dict__["_egglog_proto_error"]
    assert error.code == pb.ErrorCode.INVALID_PROGRAM
    assert error.location is None
    assert egraph.freeze().to_binary() == before
    with pytest.raises(EggSmolError, match="GetCost requires a constructor target") as caught:
        egraph.check(eq(get_cost(value)).to(i64(7)))
    error = caught.value.__dict__["_egglog_proto_error"]
    assert error.code == pb.ErrorCode.INVALID_PROGRAM
    assert error.location is None
    assert egraph.freeze().to_binary() == before


def test_reverse_args_bodyless_constructor_round_trips_backend_argument_order() -> None:
    class ReverseResult(Expr): ...

    class ReverseSource(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def make(self, key: i64Like) -> ReverseResult: ...

    expr = ReverseSource(1).make(i64(2))

    egraph = EGraph(expr, set_cost(expr, 7))
    assert expr_parts(egraph.extract(expr)) == expr_parts(expr)
    egraph.check(eq(get_cost(expr)).to(i64(7)))


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


def test_bodyless_unit_callables_are_relations():
    class Work(Expr):
        def __init__(self, value: i64Like) -> None: ...
        def compile(self, next_sym: i64Like = 0) -> Unit: ...

    @function
    def pending(value: i64) -> Unit: ...

    @function
    def eager_unit(value: i64) -> Unit:
        return Unit()

    value = var("value", i64)
    graph = EGraph(Work(1).compile(), pending(i64(1)))
    advance = ruleset(
        rule(pending(value), Work(value).compile()).then(Work(value).compile(value + 1), pending(value + 1))
    )
    graph.run(1, ruleset=advance)
    graph.check(Work(1).compile(2), pending(i64(2)))
    graph.check_fail(Work(2).compile())
    graph.check_fail(pending(i64(3)))
    assert str(graph.extract(eager_unit(i64(3)))) == "Unit()"


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


@pytest.mark.parametrize("values", [(1, 2), [1, 2]])
def test_generic_family_intermediate_executes_and_replays(values) -> None:
    class Box(Expr):
        def __init__(self, values: Vec[i64]) -> None: ...

    converter(Vec[i64], Box, Box)
    value = convert(values, Box)
    expected = Box(convert(values, Vec[i64]))
    graph = EGraph(value, save_egglog_string=True)
    graph.check(eq(value).to(expected))
    assert graph.function_size(Box) == 1
    _replay_recorded_source(graph.as_egglog_string)


def test_heterogeneous_non_boolean_equality_executes_and_replays() -> None:
    class Symbolic(Expr):
        def __init__(self, value: i64) -> None: ...

        def __eq__(self, other: String) -> i64:  # type: ignore[override]  # Deliberate runtime DSL signature, not nominal LSP.
            return i64(7)

        def __ne__(self, other: String) -> String:  # type: ignore[override]  # Deliberate runtime DSL signature, not nominal LSP.
            return String("different")

    value, other = Symbolic(i64(1)), String("other")
    equal, unequal = value == other, value != other
    assert expr_parts(equal) == expr_parts(Symbolic.__eq__(value, other))
    assert expr_parts(unequal) == expr_parts(Symbolic.__ne__(value, other))
    graph = EGraph(value, save_egglog_string=True)
    graph.check(eq(equal).to(i64(7)), eq(unequal).to(String("different")))
    _replay_recorded_source(graph.as_egglog_string)


class TestMutate:
    def test_setitem_defaults(self):
        EGraph()

        class Foo(Expr):
            def __init__(self) -> None: ...
            def __setitem__(self, key: i64Like, value: i64Like) -> None: ...

        foo = Foo()
        foo[10] = 20
        assert isinstance(foo, Foo)
        assert str(foo) == "_Foo_1 = Foo()\n_Foo_1[10] = 20\n_Foo_1"
        arguments = get_callable_args(foo, Foo.__setitem__)
        assert arguments is not None
        receiver, key, value = arguments
        assert get_callable_fn(receiver) == Foo
        assert get_callable_args(receiver) == ()
        assert (get_literal_value(key), get_literal_value(value)) == (10, 20)

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
        assert get_callable_fn(x) == incr
        arguments = get_callable_args(x)
        assert arguments is not None
        assert [expr_parts(argument) for argument in arguments] == [expr_parts(x_copied)]
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
    assert expr_parts(expr) == expr_parts(Math(10) + Math(5))


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


@pytest.mark.parametrize("reverse_args", [False, True])
def test_numeric_cost_preserves_operand_order_and_only_runs_winner(reverse_args):
    class Small(Expr):
        def __init__(self, label: StringLike) -> None: ...
        def __sub__(self, other: Small) -> Small: ...

    class Large(Expr):
        def __init__(self, label: StringLike) -> None: ...

        @method(reverse_args=reverse_args)
        def __sub__(self, other: Large) -> Large: ...

        @classmethod
        def from_small(cls, value: Small) -> Large: ...

    calls = []

    def promote(value):
        calls.append("promote")
        return Large.from_small(value)

    converter(Small, Large, promote, cost=1)
    converter(Large, Small, lambda _: pytest.fail("Losing converter ran"), cost=100)
    left, right = Small("left"), Large("right")
    expression = left - right  # type: ignore[operator]  # Conversion dispatch promotes the authored Small receiver.
    expected = Large.__sub__(Large.from_small(left), right)
    assert expr_parts(expression) == expr_parts(expected)
    assert calls == ["promote"]
    egraph = EGraph(expression, save_egglog_string=True)
    egraph.check(eq(expression).to(expected))  # type: ignore[arg-type]  # Runtime promotion returns Large, not Small.
    egraph.check(eq(i64(10) - 3).to(i64(7)), eq(10 - i64(3)).to(i64(7)))
    assert egraph.function_size(Large.__sub__) == 1
    _replay_recorded_source(egraph.as_egglog_string)
    assert calls == ["promote"]


@pytest.mark.parametrize("reverse_args", [False, True])
def test_numeric_exact_heterogeneous_signature_keeps_primitive_result(reverse_args):
    class Offset(Expr):
        def __init__(self) -> None: ...

        @method(reverse_args=reverse_args)
        def __sub__(self, other: i64) -> i64:
            return other - 2

    converter(i64, Offset, lambda _: pytest.fail("Exact heterogeneous input was converted"), cost=0)
    receiver = Offset()
    expression = receiver - i64(5)
    assert expr_parts(expression) == expr_parts(Offset.__sub__(receiver, i64(5)))
    egraph = EGraph(receiver, save_egglog_string=True)
    egraph.check(eq(expression).to(i64(3)))
    _replay_recorded_source(egraph.as_egglog_string)


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
    def test_builtin_function_without_body(self, monkeypatch):
        engine = MagicMock(side_effect=AssertionError("Bodyless builtin authoring must not create an engine"))
        monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)
        from egglog._program import Builder, pack  # noqa: PLC0415

        @function(builtin=True, egg_fn="egglog.core.i64.add")
        def f(x: i64Like, y: i64Like) -> i64: ...

        assert expr_parts(f(1, 2)) == expr_parts(f(i64(1), i64(2)))
        reference = cast("RuntimeFunction", f).__egg_ref__
        record: pb.Primitive = reference.read()
        assert isinstance(record, pb.Primitive)
        primitive = record
        assert primitive.name == "egglog.core.i64.add"
        assert primitive.signature is not None

        signature = primitive.signature
        assert signature.type_param_count == 0
        assert signature.varargs == []

        assert signature.has_field("output")
        assert len(signature.inputs) == 2
        assert all(argument.sort == signature.output for argument in signature.inputs)
        sort: pb.Sort = reference.owner.ref("sorts", signature.output).read()
        assert sort.kind is not None
        assert sort.kind.field == "primitive"
        assert sort.kind.value.name == "i64"
        assert sort.kind.value.args == []
        assert len(presentation(reference).views) == 1
        view = presentation(reference).views[0]
        assert view.kind == CallKind.FUNCTION
        assert view.path == (*__name__.split("."), "f")
        assert view.owner is None
        assert view.receiver is None
        assert view.mutates is None
        assert len(view.params) == 2
        assert view.params[0].name == "x"
        assert view.params[0].core_input == 0
        assert view.params[0].default is None

        before = record.to_binary()
        value = cast("RuntimeExpr", f(1, 2))
        assert value.__egg_ref__.read().kind.value.func == primitive.name
        assert get_literal_value(f(1, 2)) is None
        packed = pack([value.__egg_ref__])
        assert not packed.program.declarations
        decoded = Builder.from_program(pb.Program.from_binary(packed.program.to_binary())).publish()
        restored = RuntimeExpr(decoded.ref("nodes", packed.indices[0]))
        assert expr_parts(cast("BaseExpr", restored)) == expr_parts(f(1, 2))
        assert cast("RuntimeFunction", f).__egg_ref__ is reference
        assert reference.read().to_binary() == before
        engine.assert_not_called()

        @function(builtin=True, egg_fn="missing.provider")
        def missing(x: i64Like) -> i64: ...

        with pytest.raises(LookupError, match="owned canonical definition"):
            missing(1)

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
        from egglog._program import Builder, pack  # noqa: PLC0415

        merged = constant("merged", A, merge=lambda old, _new: old)

        egraph = EGraph()
        egraph.register(set_(merged).to(A()))

        egraph.check(eq(merged).to(A()))
        assert egraph.function_values(merged) == {merged: A()}
        snapshot = egraph.freeze()
        assert snapshot.program is not None
        owner = Builder.from_program(snapshot.program).publish()
        restored = EGraph()
        restored._state.run_program(
            pack(
                roots=[
                    owner.ref(role, index)
                    for role in ("sorts", "nodes", "declarations")
                    for index, record in enumerate(getattr(snapshot.program, role))
                    if role == "declarations" or (record.kind is not None and record.kind.field in {"eq", "binding"})
                ],
                commands=[owner.ref("commands", index) for index in range(len(snapshot.program.commands))],
            )
        )
        restored.check(eq(merged).to(A()))

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
    assert V.__parameters__ == (T,)
    assert get_origin(get_args(V)[0]) is Vec
    assert get_args(get_args(V)[0]) == (T,)
    assert get_args(V[Unit]) == (Vec[Unit], int)  # type: ignore[misc]

    reverse_alias = int | Vec[T]
    assert reverse_alias.__parameters__ == (T,)
    assert get_args(reverse_alias[Unit]) == (int, Vec[Unit])  # type: ignore[misc]
    typing_alias = Union[Vec[T], int]  # noqa: UP007 -- exercise typing.Union substitution too.
    assert typing_alias.__parameters__ == (T,)  # type: ignore[attr-defined]
    assert get_args(typing_alias[Unit]) == (Vec[Unit], int)  # type: ignore[misc]

    partial_alias = Map[i64, T] | int
    assert get_args(partial_alias[String]) == (Map[i64, String], int)  # type: ignore[misc]
    repeated_alias = Map[T, T] | int
    assert repeated_alias.__parameters__ == (T,)
    assert get_args(repeated_alias[String]) == (Map[String, String], int)  # type: ignore[misc]
    nested_alias = Vec[Vec[T]] | int
    assert get_args(nested_alias[Unit]) == (Vec[Vec[Unit]], int)  # type: ignore[misc]

    with pytest.raises(TypeError, match="Too many arguments"):
        V[Unit, String]  # type: ignore[misc, type-arg]

    value = convert((), get_args(V[Unit])[0])  # type: ignore[misc]
    graph = EGraph(value, save_egglog_string=True)
    graph.check(eq(value).to(Vec[Unit].empty()))
    _replay_recorded_source(graph.as_egglog_string)


def test_override_hash():
    class A(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __hash__(self) -> int:
            return 42

    assert hash(A()) == 42


def test_serialize_warning_max_functions(monkeypatch):
    monkeypatch.setattr("egglog.visualizer_widget.VisualizerWidget.display_or_open", lambda self: None)

    class A(Expr):
        def __init__(self) -> None: ...

    egraph = EGraph()
    egraph.register(A())
    with pytest.warns(UserWarning, match="A"):
        egraph.display(max_functions=0)


def test_serialize_warning_max_calls(monkeypatch):
    monkeypatch.setattr("egglog.visualizer_widget.VisualizerWidget.display_or_open", lambda self: None)

    class A(Expr): ...

    @function
    def f(x: StringLike) -> A: ...

    egraph = EGraph()
    egraph.register(f("a"), f("b"))
    with pytest.warns(UserWarning, match="f"):
        egraph.display(max_calls_per_function=1)


def test_display_reads_snapshot_rows_and_cycles_without_mutation(monkeypatch):
    import json  # noqa: PLC0415

    graphs = []
    monkeypatch.setattr(
        "egglog.visualizer_widget.VisualizerWidget.display_or_open", lambda self: graphs.extend(self.egraphs)
    )

    class Displayed(Expr):
        def __init__(self, value: i64) -> None: ...
        def wrap(self) -> Displayed: ...

    @function
    def cell() -> String: ...

    member = relation("display_member", Displayed)
    value = Displayed(i64(1))
    graph = EGraph(union(value).with_(value.wrap()), set_(cell()).to(String('quote"\n</script>')), member(value))
    before = graph.freeze().to_binary()
    graph.display(split_primitive_outputs=False)
    data = json.loads(graphs[0])
    by_op = {node["op"]: node for node in data["nodes"].values()}
    wrapped = by_op["Displayed.wrap"]
    assert wrapped["eclass"] == by_op["Displayed"]["eclass"]
    assert data["nodes"][wrapped["children"][0]]["eclass"] == wrapped["eclass"]
    assert by_op["cell"]["eclass"] == by_op[repr('quote"\n</script>')]["eclass"]
    assert "display_member" in by_op
    assert graph.freeze().to_binary() == before
    dot = graph._graphviz().source
    assert "Displayed.wrap" in dot
    assert "cluster_" in dot
    assert "</script>" in dot
    assert "<svg" in graph._graphviz().pipe(format="svg", quiet=True, encoding="utf-8")


def test_saturate_visualizes_snapshot_history_and_failure_prefix(monkeypatch):
    import json  # noqa: PLC0415

    graphs = []
    monkeypatch.setattr(
        "egglog.visualizer_widget.VisualizerWidget.display_or_open", lambda self: graphs.append(self.egraphs)
    )

    @function(merge=lambda old, new: new)
    def cell() -> i64: ...

    graph = EGraph(set_(cell()).to(i64(1)))
    graph.saturate(max=1)
    assert len(graphs[0]) == 2
    assert all(any(node["op"] == "1" for node in json.loads(data)["nodes"].values()) for data in graphs[0])

    def failing_run(*args, **kwargs):
        graph.register(set_(cell()).to(i64(2)))
        msg = "after the prefix"
        raise RuntimeError(msg)

    monkeypatch.setattr(graph, "run", failing_run)
    with pytest.raises(RuntimeError, match="after the prefix"):
        graph.saturate(max=1)
    assert len(graphs[1]) == 2
    assert any(node["op"] == "2" for node in json.loads(graphs[1][-1])["nodes"].values())


def test_saturate_observation_failures_do_not_replace_the_original_error(monkeypatch):
    graph = EGraph()
    original = RuntimeError("original step failure")
    reads = 0

    def observe(**kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"nodes": {}, "class_data": {}, "root_eclasses": []}
        msg = "snapshot unavailable"
        raise ValueError(msg)

    def failing_run(*args, **kwargs):
        raise original

    def failing_display(self):
        msg = "display unavailable"
        raise OSError(msg)

    monkeypatch.setattr(graph, "_serialize", observe)
    monkeypatch.setattr(graph, "run", failing_run)
    monkeypatch.setattr("egglog.visualizer_widget.VisualizerWidget.display_or_open", failing_display)
    with pytest.raises(RuntimeError) as raised:
        graph.saturate(max=1)
    assert raised.value is original
    assert raised.value.__notes__ == [
        "Could not visualize the failure prefix (ValueError)",
        "Could not display snapshot history (OSError)",
    ]


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


def test_overall_diagnostic_document():
    assert isinstance(EGraph().stats(), pb.Diagnostic)


def test_function_values():
    egraph = EGraph()

    @function
    def f(x: i64Like) -> i64: ...

    egraph.register(set_(f(i64(1))).to(i64(2)))
    values = egraph.function_values(f)
    assert values == {f(i64(1)): i64(2)}


def test_lookup_function_value_does_not_materialize_constructor_arguments() -> None:
    class LookupReplayValue(Expr, egg_sort="LookupReplayValueSort"):
        @method(egg_fn="LookupReplayValue")
        def __init__(self, value: i64Like) -> None: ...

    @function(egg_fn="lookup-replay-result")
    def result(value: LookupReplayValue) -> i64: ...

    egraph = EGraph()
    egraph.register(set_(result(LookupReplayValue(1))).to(i64(10)))

    assert egraph.lookup_function_value(result(LookupReplayValue(2))) is None
    assert egraph.function_size(LookupReplayValue) == 1
    assert egraph.function_size(result) == 1

    egraph.register(LookupReplayValue(2))
    assert egraph.function_size(LookupReplayValue) == 2
    assert egraph.lookup_function_value(result(LookupReplayValue(2))) is None

    egraph.register(set_(result(LookupReplayValue(2))).to(i64(20)))
    assert egraph.lookup_function_value(result(LookupReplayValue(2))) == i64(20)
    assert egraph.extract(LookupReplayValue(2)) == LookupReplayValue(2)
    with pytest.raises(ValueError, match="cannot evaluate primitive keys"):
        egraph.lookup_function_value(result(LookupReplayValue(i64(1) + i64(1))))
    assert egraph.function_size(LookupReplayValue) == 2
    assert egraph.function_size(result) == 2


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


def test_builtin_callable_table_capabilities_reject_without_execution(monkeypatch, tmp_path) -> None:
    from egglog.egraph import get_callable_cost  # noqa: PLC0415

    egraph = EGraph()

    def forbidden(*args, **kwargs):
        pytest.fail("Builtin table capability inspection attempted engine execution")

    monkeypatch.setattr(egraph._state, "run_program", forbidden)
    expression = i64(1) + 2
    with pytest.raises(ValueError, match="table-backed"):
        egraph.function_size(i64.__add__)
    with pytest.raises(ValueError, match="table-backed"):
        egraph.function_values(i64.__add__)
    with pytest.raises(ValueError, match="table-backed"):
        egraph.lookup_function_value(expression)
    with pytest.raises(ValueError, match="table-backed"):
        egraph.keep_best(i64.__add__)
    with pytest.raises(ValueError, match="table-backed"):
        egraph.input(i64.__add__, str(tmp_path / "absent.tsv"))
    with pytest.raises(TypeError, match="table-backed Function"):
        set_(expression).to(i64(3))
    assert not egraph.has_custom_cost(i64.__add__)
    assert get_callable_cost(i64.__add__) == 1


@pytest.mark.parametrize(
    "filename",
    ['input\r"\\.csv', "input\u0300.csv"],
    ids=["control-and-escapes", "combining-unicode"],
)
def test_saved_transcript_replays_input_rows_after_file_removal(tmp_path: pathlib.Path, filename: str) -> None:
    @function
    def loaded() -> String: ...

    path = tmp_path / filename
    path.write_text("value\n")
    recorded = EGraph(save_egglog_string=True)

    recorded.input(loaded, str(path))
    recorded.check(eq(loaded()).to(String("value")))
    path.unlink()
    _replay_recorded_source(recorded.as_egglog_string)


def test_input_parses_all_rows_before_mutating(tmp_path: pathlib.Path) -> None:
    @function(merge=lambda old, new: new)
    def loaded(key: i64Like) -> i64: ...

    path = tmp_path / "rows.tsv"
    path.write_text("0\t2\ninvalid\t3\n")
    graph = EGraph(set_(loaded(9)).to(i64(99)), save_egglog_string=True)
    with pytest.raises(ValueError, match="format"):
        graph.input(loaded, str(path))
    assert graph.function_values(loaded) == {loaded(9): i64(99)}
    _replay_recorded_source(graph.as_egglog_string)


def test_input_preserves_order_and_relation_constructor_keys(tmp_path: pathlib.Path) -> None:
    def ordered_merge(old: i64, new: i64) -> i64:
        return old * 10 + new

    @function(merge=ordered_merge)
    def loaded(key: i64Like) -> i64: ...

    class Imported(Expr):
        def __init__(self, key: i64Like, name: StringLike) -> None: ...

    selected = relation("csv_selected", i64)
    path = tmp_path / "rows.tsv"
    graph = EGraph(set_(loaded(0)).to(i64(100)), save_egglog_string=True)
    path.write_text("0\t2\n0\t3\n")
    graph.input(loaded, str(path))
    assert graph.lookup_function_value(loaded(0)) == i64(10023)
    path.write_text("1\n1\n2\n")
    graph.input(selected, str(path))
    assert graph.function_size(selected) == 2
    path.write_text("1\tA\n1\tA\n2\tB\n")
    graph.input(Imported, str(path))
    assert graph.function_size(Imported) == 2
    path.unlink()
    _replay_recorded_source(graph.as_egglog_string)


def test_input_row_error_preserves_prefix_and_recovery(tmp_path: pathlib.Path) -> None:
    @function
    def loaded(key: i64Like) -> i64: ...

    path = tmp_path / "conflict.tsv"
    path.write_text("0\t1\n0\t2\n1\t3\n")
    graph = EGraph(set_(loaded(9)).to(i64(99)))
    with pytest.raises(EggSmolError, match="Illegal merge"):
        graph.input(loaded, str(path))
    assert graph.lookup_function_value(loaded(0)) == i64(1)
    assert graph.lookup_function_value(loaded(1)) is None
    graph.register(set_(loaded(8)).to(i64(88)))
    graph.check(eq(loaded(9)).to(i64(99)), eq(loaded(8)).to(i64(88)))


def test_input_rejects_unsupported_columns_before_opening_file(tmp_path: pathlib.Path) -> None:
    @function
    def loaded(key: Bool) -> i64: ...

    with pytest.raises(ValueError, match=r"[Uu]nsupported.*bool"):
        EGraph().input(loaded, str(tmp_path / "missing.tsv"))


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
        _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("explicit_ruleset", [False, True])
def test_initializer_cost_controls_extraction_and_source_replay(explicit_ruleset):
    destination = ruleset()

    class Costed(Expr, ruleset=destination if explicit_ruleset else None):
        @method(cost=3)
        def __init__(self, value: i64Like) -> None: ...

        @method(cost=1)
        @classmethod
        def cheap(cls, value: i64Like) -> Costed: ...

    expression = Costed(7)
    egraph = EGraph(expression, save_egglog_string=True)
    extracted, cost = egraph.extract(expression, include_cost=True)
    assert expr_parts(extracted) == expr_parts(expression)
    assert cost == 4
    cheaper = Costed.cheap(7)
    egraph.register(union(expression).with_(cheaper))
    extracted, cost = egraph.extract(expression, include_cost=True)
    assert expr_parts(extracted) == expr_parts(cheaper)
    assert cost == 2
    _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_dynamic_cost_rule_materializes_constructor_without_shadowing(*, save_egglog_string: bool) -> None:
    class RuleMaterializedCost(Expr):
        def __init__(self, value: i64Like) -> None: ...

    source = relation("dynamic_cost_materialize_source", i64)
    # Exercise a name used by compiler-generated rule lets: lowering must
    # allocate a distinct binding instead of shadowing this variable.
    value = var("__expr_0", i64)
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
        _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("save_egglog_string", [False, True], ids=["direct", "saved"])
def test_dynamic_cost_rejects_function_targets_without_effects(*, save_egglog_string: bool) -> None:
    @function
    def missing_target(value: i64Like) -> i64: ...

    expression = missing_target(1)
    egraph = EGraph(save_egglog_string=save_egglog_string)
    assert egraph.function_size(missing_target) == 0
    before = egraph.freeze()
    history = egraph.as_egglog_string if save_egglog_string else None
    expected_error = EggSmolError

    with pytest.raises(expected_error, match=r"^invalid action target subtype: ") as caught:
        egraph.register(set_cost(expression, 7))

    assert type(caught.value) is expected_error
    error = caught.value.__dict__["_egglog_proto_error"]
    assert error.code == pb.ErrorCode.INVALID_PROGRAM
    assert error.location is None
    assert egraph.freeze() == before
    if save_egglog_string:
        assert egraph.as_egglog_string == history
    with pytest.raises(expected_error, match=r"^GetCost requires a constructor target$") as caught:
        egraph.check(eq(get_cost(expression)).to(i64(7)))
    assert type(caught.value) is expected_error
    error = caught.value.__dict__["_egglog_proto_error"]
    assert error.code == pb.ErrorCode.INVALID_PROGRAM
    assert error.location is None
    assert egraph.freeze() == before
    if save_egglog_string:
        assert egraph.as_egglog_string == history
    assert egraph.function_size(missing_target) == 0
    assert egraph.lookup_function_value(expression) is None
    egraph.register(set_(expression).to(i64(2)))
    egraph.check(eq(expression).to(i64(2)))
    assert egraph.function_size(missing_target) == 1
    if save_egglog_string:
        _replay_recorded_source(egraph.as_egglog_string)


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
    _replay_recorded_source(egraph.as_egglog_string)
    assert marker.read_text() == "xx"


@pytest.mark.parametrize("cost_first", [False, True], ids=["raw-table-first", "cost-first"])
def test_dynamic_cost_is_independent_of_a_cost_named_user_table(cost_first: bool) -> None:
    class Costed(Expr): ...

    @function(egg_fn="canonical_cost_target", cost=41)
    def target(x: i64Like) -> Costed: ...

    @function(egg_fn="cost_table_canonical_cost_target")
    def raw_cost(x: i64Like) -> i64: ...

    egraph = EGraph(save_egglog_string=True)
    cost_action = set_cost(target(1), 7)
    raw_table_action = set_(raw_cost(2)).to(i64(5))
    cost_actions = (cost_action, raw_table_action) if cost_first else (raw_table_action, cost_action)
    egraph.register(
        target(2),
        *cost_actions,
    )

    assert egraph.lookup_function_value(raw_cost(1)) is None
    assert egraph.lookup_function_value(raw_cost(2)) == i64(5)
    assert egraph.lookup_function_value(get_cost(target(1))) == i64(7)
    assert egraph.extract(target(2), include_cost=True) == (target(2), 42)
    assert egraph.lookup_function_value(get_cost(target(2))) is None  # No declared-cost fallback.
    egraph.register(set_cost(target(1), 0), set_(raw_cost(1)).to(i64(29)))
    assert egraph.lookup_function_value(get_cost(target(1))) == i64(0)
    assert egraph.lookup_function_value(raw_cost(1)) == i64(29)
    assert egraph.has_custom_cost(target)
    egraph.register(delete(target(1)))
    assert egraph.lookup_function_value(get_cost(target(1))) is None
    assert not egraph.has_custom_cost(target)
    assert egraph.lookup_function_value(raw_cost(1)) == i64(29)
    assert egraph.lookup_function_value(raw_cost(2)) == i64(5)
    _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("cost_first", [False, True], ids=["raw-table-first", "cost-first"])
def test_dynamic_cost_uses_incoming_value_independently_of_user_merge(cost_first: bool) -> None:
    class Costed(Expr):
        @method(egg_fn="merged_cost_target")
        def __init__(self, x: i64Like) -> None: ...

    @function(egg_fn="cost_table_merged_cost_target", merge=lambda old, _new: old)
    def raw_cost(x: i64Like) -> i64: ...

    cost_action = set_cost(Costed(1), 7)
    raw_table_action = set_(raw_cost(2)).to(i64(5))
    actions = (cost_action, raw_table_action) if cost_first else (raw_table_action, cost_action)

    egraph = EGraph(*actions, save_egglog_string=True)
    egraph.register(
        set_(raw_cost(2)).to(i64(9)),
        set_cost(Costed(1), 0),
        set_cost(Costed(1), get_cost(Costed(1)) + raw_cost(2) + 4),
    )
    assert egraph.lookup_function_value(raw_cost(2)) == i64(5)
    assert egraph.lookup_function_value(raw_cost(1)) is None
    assert egraph.lookup_function_value(get_cost(Costed(1))) == i64(9)
    egraph.check(eq(get_cost(Costed(1))).to(i64(9)), eq(raw_cost(2)).to(i64(5)))
    _replay_recorded_source(egraph.as_egglog_string)


def test_cost_named_incompatible_declarations_are_rejected_without_effects() -> None:
    class Costed(Expr):
        def __init__(self, x: i64Like) -> None: ...

    @function(egg_fn="cost_table_aliased_cost_target")
    def first_raw_cost(x: i64Like) -> i64: ...

    @function(egg_fn="cost_table_aliased_cost_target", merge=lambda old, _new: old)
    def second_raw_cost(x: i64Like) -> i64: ...

    egraph = EGraph(Costed(1), save_egglog_string=True)
    before, history = egraph.freeze(), egraph.as_egglog_string
    with pytest.raises(
        ValueError, match=r"^Ambiguous same-name definitions require reconciliation:.*cost_table_aliased"
    ):
        egraph.register(
            set_cost(Costed(1), 7),
            set_(first_raw_cost(2)).to(i64(5)),
            set_(second_raw_cost(3)).to(i64(6)),
        )
    assert egraph.freeze() == before
    assert egraph.as_egglog_string == history
    assert egraph.lookup_function_value(get_cost(Costed(1))) is None
    assert egraph.function_size(first_raw_cost) == 0
    egraph.register(set_(first_raw_cost(2)).to(i64(5)), set_cost(Costed(1), 9))
    egraph.check(eq(first_raw_cost(2)).to(i64(5)), eq(get_cost(Costed(1))).to(i64(9)))
    _replay_recorded_source(egraph.as_egglog_string)


@pytest.mark.parametrize("cost_first", [False, True], ids=["raw-table-first", "cost-first"])
def test_dynamic_cost_rejects_primitive_targets_without_effects(cost_first: bool) -> None:
    @function(egg_fn="cost_table_/")
    def raw_cost(left: i64Like, right: i64Like) -> i64: ...

    cost_action = set_cost(i64(4) / i64(2), 7)
    raw_table_action = set_(raw_cost(1, 0)).to(i64(5))
    actions = (cost_action, raw_table_action) if cost_first else (raw_table_action, cost_action)

    for saved in [False, True]:
        egraph = EGraph(save_egglog_string=saved)
        assert egraph.function_size(raw_cost) == 0
        before = egraph.freeze()
        history = egraph.as_egglog_string if saved else None
        expected_error = EggSmolError
        with pytest.raises(expected_error, match=r"^unknown action target egglog.core.i64.div$") as caught:
            egraph.register(*actions)
        assert type(caught.value) is expected_error
        error = caught.value.__dict__["_egglog_proto_error"]
        assert error.code == pb.ErrorCode.RUNTIME_FAILURE
        assert error.location is None
        assert egraph.freeze() == before
        if saved:
            assert egraph.as_egglog_string == history
        assert egraph.function_size(raw_cost) == 0
        egraph.register(raw_table_action)
        egraph.check(eq(raw_cost(1, 0)).to(i64(5)))
        if saved:
            _replay_recorded_source(egraph.as_egglog_string)


def test_freeze_preserves_assigned_costs_and_independent_user_rows() -> None:
    class Costed(Expr):
        @method(egg_fn="freeze_cost_target")
        def __init__(self, x: i64Like) -> None: ...

    @function(egg_fn="cost_table_freeze_cost_target")
    def raw_cost(x: i64Like) -> i64: ...

    egraph = EGraph(
        set_(raw_cost(2)).to(i64(5)),
        Costed(2),
        set_cost(Costed(1), 7),
    )

    replayed = _restore_snapshot(egraph.freeze(), environment=(builtin_catalog().owner,))
    assert replayed.lookup_function_value(raw_cost(1)) is None
    assert replayed.lookup_function_value(raw_cost(2)) == i64(5)
    assert replayed.lookup_function_value(get_cost(Costed(1))) == i64(7)
    assert replayed.lookup_function_value(get_cost(Costed(2))) is None
    assert replayed.has_custom_cost(Costed)
    assert replayed.function_values(Costed) == egraph.function_values(Costed)


def test_freeze_preserves_costs_through_ordinary_constructor_aliases() -> None:
    class Costed(Expr):
        def __init__(self, value: i64Like) -> None: ...

    alias = Costed
    egraph = EGraph(set_cost(Costed(1), 5), set_cost(alias(2), 6))
    replayed = _restore_snapshot(egraph.freeze(), environment=(builtin_catalog().owner,))
    assert replayed.has_custom_cost(Costed)
    assert replayed.has_custom_cost(alias)
    assert replayed.lookup_function_value(get_cost(alias(1))) == i64(5)
    assert replayed.lookup_function_value(get_cost(Costed(2))) == i64(6)


def test_builtin_free_callable_alias_resolves_actual_canonical_provider() -> None:
    @function(egg_fn="egglog.core.i64.add", builtin=True)
    def plus_alias(left: i64Like, right: i64Like) -> i64: ...

    value = cast("RuntimeExpr", plus_alias(1, 2))
    assert value.__egg_ref__.read().kind.value.func == "egglog.core.i64.add"
    EGraph().check(eq(cast("BaseExpr", value)).to(i64(3)))


@pytest.mark.parametrize("frontier", ["mutation", "variadic", "generic"])
def test_bodyless_builtin_rejects_unmigrated_frontiers_before_conversion(monkeypatch, frontier: str) -> None:
    engine = MagicMock(side_effect=AssertionError("Invalid builtin headers must not create an engine"))
    monkeypatch.setattr(egg_bindings, "_ProtoEngine", engine)

    class Host:
        pass

    calls = []

    def convert_host(value: Host) -> i64:
        calls.append(value)
        return i64(3)

    converter(Host, i64, convert_host)
    if frontier == "mutation":

        @function(builtin=True, mutates_first_arg=True)
        def mutation(value: i64Like) -> None: ...

        candidate = cast("Callable[[object], object]", mutation)
    elif frontier == "variadic":

        @function(builtin=True)
        def variadic(*values: i64Like) -> i64: ...

        candidate = cast("Callable[[object], object]", variadic)
    elif frontier == "generic":

        @function(builtin=True)
        def generic(value: Vec[_BuiltinExprT]) -> i64: ...

        candidate = cast("Callable[[object], object]", generic)
    with pytest.raises(NotImplementedError):
        candidate(Host())
    assert calls == []
    engine.assert_not_called()


def test_freeze_preserves_reverse_function_rows_and_constructor_costs() -> None:
    class ReverseFreeze(Expr):
        def __init__(self, value: i64Like) -> None: ...

        @method(reverse_args=True)
        def label(self, value: StringLike) -> i64: ...

        @method(reverse_args=True)
        def make(self, key: StringLike) -> ReverseFreeze: ...

    expression = ReverseFreeze(1).label("key")
    constructor = ReverseFreeze(1).make("other")
    egraph = EGraph(set_(expression).to(i64(3)), set_cost(constructor, 7))
    replayed = _restore_snapshot(egraph.freeze(), environment=(builtin_catalog().owner,))
    assert replayed.lookup_function_value(expression) == i64(3)
    assert replayed.function_values(ReverseFreeze.label) == {expression: i64(3)}
    assert expr_parts(replayed.extract(constructor)) == expr_parts(constructor)
    assert replayed.lookup_function_value(get_cost(constructor)) == i64(7)


def test_dynamic_cost_rejects_exact_primitive_targets_without_recording_them() -> None:
    from egglog._program import pack  # noqa: PLC0415

    expressions = [i64(1) + i64(2), Rational(1, 2) + Rational(1, 3)]
    targets = []
    for expression in expressions:
        packed = pack([expr_parts(expression).ref])
        node = packed.program.nodes[packed.indices[0]]
        assert node.kind is not None
        assert node.kind.field == "call"
        targets.append(node.kind.value.func)
    assert targets == ["egglog.core.i64.add", "egglog.experimental.rational.add"]
    egraph = EGraph(save_egglog_string=True)
    before, history = egraph.freeze(), egraph.as_egglog_string
    for expression, target in zip(expressions, targets, strict=True):
        with pytest.raises(ValueError, match=r"^unknown action target ") as caught:
            egraph.register(set_cost(expression, 5))
        assert str(caught.value) == f"unknown action target {target}"
        assert egraph.freeze() == before
        assert egraph.as_egglog_string == history
    assert not egraph.has_custom_cost(i64.__add__)
    assert not egraph.has_custom_cost(Rational.__add__)
    _replay_recorded_source(egraph.as_egglog_string)


def test_signed_assigned_costs_fail_only_in_the_affected_default_extraction_domain() -> None:
    class Costed(Expr):
        def __init__(self, value: i64Like) -> None: ...

    class Unrelated(Expr):
        def __init__(self, value: i64Like) -> None: ...

    extractors: list[ExtractionMode] = ["tree", "greedy-dag"]
    for extractor in extractors:
        egraph = EGraph(set_cost(Costed(1), -1), Unrelated(1))
        assert egraph.lookup_function_value(get_cost(Costed(1))) == i64(-1)
        assert egraph.extract(Unrelated(1), extractor=extractor) == Unrelated(1)
        before = egraph.freeze()
        with pytest.raises(EggSmolError, match="negative assigned costs") as caught:
            egraph.extract(Costed(1), extractor=extractor)
        error = caught.value.__dict__["_egglog_proto_error"]
        assert error.code == pb.ErrorCode.RUNTIME_FAILURE
        assert error.location == pb.CommandLocation(path=[0])
        assert egraph.freeze() == before
        assert egraph.lookup_function_value(get_cost(Costed(1))) == i64(-1)


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

        transcript = egraph.as_egglog_string
        source = egg_bindings._ProtoEngine.source(transcript)
        bindings, runs, actions = [], [], []
        while (submission := source.next_submission()) is not None:
            kind, encoded, count = submission
            assert kind == "program"
            assert count == 0
            assert encoded is not None
            pending = list(reversed(pb.Program.from_binary(encoded).commands))
            while pending:
                command = pending.pop()
                assert command.kind is not None
                if command.kind.field == "bind_scheduler":
                    bindings.append(command.kind.value)
                elif command.kind.field == "action":
                    actions.append(command.kind.value)
                elif command.kind.field == "run":
                    runs.append(command.kind.value)
                elif command.kind.field == "loop":
                    pending.extend(reversed(command.kind.value.body))
                else:
                    pytest.fail(f"Unexpected recorded command: {command.kind.field}")
        assert len(bindings) == 1
        assert len(runs) == 2
        assert len(actions) == 2
        for action in actions:
            action_kind = action.kind
            assert action_kind is not None
            assert action_kind.field == "term"
        assert {item.scheduler for item in runs} == {bindings[0].name}
        _replay_recorded_source(transcript)

    def test_scheduler_scope_does_not_leak_to_sequence_sibling(self):
        r = ruleset(name="scheduler-lexical-scope")
        scheduler = back_off(match_limit=2, ban_length=2)
        egraph = EGraph(save_egglog_string=True)

        egraph.run(seq(scheduler.scope(run(r, scheduler=scheduler)), run(r, scheduler=scheduler)))

        transcript = egraph.as_egglog_string
        source = egg_bindings._ProtoEngine.source(transcript)
        commands = []
        while (submission := source.next_submission()) is not None:
            kind, encoded, count = submission
            assert kind == "program"
            assert count == 0
            assert encoded is not None
            commands.extend(pb.Program.from_binary(encoded).commands)
        pending = [(command, 0) for command in reversed(commands)]
        bindings, runs = [], []
        while pending:
            command, depth = pending.pop()
            kind = command.kind
            assert kind is not None
            if kind.field == "bind_scheduler":
                bindings.append((kind.value, depth))
            elif kind.field == "run":
                runs.append((kind.value, depth))
            else:
                assert kind.field == "loop"
                pending.extend((child, depth + 1) for child in reversed(kind.value.body))
        assert len(bindings) == len(runs) == 2
        (outer, outer_depth), (inner, inner_depth) = bindings
        assert outer.name != inner.name
        assert outer_depth < inner_depth
        for binding in (outer, inner):
            assert binding.back_off is not None
            assert binding.back_off.match_limit == 2
            assert binding.back_off.ban_length == 2
        assert runs[0][0].scheduler == inner.name
        assert runs[0][1] >= inner_depth
        assert runs[1][0].scheduler == outer.name
        assert runs[1][1] < inner_depth
        assert runs[0][0].rulesets == runs[1][0].rulesets
        _replay_recorded_source(transcript)

    def test_persistent_scheduler_child_binding_is_redeployed_after_pop(self):
        source = relation("child_scheduler_source", i64)
        copied = relation("child_scheduler_copied", i64)
        x = var("x", i64)
        copy = ruleset(rule(source(x)).then(copied(x)), name="child-scheduler-copy")
        scheduler = back_off(match_limit=2, ban_length=2).persistent()
        egraph = EGraph(source(1))  # type: ignore[arg-type]
        with egraph:
            egraph.run(run(copy, scheduler=scheduler))
            egraph.check(copied(1))  # type: ignore[arg-type]
        egraph.check_fail(copied(1))  # type: ignore[arg-type]
        egraph.run(run(copy, scheduler=scheduler))
        egraph.check(copied(1))  # type: ignore[arg-type]

    def test_custom_scheduler_until_accepts_equality_and_conjunctions(self):
        """Canonical loop queries have no legacy single/non-equality limit."""
        source = relation("scheduler_until_source", i64)
        copied = relation("scheduler_until_copied", i64)
        x = var("x", i64)
        r = ruleset(rule(source(x)).then(copied(x)), name="scheduler-until-copy")
        bo = back_off(match_limit=1)
        egraph = EGraph(source(1))  # type: ignore[arg-type]
        egraph.run(run(r, eq(i64(1)).to(i64(1)), scheduler=bo))
        egraph.check_fail(copied(1))  # type: ignore[arg-type]
        # A conjunction with only one present fact does not stop the body.
        egraph.run(run(r, source(1), source(2), scheduler=bo))  # type: ignore[arg-type]
        egraph.check(copied(1))  # type: ignore[arg-type]
        # When every conjunct holds, including equality, no body executes.
        egraph = EGraph(source(1), source(2))  # type: ignore[arg-type]
        egraph.run(run(r, source(1), source(2), eq(i64(1)).to(i64(1)), scheduler=bo))  # type: ignore[arg-type]
        egraph.check_fail(copied(1))  # type: ignore[arg-type]
        egraph.check_fail(copied(2))  # type: ignore[arg-type]


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


def test_keep_best_discards_unselected_bindings_until_a_parent_scope_is_restored() -> None:
    class CompactKey(Expr):
        def __init__(self, value: i64Like) -> None: ...

    @function
    def score(key: CompactKey) -> i64: ...

    egraph = EGraph(CompactKey(1), set_(score(CompactKey(1))).to(i64(11)))
    parent_key = egraph.let("parent-key", CompactKey(1))
    observed = egraph.lookup_function_value(CompactKey(1))
    assert observed is not None

    egraph.push()
    egraph.keep_best(score)
    compacted = egraph.freeze()
    assert compacted.program is not None
    retained_slots = [
        index for index, node in enumerate(compacted.program.nodes) if node.kind == Oneof("binding", "parent-key")
    ]
    assert len(retained_slots) == 1
    assert not any(
        command.kind is not None and command.kind.field == "let" and command.kind.value.binding == retained_slots[0]
        for command in compacted.program.commands
    )
    assert egraph.lookup_function_value(parent_key) is None
    fresh_key = egraph.let("fresh-key", CompactKey(1))
    assert egraph.lookup_function_value(score(fresh_key)) == i64(11)
    assert egraph.lookup_function_value(score(parent_key)) is None
    with pytest.raises(ValueError, match="observation scope"):
        egraph.lookup_function_value(score(observed))

    egraph.pop()
    assert egraph.lookup_function_value(score(parent_key)) == i64(11)
    assert egraph.lookup_function_value(score(fresh_key)) is None

    egraph.keep_best(score)
    assert egraph.lookup_function_value(score(parent_key)) is None
    root_fresh_key = egraph.let("root-fresh-key", CompactKey(1))
    assert egraph.lookup_function_value(score(root_fresh_key)) == i64(11)


class TestCustomExtract:
    def test_callback_keeps_builtin_and_user_class_member_identity(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...
            def size(self) -> i64: ...

        class Root(Expr):
            def __init__(self, key: Key) -> None: ...

        graph = EGraph(set_(Key(3).size()).to(i64(17)))

        def model(callback_graph: EGraph, expr: BaseExpr, children: list[int]) -> int:
            if isinstance(expr, i64):
                return 1
            if isinstance(expr, Key):
                return sum(children)
            assert isinstance(expr, Root)
            args = get_callable_args(expr)
            assert args is not None
            key = args[0]
            assert isinstance(key, Key)
            value = callback_graph.lookup_function_value(cast("Key", key).size())
            assert isinstance(value, i64)
            return int(value) + sum(children)

        assert graph.extract(Root(Key(3)), include_cost=True, cost_model=model) == (Root(Key(3)), 18)

    def test_param_cost_callback_recognizes_original_classes(self) -> None:
        from egglog.exp.param_eq import Num, ParamCost, param_cost_model  # noqa: PLC0415

        def model(callback_graph: EGraph, expr: BaseExpr, children: list[ParamCost]) -> ParamCost:
            assert isinstance(expr, (f64, Num))
            return param_cost_model(callback_graph, expr, children)

        expr = Num(2.5)
        assert EGraph().extract(expr, include_cost=True, cost_model=model) == (expr, ParamCost(floats=1))

    def test_callback_freeze_rejects_reentry_and_other_graph_remains_independent(self) -> None:
        graph, other = EGraph(), EGraph()

        def model(callback_graph: EGraph, _expr: BaseExpr, _children: list[int]) -> int:
            snapshot = other.freeze()
            assert snapshot.options is not None
            assert snapshot.options.has_field("cost_sort")
            callback_graph.freeze()
            return 1

        with pytest.raises(RuntimeError, match="Cannot execute engine operations during a cost-model callback"):
            graph.extract(i64(1), cost_model=model)
        assert graph.freeze().options is not None

    def test_model_objectives_leave_i64_annotations_independent(self) -> None:
        class Item(Expr):
            def __init__(self, value: i64Like) -> None: ...

        graph = EGraph(Item(1), set_cost(Item(1), 7))

        def boolean_objective(_graph: EGraph, expr: BaseExpr, children: list[bool]) -> bool:
            return isinstance(expr, Item) or any(children)

        assert graph.extract(Item(1), include_cost=True) == (Item(1), 8)
        assert graph.extract(Item(1), include_cost=True, cost_model=boolean_objective) == (Item(1), True)
        assert graph.extract(get_cost(Item(1))) == i64(7)
        graph.push()
        try:
            graph.register(set_cost(Item(1), 9))
            assert graph.extract(Item(1), include_cost=True, cost_model=default_cost_model) == (Item(1), 10)
            assert graph.extract(Item(1), include_cost=True, cost_model=boolean_objective) == (Item(1), True)
        finally:
            graph.pop()
        assert graph.extract(Item(1), include_cost=True, cost_model=default_cost_model) == (Item(1), 8)

    def test_tree_keep_best_ranks_in_d_and_preserves_c(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function(merge=lambda old, new: new)
        def selected() -> Key: ...

        graph = EGraph(
            set_(selected()).to(Key(1)),
            union(Key(1)).with_(Key(2)),
            set_cost(Key(1), 99),
            set_cost(Key(2), 7),
        )

        def prefer_two(_graph: EGraph, expr: BaseExpr, children: list[bool]) -> bool:
            args = get_callable_args(expr)
            return bool(isinstance(expr, Key) and args is not None and cast("i64", args[0]).value == 1) or any(children)

        graph.keep_best(selected, cost_model=prefer_two)
        assert graph.extract(selected()) == Key(2)
        assert graph.extract(get_cost(Key(2))) == i64(7)

    def test_callback_exception_retains_identity_and_next_request_works(self) -> None:
        class Item(Expr):
            def __init__(self, value: i64Like) -> None: ...

        graph = EGraph()
        root = Item(1)
        original = LookupError("original custom model failure")

        def fail(_graph: EGraph, _expr: BaseExpr, _children: list[int]) -> int:
            raise original

        with pytest.raises(LookupError) as caught:
            graph.extract(root, cost_model=fail)
        assert caught.value is original
        name = cast("RuntimeExpr", root).__egg_ref__.read().kind.value.func
        assert ("callable", name) in graph._state.definitions
        assert graph.extract(i64(2)) == i64(2)

    def test_literal_root(self) -> None:
        def is_even_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            del egraph, children_costs
            return int(isinstance(expr, i64) and expr.value % 2 == 0)

        assert EGraph().extract(i64(10), include_cost=True, cost_model=is_even_cost_model) == (i64(10), 1)
        assert EGraph().extract(i64(5), include_cost=True, cost_model=is_even_cost_model) == (i64(5), 0)

    def test_lookup_value_root_uses_explicit_binding(self) -> None:
        class LookupExpr(Expr):
            def __init__(self, value: i64Like) -> None: ...

        egraph = EGraph(LookupExpr(1))
        observed = egraph.lookup_function_value(LookupExpr(1))

        assert observed is not None
        assert egraph.lookup_function_value(LookupExpr(2)) is None
        assert egraph.function_size(LookupExpr) == 1
        value = egraph.let("lookup-root", LookupExpr(1))
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
    def test_host_model_preserves_portable_value_roots(self, expr):
        egraph = EGraph()
        expected = egraph.extract(expr)
        assert egraph.extract(expr, cost_model=default_cost_model) == expected

    def test_compare_values(self):
        egraph = EGraph()
        egraph.register(E(), gg())
        e_value = egraph.extract(E())
        gg_value = egraph.extract(gg())
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
        xs = EGraph().extract(Vec(x))
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
            should_capture=lambda candidate, children_costs: (
                isinstance(candidate, MultiSet) and len(children_costs) == 4
            ),
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

    @pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
    @pytest.mark.parametrize("missing_first", [False, True], ids=["opaque-first", "missing-first"])
    @pytest.mark.parametrize("foreign", [False, True], ids=["stale", "foreign"])
    def test_lookup_validates_all_keys_before_returning_missing_row(
        self, *, nested: bool, missing_first: bool, foreign: bool
    ) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        class PairKey(Expr):
            def __init__(self, left: Key, right: Key) -> None: ...

        @function
        def score(left: Key, right: Key) -> i64: ...

        @function
        def nested_score(value: PairKey) -> i64: ...

        egraph = EGraph(
            set_(score(Key(1), Key(1))).to(i64(7)),
            set_(nested_score(PairKey(Key(1), Key(1)))).to(i64(8)),
        )
        observed_graph = EGraph(Key(1)) if foreign else egraph
        observed = observed_graph.lookup_function_value(Key(1))
        assert observed is not None
        arguments = (Key(99), observed) if missing_first else (observed, Key(99))
        query = nested_score(PairKey(*arguments)) if nested else score(*arguments)
        with pytest.raises(ValueError, match="observation scope"):
            egraph.lookup_function_value(query)

        assert egraph.lookup_function_value(score(Key(99), Key(1))) is None
        assert egraph.lookup_function_value(nested_score(PairKey(Key(99), Key(1)))) is None
        assert egraph.function_size(Key) == 1
        assert egraph.function_size(PairKey) == 1
        assert egraph.function_size(score) == 1
        assert egraph.function_size(nested_score) == 1

    def test_lookup_captured_reads_follow_push_scope_lifetimes(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function
        def score(value: Key) -> i64: ...

        egraph = EGraph(Key(1), set_(score(Key(1))).to(i64(11)))
        parent_key = egraph.let("parent-key", Key(1))
        observed = egraph.lookup_function_value(Key(1))
        assert observed is not None

        # Push clones the graph, including the parent's ordinary binding row.
        egraph.push()
        assert egraph.lookup_function_value(score(parent_key)) == i64(11)
        with pytest.raises(ValueError, match="observation scope"):
            egraph.lookup_function_value(score(observed))
        egraph.register(Key(2), set_(score(Key(2))).to(i64(22)))
        child_key = egraph.let("child-key", Key(2))
        assert egraph.lookup_function_value(score(child_key)) == i64(22)
        egraph.pop()

        assert egraph.lookup_function_value(score(parent_key)) == i64(11)
        egraph.register(Key(3), set_(score(Key(3))).to(i64(33)))
        assert egraph.lookup_function_value(score(child_key)) is None
        assert egraph.function_size(Key) == 2
        independent = EGraph(set_(score(Key(1))).to(i64(111)))
        assert independent.lookup_function_value(score(parent_key)) is None

    def test_captured_lookup_read_survives_source_writes_and_delete_without_replay(self) -> None:
        class Key(Expr):
            def __init__(self, value: i64Like) -> None: ...

        @function(merge=lambda old, new: new)
        def source(index: i64Like) -> Key: ...

        @function
        def score(value: Key) -> i64: ...

        egraph = EGraph(
            set_(source(0)).to(Key(1)),
            set_(score(Key(1))).to(i64(11)),
            set_(score(Key(2))).to(i64(22)),
        )
        saved = egraph.let("saved-source", source(0))
        egraph.register(set_(source(0)).to(Key(2)))
        assert egraph.lookup_function_value(score(saved)) == i64(11)
        assert egraph.lookup_function_value(score(source(0))) == i64(22)

        egraph.register(delete(source(0)), delete(Key(1)), delete(Key(2)))
        assert egraph.lookup_function_value(source(0)) is None
        assert egraph.lookup_function_value(score(source(0))) is None
        assert egraph.lookup_function_value(score(saved)) == i64(11)
        assert egraph.function_size(source) == 0
        assert egraph.function_size(Key) == 0
        with pytest.raises(ValueError, match="Unextractable root"):
            egraph.extract(saved)
        assert egraph.function_size(source) == 0
        assert egraph.function_size(Key) == 0

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

        expr = Root(EGraph().extract(Map[Key, i64].empty().insert(Key(3), i64(0))))
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
