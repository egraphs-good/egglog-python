from __future__ import annotations

import gc
import pathlib
from typing import Literal, TextIO, cast

import pytest

import egglog.bindings as egg_bindings
from egglog import (
    EggSmolError,
    EGraph,
    Map,
    PyObject,
    eq,
    f64,
    i64,
    map_fold_kv,
    relation,
    rule,
    ruleset,
    run,
    set_current_ruleset,
    var,
)
from egglog.declarations import (
    ClassDecl,
    Declarations,
    FunctionDecl,
    FunctionRef,
    FunctionSignature,
    HasDeclarations,
    Ident,
    JustTypeRef,
    LitDecl,
    TypedExprDecl,
    TypeRefWithVars,
)
from egglog.egraph import get_current_ruleset
from egglog.runtime import RuntimeExpr


def test_saved_egglog_transcript_close_removes_backing_file() -> None:
    egraph = EGraph(save_egglog_string=True)
    assert egraph._state.egglog_file_state is not None
    path = pathlib.Path(egraph._state.egglog_file_state.path)

    egraph.close()

    assert not path.exists()


def test_saved_egglog_transcript_is_removed_on_finalization() -> None:
    egraph = EGraph(save_egglog_string=True)
    assert egraph._state.egglog_file_state is not None
    path = pathlib.Path(egraph._state.egglog_file_state.path)
    assert path.exists()

    del egraph
    gc.collect()

    assert not path.exists()


def test_closing_saved_transcript_inside_context_restores_scope() -> None:
    egraph = EGraph(save_egglog_string=True)
    parent_state = egraph._state

    with egraph:
        egraph.close()

    assert egraph._state is parent_state
    assert not egraph._state_stack


def test_file_backed_errors_report_saved_file_line() -> None:
    egraph = EGraph(save_egglog_string=True)
    egraph.let("x", i64(1))
    egraph.let("y", i64(2))
    expected_line = len(egraph.as_egglog_string.splitlines()) + 1
    assert egraph._state.egglog_file_state is not None
    path = egraph._state.egglog_file_state.path

    with pytest.raises(EggSmolError) as exc_info:
        egraph.check(eq(i64(1)).to(i64(2)))

    error_text = exc_info.value.context
    assert exc_info.value.replayable_by_fail
    assert path in error_text
    assert f"In {expected_line}:" in error_text
    lines = egraph.as_egglog_string.splitlines()
    assert "(fail (check (= 1 2))) ; Check failed:" in lines
    assert "(check (= 1 2))" not in lines


def test_non_replayable_egglog_error_invalidates_saved_transcript() -> None:
    egraph = EGraph(save_egglog_string=True)
    command = egg_bindings.Check(
        egg_bindings.RustSpan(__name__, 0, 0),
        [
            egg_bindings.Eq(
                egg_bindings.RustSpan(__name__, 0, 0),
                egg_bindings.Lit(egg_bindings.RustSpan(__name__, 0, 0), egg_bindings.Int(1)),
                egg_bindings.Lit(egg_bindings.RustSpan(__name__, 0, 0), egg_bindings.Float(1.0)),
            )
        ],
    )

    with pytest.raises(EggSmolError) as exc_info:
        egraph._state.run_program(command)

    assert not exc_info.value.replayable_by_fail
    with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
        _ = egraph.as_egglog_string
    with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
        egraph.register(relation("after_non_replayable_failure")())


def _raise_after_partial_write(_: object) -> object:
    message = "callback failed"
    raise ValueError(message)


def test_non_egglog_failure_invalidates_saved_transcript() -> None:
    trigger = relation("transcript_failure_trigger", i64)
    done = relation("transcript_failure_done", i64)
    x = var("x", i64)
    failing_rules = ruleset(
        rule(trigger(x)).then(done(x), PyObject(_raise_after_partial_write)(PyObject(None))),
        name="transcript_failure_rules",
    )

    unrecorded = EGraph(trigger(i64(1)))
    with pytest.raises(ValueError, match="callback failed"):
        unrecorded.run(run(failing_rules))
    unrecorded.check(done(i64(1)))

    recorded = EGraph(trigger(i64(1)), save_egglog_string=True)
    with pytest.raises(ValueError, match="callback failed"):
        recorded.run(run(failing_rules))
    with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
        recorded.check(done(i64(1)))
    with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
        _ = recorded.as_egglog_string
    recorded.close()

    scoped = EGraph(trigger(i64(1)), save_egglog_string=True)
    parent_state = scoped._state
    with pytest.raises(ValueError, match="callback failed"), scoped:
        scoped.run(run(failing_rules))
    assert scoped._state is parent_state
    assert not scoped._state_stack
    with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
        scoped.check(done(i64(1)))
    scoped.close()


class _FaultingWriter:
    def __init__(self, file: TextIO, fail_on: Literal["write", "flush"]) -> None:
        self.file = file
        self.fail_on = fail_on

    @property
    def closed(self) -> bool:
        return self.file.closed

    def write(self, text: str) -> int:
        if self.fail_on == "write":
            message = "transcript write failed"
            raise OSError(message)
        return self.file.write(text)

    def flush(self) -> None:
        if self.fail_on == "flush":
            message = "transcript flush failed"
            raise OSError(message)
        self.file.flush()


@pytest.mark.parametrize("fail_on", ["write", "flush"])
@pytest.mark.parametrize("command_succeeds", [False, True], ids=["egglog-error", "success"])
def test_saved_transcript_io_failure_invalidates_transcript(
    fail_on: Literal["write", "flush"], *, command_succeeds: bool
) -> None:
    marker = relation(f"transcript_io_{fail_on}_{command_succeeds}", i64)
    egraph = EGraph(marker(i64(0)), save_egglog_string=True)
    assert egraph._state.egglog_file_state is not None
    file = egraph._state.egglog_file_state.file
    egraph._state.egglog_file_state.file = cast("TextIO", _FaultingWriter(file, fail_on))

    def execute_command() -> None:
        if command_succeeds:
            egraph.register(marker(i64(1)))
        else:
            egraph.check(eq(i64(1)).to(i64(2)))

    with pytest.raises(OSError, match=f"transcript {fail_on} failed"):
        execute_command()
    with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
        _ = egraph.as_egglog_string
    egraph.close()


def test_higher_order_callable_inference_does_not_mutate_ambient_ruleset() -> None:
    ambient = ruleset(name="hof-inference-ambient")
    initial_rules = tuple(ambient.__egg_ruleset__.rules)

    with set_current_ruleset(ambient):
        initial: Map[i64, f64] = Map[i64, f64].empty()
        expr = map_fold_kv(
            lambda result, key, value: result.insert(key, -value),
            initial,
            Map[i64, f64].empty().insert(i64(1), f64(2.0)),
        )
        _ = cast("RuntimeExpr", expr).__egg_decls__

    assert tuple(ambient.__egg_ruleset__.rules) == initial_rules


def test_set_current_ruleset_restores_nested_contexts() -> None:
    outer = ruleset(name="current-ruleset-outer")
    inner = ruleset(name="current-ruleset-inner")
    initial = get_current_ruleset()

    with set_current_ruleset(outer):
        assert get_current_ruleset() is outer
        with set_current_ruleset(inner):
            assert get_current_ruleset() is inner
        assert get_current_ruleset() is outer

    assert get_current_ruleset() is initial


def test_generated_names_are_fully_qualified() -> None:
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

    assert state.callable_ref_to_egg(FunctionRef(fn1))[0] == "pkg_one_make"
    assert state.callable_ref_to_egg(FunctionRef(fn2))[0] == "pkg_two_make"
    assert state.type_ref_to_egg(JustTypeRef(ret1)) == "pkg.one.Ret"
    assert state.type_ref_to_egg(JustTypeRef(ret2)) == "pkg.two.Ret"


def test_missing_function_lookup_does_not_reserve_generated_name() -> None:
    state = EGraph(save_egglog_string=True)._state
    ret = Ident("LookupRet", "pkg.lookup")
    fn = Ident("lookup_short_name", "pkg.lookup")
    state.__egg_decls__ |= Declarations(
        _classes={ret: ClassDecl()},
        _functions={fn: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret)))},
    )

    qualified_name = "pkg_lookup_lookup_short_name"
    assert list(state.possible_egglog_functions([qualified_name])) == []
    assert state.callable_ref_to_egg(FunctionRef(fn))[0] == qualified_name


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


def test_generated_callable_name_avoids_an_existing_cost_table() -> None:
    state = EGraph(save_egglog_string=True)._state
    state.__egg_decls__ |= cast("HasDeclarations", i64)
    ret = Ident("CostRet", "pkg.cost")
    fn = Ident("f", "pkg.cost")
    conflict = Ident("cost_table_pkg_cost_f")
    state.__egg_decls__ |= Declarations(
        _classes={ret: ClassDecl()},
        _functions={
            fn: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret))),
            conflict: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret))),
        },
    )

    assert state.create_cost_table(FunctionRef(fn)) == "cost_table_pkg_cost_f"
    assert state.callable_ref_to_egg(FunctionRef(conflict))[0] == "cost_table_pkg_cost_f_1"


def test_canonical_cost_table_rejects_an_incompatible_callable() -> None:
    state = EGraph(save_egglog_string=True)._state
    state.__egg_decls__ |= cast("HasDeclarations", i64)
    ret = Ident("CostRet", "pkg.cost")
    fn = Ident("f", "pkg.cost")
    conflict = Ident("raw_cost", "pkg.cost")
    state.__egg_decls__ |= Declarations(
        _classes={ret: ClassDecl()},
        _functions={
            fn: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret))),
            conflict: FunctionDecl(
                signature=FunctionSignature(return_type=TypeRefWithVars(ret)),
                egg_name="cost_table_pkg_cost_f",
            ),
        },
    )

    assert state.callable_ref_to_egg(FunctionRef(conflict))[0] == "cost_table_pkg_cost_f"
    with pytest.raises(ValueError, match="already used by an incompatible callable"):
        state.create_cost_table(FunctionRef(fn))


def test_canonical_cost_table_reuses_a_compatible_raw_table() -> None:
    state = EGraph(save_egglog_string=True)._state
    state.__egg_decls__ |= cast("HasDeclarations", i64)
    ret = Ident("CostRet", "pkg.cost")
    fn = Ident("f", "pkg.cost")
    raw_cost = Ident("raw_cost", "pkg.cost")
    state.__egg_decls__ |= Declarations(
        _classes={ret: ClassDecl()},
        _functions={
            fn: FunctionDecl(signature=FunctionSignature(return_type=TypeRefWithVars(ret))),
            raw_cost: FunctionDecl(
                signature=FunctionSignature(return_type=TypeRefWithVars(Ident.builtin("i64"))),
                egg_name="cost_table_pkg_cost_f",
            ),
        },
    )

    assert state.callable_ref_to_egg(FunctionRef(raw_cost))[0] == "cost_table_pkg_cost_f"
    assert state.create_cost_table(FunctionRef(fn)) == "cost_table_pkg_cost_f"
    assert state.cost_table_names[FunctionRef(fn)] == "cost_table_pkg_cost_f"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(None, id="bodyless-function"),
        pytest.param(
            TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(1)),
            id="eager-primitive",
        ),
    ],
)
@pytest.mark.parametrize("sort_first", [True, False], ids=["sort-first", "callable-first"])
def test_generated_names_share_the_backend_sort_and_callable_namespace(
    body: TypedExprDecl | None, *, sort_first: bool
) -> None:
    state = EGraph(save_egglog_string=True)._state
    state.__egg_decls__ |= cast("HasDeclarations", i64)
    sort_ident = Ident("Node")
    fn_ident = Ident("Node")
    sort_ref = JustTypeRef(sort_ident)
    fn_ref = FunctionRef(fn_ident)
    state.__egg_decls__ |= Declarations(
        _classes={sort_ident: ClassDecl()},
        _functions={
            fn_ident: FunctionDecl(
                signature=FunctionSignature(return_type=TypeRefWithVars(Ident.builtin("i64"))),
                body=body,
            )
        },
    )

    if sort_first:
        assert state.type_ref_to_egg(sort_ref) == "Node"
        assert state.callable_ref_to_egg(fn_ref)[0] == "Node_1"
    else:
        assert state.callable_ref_to_egg(fn_ref)[0] == "Node"
        assert state.type_ref_to_egg(sort_ref) == "Node_1"
