from __future__ import annotations

import gc
import pathlib
from typing import Literal, TextIO, cast

import pytest

import egglog.bindings as egg_bindings
import egglog.egraph as egraph_module
from egglog import (
    EggSmolError,
    EGraph,
    Map,
    PyObject,
    eq,
    f64,
    get_callable_args,
    i64,
    map_fold_kv,
    relation,
    rule,
    ruleset,
    run,
    set_current_ruleset,
    var,
)
from egglog.egraph import get_current_ruleset


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


def test_invalid_saved_transcript_inside_context_restores_scope() -> None:
    marker = relation("invalid_transcript_scope_marker", i64)
    egraph = EGraph(marker(i64(1)), save_egglog_string=True)
    parent_state = egraph._state
    original_error = None

    try:
        with pytest.raises(EggSmolError) as exc_info, egraph:  # noqa: PT012 -- exercise failure and rollback together
            active_state = egraph._state
            egraph.register(marker(i64(2)))
            try:
                egraph.check(eq(i64(1)).to(i64(2)))
            except EggSmolError as error:
                original_error = error
                raise

        assert exc_info.value is original_error
        assert egraph._state is parent_state
        assert not egraph._state_stack
        assert active_state.handle is None
        snapshot = egraph.freeze()
        assert snapshot.program is not None
        marker_calls = [
            node.kind.value
            for node in snapshot.program.nodes
            if node.kind is not None
            and node.kind.field == "call"
            and node.kind.value.func == "invalid_transcript_scope_marker"
        ]
        assert len(marker_calls) == 1
        value = snapshot.program.nodes[marker_calls[0].args[0]].kind.value.value
        assert value.field == "i64"
        assert value.value == 1
        with pytest.raises(RuntimeError, match="cannot be replayed"):
            _ = egraph.as_egglog_string
        with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
            egraph.register(marker(i64(3)))
        with pytest.raises(RuntimeError, match="partial effects cannot be replayed"):
            egraph.push()
        assert egraph._state is parent_state
        assert not egraph._state_stack
    finally:
        egraph._state.destroy()


def test_cleanup_failure_does_not_mask_propagating_base_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class BodyError(BaseException):
        pass

    def fail_cleanup(*args: object, **kwargs: object) -> None:
        message = "cleanup failed"
        raise RuntimeError(message)

    egraph = EGraph(save_egglog_string=True)
    parent_state = egraph._state

    def enter_and_fail() -> None:
        with egraph:
            egraph.close()
            monkeypatch.setattr(egraph_module, "call_with_current_trace", fail_cleanup)
            raise BodyError

    with pytest.raises(BodyError):
        enter_and_fail()

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
    initial_snapshot = ambient._snapshot({})

    with set_current_ruleset(ambient):
        initial: Map[i64, f64] = Map[i64, f64].empty()
        expr = map_fold_kv(
            lambda result, key, value: result.insert(key, -value),
            initial,
            Map[i64, f64].empty().insert(i64(1), f64(2.0)),
        )
        arguments = get_callable_args(expr)
        assert arguments is not None
        assert len(arguments) == 3

    assert ambient._snapshot({}) == initial_snapshot


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
