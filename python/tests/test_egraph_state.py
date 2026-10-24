from __future__ import annotations

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb

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
            # A relation's authored Callable is its RuntimeFunction presentation at runtime.
            and node.kind.value.func == marker.__egg_ref__.read().kind.value.name  # type: ignore[attr-defined]
        ]
        assert len(marker_calls) == 1
        value_kind = snapshot.program.nodes[marker_calls[0].args[0]].kind
        assert value_kind is not None
        assert value_kind.field == "primitive_value"
        value = value_kind.value.value
        assert value is not None
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


def test_closing_saved_transcript_preserves_propagating_base_exception() -> None:
    class BodyError(BaseException):
        pass

    original = BodyError("body sentinel")
    egraph = EGraph(save_egglog_string=True)
    parent_state = egraph._state

    with pytest.raises(BodyError) as caught, egraph:  # noqa: PT012 -- exercise body failure and rollback together
        egraph.close()
        raise original

    assert caught.value is original
    assert egraph._state is parent_state
    assert not egraph._state_stack


def test_saved_transcript_errors_preserve_command_location() -> None:
    egraph = EGraph(save_egglog_string=True)
    egraph.let("x", i64(1))
    egraph.let("y", i64(2))

    with pytest.raises(EggSmolError) as exc_info:
        egraph.check(eq(i64(1)).to(i64(2)))

    error = exc_info.value.__dict__["_egglog_proto_error"]
    assert error.code == pb.ErrorCode.CHECK_FAILED
    assert error.location == pb.CommandLocation(path=[0])
    assert str(exc_info.value) == error.message
    assert exc_info.value.args == (error.message,)
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        _ = egraph.as_egglog_string


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
    with pytest.raises(
        RuntimeError, match=r"^The saved transcript cannot be replayed after a failed engine operation$"
    ):
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


def test_higher_order_callable_inference_does_not_mutate_ambient_ruleset() -> None:
    ambient = ruleset(name="hof-inference-ambient")
    initial_snapshot = ambient._snapshot()

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

    assert ambient._snapshot() == initial_snapshot


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
