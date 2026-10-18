from __future__ import annotations

import threading

import cloudpickle
import pytest

from egglog import EGraph, PyObject, String, eq, i64, relation, rule, ruleset, run, var


def _raise_worker_error(caller_thread_id: int) -> object:
    if threading.get_ident() == caller_thread_id:
        message = "primitive did not run on a worker thread"
        raise AssertionError(message)
    message = "parallel primitive failed"
    raise ValueError(message)


@pytest.mark.parametrize("position", ["query", "action"])
def test_parallel_py_object_primitive_error_is_propagated(position: str):
    noise = relation("noise", i64)
    target = relation("target", PyObject, PyObject)
    hit = relation("hit", PyObject)
    egraph = EGraph(num_threads=2)
    # Cross Egglog's default 10,000-row cutoff so rule evaluation uses its worker pool.
    egraph.register(
        *(noise(i64(i)) for i in range(10_001)),
        target(PyObject(_raise_worker_error), PyObject(threading.get_ident())),
    )
    fn, arg, value = (var(name, PyObject) for name in ("fn", "arg", "value"))
    failing_rule = (
        rule(target(fn, arg), eq(value).to(fn(arg))).then(hit(arg))
        if position == "query"
        else rule(target(fn, arg)).then(hit(fn(arg)))
    )
    try:
        with pytest.raises(ValueError, match="parallel primitive failed"):
            egraph.run(run(ruleset(failing_rule)))

        # Draining the worker error leaves the same graph usable by later Python primitives.
        egraph.check(PyObject("recovered").to_string() == String("recovered"))
    finally:
        egraph._state.destroy()


def test_serialize_string(monkeypatch: pytest.MonkeyPatch):
    """Snapshot observation keeps PyObject bytes opaque instead of materializing a repr."""
    obj = "my object"
    obj_expr = PyObject(obj)
    value = obj_expr.__egg_ref__.read().kind.value.value
    assert value.field == "custom"
    payload_size = len(value.value.payload)
    member = relation("my_relation", PyObject)
    egraph = EGraph(member(obj_expr))

    def forbidden(_payload):
        pytest.fail("Snapshot observation materialized a PyObject")

    monkeypatch.setattr(cloudpickle, "loads", forbidden)
    try:
        serialized = egraph._serialize()
        ops = [node["op"] for node in serialized["nodes"].values()]
        assert f"PyObject(<{payload_size} opaque bytes>)" in ops
        assert repr(obj) not in ops
    finally:
        egraph._state.destroy()
