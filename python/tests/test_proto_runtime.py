# mypy: allow-empty-bodies
# Ellipsis bodies are DSL declarations or annotation-introspection fixtures.
"""Record-native authoring gates for the single public runtime migration."""

import gc
import math
import operator
import struct
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from fractions import Fraction
from functools import partial
from threading import Event
from time import sleep
from typing import ClassVar, Literal, TypeVar, cast, get_type_hints
from weakref import ref as weakref

import black
import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import (
    Action,
    Bool,
    BoolLike,
    ConvertError,
    EGraph,
    Expr,
    Fact,
    Map,
    MapLike,
    Pair,
    Ruleset,
    String,
    StringLike,
    Unit,
    UnstableFn,
    Vec,
    VecLike,
    bindings,
    birewrite,
    constant,
    convert,
    converter,
    eq,
    expr_action,
    expr_parts,
    f64,
    function,
    get_type_args,
    get_var_name,
    i64,
    join,
    method,
    ne,
    rewrite,
    rule,
    ruleset,
    runtime,
    set_,
    union,
    var,
    vars_,
)
from egglog._catalog import builtin_catalog
from egglog._program import Builder, Ref, StructuralView, pack
from egglog.builtins import ExprValueError, _container_items, _map_entries, _pair_value
from egglog.deconstruct import get_callable_args, get_callable_fn, get_constant_name, get_literal_value
from egglog.egraph import get_current_ruleset, set_current_ruleset, to_runtime_expr
from egglog.runtime import (
    _CLASS_CACHE,
    RuntimeClass,
    RuntimeExpr,
    RuntimeFunction,
    _author_call,
    _ClassContext,
    resolve_callable,
    resolve_type_annotation,
)
from egglog.type_constraint_solver import TypeConstraintError, infer_sort, substitute_sort


def _required_oneof[T](kind: object, field: str, payload_type: type[T]) -> T:
    """Require one encoded arm and its payload class at a test observation boundary."""
    assert isinstance(kind, Oneof), (field, kind)
    assert kind.field == field, (field, kind.field)
    value = kind.value
    assert isinstance(value, payload_type), (field, type(value), payload_type)
    return value


def _required_callable_args(x: object, fn: object = None) -> tuple[RuntimeExpr, ...]:
    """Require the existing deconstructor to match this known-call observation."""
    arguments = get_callable_args(x, fn)
    assert arguments is not None
    return arguments


def _required_runtime_function(value: object) -> RuntimeFunction:
    """Require the actual callable presentation without invoking or preparing it."""
    assert isinstance(value, RuntimeFunction)
    return value


def _required_python_bindings(declaration: pb.Declaration) -> pb.PythonBindings:
    """Require authored Python metadata at this positive observation boundary."""
    bindings = declaration.bindings
    assert bindings is not None
    python = bindings.python
    assert python is not None
    return python


def _required_authored_class(cls: object) -> tuple[RuntimeClass, _ClassContext]:
    """Require a real authored class and its retained context, without preparation."""
    assert isinstance(cls, RuntimeClass)
    context = cls.__egg_context__
    assert context is not None
    return cls, context


def test_pyobject_local_values_and_trusted_display(class_authoring_without_engine, monkeypatch):
    import cloudpickle  # noqa: PLC0415

    from egglog import PyObject, py_eval  # noqa: PLC0415
    from egglog.pretty import pretty_ref  # noqa: PLC0415

    original = {"x": [1]}
    value = PyObject(original)
    original["x"].append(2)
    assert value.value == {"x": [1]}
    restored = value.value
    restored["x"].append(3)
    assert value.value == {"x": [1]}
    assert PyObject(None).value is None
    match PyObject(None):
        case PyObject(None):
            pass
        case _:
            pytest.fail("PyObject(None) did not match its actual None value")
    assert get_literal_value(PyObject("test")) == "test"
    assert get_literal_value(PyObject.dict()) is None
    with pytest.raises(ExprValueError, match="PyObject literal"):
        _ = PyObject.dict().value

    class BadRepr:
        def __repr__(self) -> str:
            return "&&&"

    assert str(PyObject("hi")) == 'PyObject("hi")'
    assert repr(PyObject(BadRepr())) == 'PyObject(eval("&&&"))'
    expression = py_eval("x", {}, {"x": 1})
    assert '{"x": 1}' in str(expression)
    action = expr_action(value)
    assert str(action) == str(value)
    rewrite_value = rewrite(value).to(PyObject({"x": [2]}))  # type: ignore[type-var]
    assert str(value) in str(rewrite_value)

    packed = pack([value.__egg_ref__])

    def forbidden(_payload):
        pytest.fail("Opaque record inspection materialized a PyObject")

    monkeypatch.setattr(cloudpickle, "loads", forbidden)
    decoded = pb.Program.from_binary(packed.program.to_binary())
    detached = RuntimeExpr(Builder.from_program(decoded).publish().ref("nodes", packed.indices[0]))
    assert StructuralView(detached.__egg_ref__) == StructuralView(value.__egg_ref__)
    assert hash(detached) == hash(value)
    assert pack([detached.__egg_ref__]).program.to_binary() == packed.program.to_binary()
    with pytest.raises(NotImplementedError, match="custom"):
        pretty_ref(detached.__egg_ref__)


@pytest.mark.parametrize("annotation", [object, "PyObject", "PyObjectLike"])
def test_object_annotations_share_canonical_pyobject_and_box_once(annotation, class_authoring_without_engine):
    from egglog import PyObject, PyObjectLike  # noqa: PLC0415

    annotation = {"PyObject": PyObject, "PyObjectLike": PyObjectLike}.get(annotation, annotation)
    expected = resolve_type_annotation(PyObject)
    assert isinstance(expected, Ref)
    resolved = resolve_type_annotation(annotation)
    assert isinstance(resolved, Ref)
    assert StructuralView(resolved) == StructuralView(expected)

    class Holder(Expr):
        def __init__(self, value: annotation = None) -> None: ...  # type: ignore[valid-type]

        def echo(self, value: annotation) -> annotation: ...  # type: ignore[valid-type]

    missing = Holder()
    assert _required_callable_args(missing, Holder)[0].value is None
    calls = []

    class Payload:
        def __reduce__(self) -> tuple[type[dict[str, int]], tuple[list[tuple[str, int]]]]:
            calls.append("serialize")
            return dict, ([("value", 7)],)

    held = Holder(Payload())
    argument = _required_callable_args(held, Holder)[0]
    assert isinstance(argument, PyObject)
    assert argument.value == {"value": 7}
    assert calls == ["serialize"]
    supplied = Holder(argument)
    assert _required_callable_args(supplied, Holder)[0].__egg_ref__ == argument.__egg_ref__
    echoed = supplied.echo(argument)
    assert isinstance(echoed, PyObject)
    assert _required_callable_args(echoed)[1].__egg_ref__ == argument.__egg_ref__
    assert calls == ["serialize"]
    encoded = pack([held.__egg_ref__, echoed.__egg_ref__])  # type: ignore[attr-defined]
    decoded = Builder.from_program(pb.Program.from_binary(encoded.program.to_binary())).publish()
    assert StructuralView(decoded.ref("nodes", encoded.indices[0])) == StructuralView(held.__egg_ref__)
    assert StructuralView(decoded.ref("nodes", encoded.indices[1])) == StructuralView(echoed.__egg_ref__)  # type: ignore[attr-defined]
    with pytest.raises(ConvertError):
        Holder(i64(1))


def test_object_callable_annotations_keep_concrete_sorts(class_authoring_without_engine):
    from egglog import PyObject  # noqa: PLC0415

    function_sort = resolve_type_annotation(Callable[[object], object])
    assert isinstance(function_sort, Ref)
    signature = _required_oneof(function_sort.read().kind, "func", pb.FuncSort)
    input_sort = resolve_type_annotation(PyObject)
    assert isinstance(input_sort, Ref)
    assert StructuralView(function_sort.owner.ref("sorts", signature.params[0])) == StructuralView(input_sort)
    result_sort = resolve_type_annotation(PyObject)
    assert isinstance(result_sort, Ref)
    assert StructuralView(function_sort.owner.ref("sorts", signature.result)) == StructuralView(result_sort)
    for unsupported in (int, str, type(None)):
        with pytest.raises(TypeError, match="Unexpected type annotation"):
            resolve_type_annotation(unsupported)


def test_object_annotated_constructor_roundtrips_native_payload():
    from egglog import PyObject  # noqa: PLC0415

    class Holder(Expr):
        def __init__(self, value: object = None) -> None: ...

    payload = {"values": [1]}
    expression = Holder(payload)
    payload["values"].append(2)
    graph = EGraph(expression, Holder())
    try:
        restored = graph.extract(expression)
        value = _required_callable_args(restored, Holder)[0]
        assert isinstance(value, PyObject)
        assert value.value == {"values": [1]}
        graph.check(eq(expression).to(Holder(PyObject({"values": [1]}))))
        graph.check(eq(Holder()).to(Holder(PyObject(None))))
    finally:
        graph._state.destroy()


@pytest.mark.parametrize("corruption", ["family", "arity", "declaration", "arm", "children"])
def test_pyobject_materialization_rejects_malformed_shape(corruption, class_authoring_without_engine, monkeypatch):
    import cloudpickle  # noqa: PLC0415

    from egglog import PyObject  # noqa: PLC0415

    packed = pack([PyObject(None).__egg_ref__])
    program = pb.Program.from_binary(packed.program.to_binary())
    node = program.nodes[packed.indices[0]]
    sort = program.sorts[node.sort_id]
    family = _required_oneof(sort.kind, "family", pb.HostSort)
    primitive = _required_oneof(node.kind, "primitive_value", pb.PrimitiveValue)
    custom = _required_oneof(primitive.value, "custom", pb.CustomValue)
    if corruption == "family":
        family.name = "String"
    elif corruption == "arity":
        family.args.append(node.sort_id)
    elif corruption == "declaration":
        next(
            _required_oneof(d.kind, "host_sort_family", pb.HostSortFamily)
            for d in program.declarations
            if (d.kind.field == "host_sort_family" if d.kind is not None else pytest.fail("Missing declaration kind"))
        ).arity = 1
    elif corruption == "arm":
        primitive.value = Oneof[Literal["string"], str]("string", "not a pickle")
    else:
        custom.args.append(packed.indices[0])

    def forbidden(_payload):
        pytest.fail("Malformed Custom value reached pickle decoding")

    monkeypatch.setattr(cloudpickle, "loads", forbidden)
    expression = RuntimeExpr(Builder.from_program(program).publish().ref("nodes", packed.indices[0]))
    with pytest.raises(ValueError, match="PyObject value"):
        get_literal_value(expression)


@pytest.mark.parametrize("diagnostic", ["normal", "type-repr", "add-note"])
def test_pyobject_serialization_preserves_original_error(diagnostic, class_authoring_without_engine, monkeypatch):
    import cloudpickle  # noqa: PLC0415

    from egglog import PyObject  # noqa: PLC0415

    class HostileMeta(type):
        def __repr__(cls) -> str:
            if diagnostic == "type-repr":
                msg = "annotation formatting failed"
                raise RuntimeError(msg)
            return "Unpickleable"

    class Unpickleable(metaclass=HostileMeta):
        pass

    class OriginalError(ValueError):
        def add_note(self, note):
            if diagnostic == "add-note":
                msg = "annotation failed"
                raise RuntimeError(msg)
            super().add_note(note)

    error = OriginalError("serialization failed")

    def fail(_value):
        raise error

    monkeypatch.setattr(cloudpickle, "dumps", fail)
    with pytest.raises(OriginalError) as caught:
        PyObject(Unpickleable())
    assert caught.value is error
    assert getattr(error, "__notes__", []) == (
        ["Failed to pickle object of type Unpickleable"] if diagnostic == "normal" else []
    )


def test_pyobject_compatibility_adapter_and_consuming_calls(class_authoring_without_engine):
    import egglog  # noqa: PLC0415
    from egglog import PyObject, py_eval_fn  # noqa: PLC0415
    from egglog import builtins as egg_builtins  # noqa: PLC0415

    assert hasattr(egg_builtins, "PyObjectFunction")
    assert "PyObjectFunction" not in egg_builtins.__all__
    assert not hasattr(egglog, "PyObjectFunction")
    with pytest.warns(DeprecationWarning, match=r"use PyObject\(fn\) directly"):
        fn = py_eval_fn(int)
    assert get_literal_value(fn) is int
    for expression, target in (
        (fn(), PyObject.__call__),
        (fn(1, 2), PyObject.__call__),  # type: ignore[arg-type]
        (PyObject({}).dict_update(), PyObject.dict_update),
    ):
        arguments = get_callable_args(expression, target)
        assert arguments is not None
        assert get_callable_fn(expression) == target
        assert StructuralView(target(*arguments).__egg_ref__) == StructuralView(expression.__egg_ref__)
    with pytest.raises(TypeError):
        fn(value=1)  # type: ignore[call-arg]


def test_ne_uses_actual_generic_catalog_and_ordered_canonical_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Generic inequality authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    declaration = builtin_catalog().definitions["callable", "egglog.core.ne"]
    record = declaration.read()
    assert record.kind is not None
    assert isinstance(record.kind.value, pb.HostPrimitive)
    typing = record.kind.value.typing
    assert typing is not None
    assert isinstance(typing.value, pb.GenericSignature)
    signature = typing.value
    assert len(signature.type_params) == 1
    assert len(signature.inputs) == 2
    assert signature.inputs[0].sort == signature.inputs[1].sort
    assert declaration.owner.ref("sorts", signature.inputs[0].sort).read().kind == Oneof("var", 0)
    assert declaration.owner.ref("sorts", signature.output).read().kind == Oneof("family", pb.HostSort(name="Unit"))
    function = RuntimeFunction(declaration)
    first, second = i64(1), i64(2)
    expression = ne(first).to(second)
    authored = to_runtime_expr(expression)
    direct = function(first, second)
    assert direct is not None
    assert StructuralView(authored.__egg_ref__) == StructuralView(direct.__egg_ref__)
    node = authored.__egg_ref__.read()
    assert node.kind is not None
    assert isinstance(node.kind.value, pb.Call)
    assert node.kind.value.func == "egglog.core.ne"
    assert [StructuralView(authored.__egg_ref__.owner.ref("nodes", index)) for index in node.kind.value.args] == [
        StructuralView(to_runtime_expr(first).__egg_ref__),
        StructuralView(to_runtime_expr(second).__egg_ref__),
    ]
    packed = pack([authored.__egg_ref__])
    decoded = pb.Program.from_binary(packed.program.to_binary())
    copied = Builder.from_program(decoded).publish().ref("nodes", packed.indices[0])
    assert StructuralView(copied) == StructuralView(authored.__egg_ref__)
    assert str(RuntimeExpr(copied)) == "ne(i64(1)).to(i64(2))"
    with pytest.raises(TypeError):
        function(first)
    with pytest.raises(TypeError):
        function(first, second, first)
    with pytest.raises(ConvertError):
        function(first, String("2"))


def test_canonical_schedule_live_repr_and_generator_failure_without_engine(monkeypatch):
    from egglog import run, seq  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Schedule authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    events = []
    selected = ruleset(name="live")

    def deferred():
        events.append("first")
        yield rule(name="first").then()

    selected.register(deferred)
    schedule = (selected + run()).saturate() * 3
    assert events == []
    assert "first" in str(schedule)
    assert events == ["first"]
    assert "first" in repr(schedule)
    assert events == ["first"]
    selected.append(rule(name="later").then())
    assert "later" in str(schedule)
    assert str(seq()) == "seq()"
    assert "* 0" in str(run() * 0)
    assert schedule._command.read().kind.field == "saturate"
    assert schedule._command.read().kind.value.max_iterations == 3
    assert ".saturate()" in str(schedule)

    def failing():
        events.append("failure")
        yield rule(name="not-appended").then()
        msg = "schedule generator"
        raise ValueError(msg)

    selected.register(failing)
    with pytest.raises(ValueError, match="schedule generator"):
        str(schedule)
    assert "not-appended" not in str(schedule)
    assert events == ["first", "failure"]


def test_canonical_backoff_records_and_persistent_identity_without_engine(monkeypatch):
    from egglog import BackOff, back_off, run  # noqa: PLC0415
    from egglog.pretty import pretty_ref  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Scheduler authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    local = back_off()
    command = local.__egg_ref__.read()
    assert command.kind.field == "bind_scheduler"
    options = command.kind.value.scheduler.kind.value
    assert not options.has_field("match_limit")
    assert not options.has_field("ban_length")
    zero = back_off(match_limit=0, ban_length=0)
    encoded = pack(commands=[zero.__egg_ref__]).program.to_binary()
    copied = pb.Program.from_binary(encoded)
    assert copied.to_binary() == encoded
    copied_binding = _required_oneof(copied.commands[0].kind, "bind_scheduler", pb.BindScheduler)
    copied_scheduler = copied_binding.scheduler
    assert copied_scheduler is not None
    limits = _required_oneof(copied_scheduler.kind, "back_off", pb.BackOff)
    assert limits.has_field("match_limit")
    assert limits.match_limit == 0
    assert limits.has_field("ban_length")
    assert limits.ban_length == 0
    assert "match_limit=0" in str(zero)
    assert "ban_length=0" in repr(zero)
    assert "match_limit=" not in str(local)
    node_limited = zero.__egg_ref__.read()
    node_limited.kind.value.scheduler.kind.value.node_limit = 0
    builder = Builder()
    index = builder.add("commands", node_limited)
    assert "node_limit=0" in pretty_ref(builder.publish().ref("commands", index))
    largest = back_off(match_limit=2**64 - 1)
    assert largest.__egg_ref__.read().kind.value.scheduler.kind.value.match_limit == 2**64 - 1
    with pytest.raises(TypeError, match="BindScheduler"):
        BackOff(i64(0).__egg_ref__)
    unnamed = local.__egg_ref__.read()
    unnamed.kind.value.name = ""
    builder = Builder()
    index = builder.add("commands", unnamed)
    with pytest.raises(TypeError, match="BindScheduler"):
        BackOff(builder.publish().ref("commands", index))

    persistent = local.persistent()
    binding = persistent.__egg_ref__.read().kind.value
    assert binding.name != command.kind.value.name
    assert binding.scheduler == command.kind.value.scheduler
    assert not local._persistent
    assert persistent._persistent
    selected = ruleset(name="persistent-source")
    schedule = run(selected, scheduler=persistent)
    root = schedule._resolve(None)
    addition, execution = _required_oneof(root.read().kind, "repeat", pb.Repeat).body
    assert _required_oneof(addition.kind, "add_rules", pb.AddRules).rules == []
    assert _required_oneof(execution.kind, "run", pb.Run).scheduler == binding.name
    assert schedule._schedulers[binding.name] is persistent
    program = pack(commands=[persistent.__egg_ref__, root]).program
    deployment, execution = program.commands
    _required_oneof(deployment.kind, "bind_scheduler", pb.BindScheduler)
    _required_oneof(execution.kind, "repeat", pb.Repeat)
    assert pb.Program.from_binary(program.to_binary()) == program
    assert ".persistent()" in repr(persistent)
    assert f"scheduler={binding.name!r}" in str(schedule)


@pytest.mark.parametrize("option", ["match_limit", "ban_length"])
@pytest.mark.parametrize("value", [-1, 2**64])
def test_canonical_backoff_unsigned_limits_without_engine(monkeypatch, option, value):
    from egglog import back_off  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Scheduler validation accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    with pytest.raises(OverflowError, match=option):
        back_off(**{option: value})


def test_canonical_backoff_hoist_and_scope_topology_without_engine(monkeypatch):
    from egglog import back_off, run, seq  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Scheduler scope authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    scheduler = back_off(match_limit=1)
    name = scheduler.__egg_ref__.read().kind.value.name
    leaf = run(ruleset(name="scoped-source"), scheduler=scheduler)
    implicit = (leaf * 3)._resolve(None).read().kind.value
    outer = scheduler.scope(leaf * 3)._resolve(None).read().kind.value
    assert implicit.body[0].kind.field == outer.body[0].kind.field == "add_rules"
    outer_scope = outer.body[1].kind.value
    for block, body in ((implicit, implicit.body[1:]), (outer_scope, outer_scope.body)):
        assert block.times == 1
        binding, loop = body
        assert binding.kind.field == "bind_scheduler"
        assert loop.kind.field == "saturate"
        assert loop.kind.value.max_iterations == 3
        assert loop.kind.value.body[0].kind.value.scheduler == binding.kind.value.name
    assert implicit.body[1].kind.value.name == name
    assert outer_scope.body[0].kind.value.name != name
    inner_prelude = (scheduler.scope(leaf) * 3)._resolve(None).read().kind.value
    assert inner_prelude.body[0].kind.field == "add_rules"
    inner = inner_prelude.body[1]
    assert inner.kind.field == "saturate"
    assert inner.kind.value.max_iterations == 3
    inner_binding, inner_run = inner.kind.value.body[0].kind.value.body
    assert inner_run.kind.value.scheduler == inner_binding.kind.value.name
    assert inner_binding.kind.value.name not in {name, outer_scope.body[0].kind.value.name}

    siblings = seq(scheduler.scope(leaf), leaf)
    block = siblings._resolve(None).read().kind.value
    assert [command.kind.field for command in block.body[:2]] == ["add_rules", "add_rules"]
    assert block.body[2].kind.value.name == name  # Default hoist for the sibling only.
    scoped, unbound = block.body[3].kind.value.body
    scoped_binding, scoped_run = scoped.kind.value.body
    assert scoped_run.kind.value.scheduler == scoped_binding.kind.value.name != name
    assert unbound.kind.value.scheduler == name
    independent = seq(scheduler.scope(leaf), scheduler.scope(leaf))._resolve(None)
    independent_prelude = independent.read().kind.value
    assert [command.kind.field for command in independent_prelude.body[:2]] == ["add_rules", "add_rules"]
    scoped_names = [child.kind.value.body[0].kind.value.name for child in independent_prelude.body[2].kind.value.body]
    assert len(set(scoped_names)) == 2
    assert name not in scoped_names
    assert "bind_scheduler(" in str(siblings)
    assert "match_limit=1" in str(siblings)
    persistent = scheduler.persistent()
    persistent_name = persistent.__egg_ref__.read().kind.value.name
    persistent_leaf = run(ruleset(name="persistent-scoped-source"), scheduler=persistent)
    scoped_persistent = persistent.scope(persistent_leaf)
    assert not scoped_persistent._schedulers
    mixed = seq(scoped_persistent, persistent_leaf)
    assert mixed._schedulers[persistent_name] is persistent
    mixed_prelude = mixed._resolve(None).read().kind.value
    assert [command.kind.field for command in mixed_prelude.body[:2]] == ["add_rules", "add_rules"]
    explicit, retained = mixed_prelude.body[2].kind.value.body
    explicit_binding, explicit_run = explicit.kind.value.body
    assert explicit_run.kind.value.scheduler == explicit_binding.kind.value.name != persistent_name
    assert retained.kind.value.scheduler == persistent_name


def test_canonical_scheduler_until_queries_and_live_rulesets_without_engine(monkeypatch):
    from egglog import back_off, relation, run  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Scheduler query authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    events = []
    selected = ruleset(name="until-live")

    def deferred():
        events.append("first")
        yield rule(name="first").then()

    selected.register(deferred)
    stop = relation("scheduler_stop", i64)
    scheduler = back_off(match_limit=0)
    schedule = run(selected, stop(1), eq(i64(1)).to(i64(1)), scheduler=scheduler)  # type: ignore[arg-type]
    assert events == []
    root = schedule._resolve(None)
    assert events == ["first"]
    program = pack(commands=[root]).program
    assert pb.Program.from_binary(program.to_binary()) == program
    addition, binding, conditional = _required_oneof(program.commands[0].kind, "repeat", pb.Repeat).body
    assert _required_oneof(addition.kind, "add_rules", pb.AddRules).rules == [0]
    binding_data = _required_oneof(binding.kind, "bind_scheduler", pb.BindScheduler)
    conditional_data = _required_oneof(conditional.kind, "repeat", pb.Repeat)
    assert conditional_data.times == 1
    stop_call, stop_equality = (program.nodes[index] for index in conditional_data.until)
    _required_oneof(stop_call.kind, "call", pb.Call)
    _required_oneof(stop_equality.kind, "union", pb.Union)
    assert _required_oneof(conditional_data.body[0].kind, "run", pb.Run).scheduler == binding_data.name
    assert "until=[" in str(schedule)
    assert "eq(" in str(schedule)
    selected.append(rule(name="later").then())
    assert "later" in repr(schedule)

    def failing():
        events.append("failure")
        yield rule(name="not-appended").then()
        msg = "scheduler source generator"
        raise ValueError(msg)

    selected.register(failing)
    with pytest.raises(ValueError, match="scheduler source generator"):
        str(schedule)
    assert "not-appended" not in str(schedule)
    assert events == ["first", "failure"]


def test_canonical_rule_key_contextual_hash_without_engine(monkeypatch):
    from egglog._program import _rule_key  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Report key accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    first = rule(name="same").then(set_(constant("report-target", i64)).to(1))  # type: ignore[arg-type]
    second = rule(name="same").then(set_(constant("report-target", i64)).to(1))  # type: ignore[arg-type]
    left, right = _rule_key(first.__egg_ref__), _rule_key(second.__egg_ref__)
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1
    assert "report-target" in repr(left)
    with pytest.raises(TypeError, match="rule"):
        _rule_key(i64(1).__egg_ref__)
    with pytest.raises(TypeError):
        hash(first)
    shared = i64(1) + i64(2)
    shared_key = _rule_key(rule().then(shared, shared).__egg_ref__)
    copied_key = _rule_key(rule().then(i64(1) + 2, i64(1) + 2).__egg_ref__)
    assert shared_key == copied_key
    assert hash(shared_key) == hash(copied_key)
    nan = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    other_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]
    first_nan = _rule_key(rule().then(f64(nan)).__egg_ref__)
    equal_nan = _rule_key(rule().then(f64(nan)).__egg_ref__)
    assert first_nan == equal_nan
    assert hash(first_nan) == hash(equal_nan)
    assert first_nan != _rule_key(rule().then(f64(other_nan)).__egg_ref__)
    assert _rule_key(rule().then(f64(0.0)).__egg_ref__) == _rule_key(rule().then(f64(-0.0)).__egg_ref__)


def test_rule_template_comparison_and_live_rulesets_without_engine(monkeypatch):
    from egglog import ruleset  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Canonical rule authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    x = var("x", i64)
    target = constant("comparison_target", i64)
    fact = eq(x).to(i64(1))
    one = rule(fact, fact, name="one").then(set_(target).to(x))
    equivalent = rule(eq(x).to(i64(1)), eq(x).to(i64(1)), name="one").then(set_(target).to(x))
    assert one == equivalent
    assert one != rule(fact, name="one").then(set_(target).to(x))
    assert one != rule(fact, fact, name="two").then(set_(target).to(x))
    with pytest.raises(TypeError):
        hash(one)
    selected = ruleset(one, equivalent, name="selected")
    before = selected._snapshot()
    assert len(before.read().kind.value.rules) == 2
    selected.append(rule(fact, name="new").then(set_(target).to(2)))  # type: ignore[arg-type]
    after = selected._snapshot()
    assert len(after.read().kind.value.rules) == 3
    assert len(before.read().kind.value.rules) == 2
    combined = selected | ruleset(name="empty")
    combined_command = combined._snapshot().read()
    assert combined_command.kind is not None
    assert combined_command.kind.field == "repeat"
    assert [command.kind.field for command in combined_command.kind.value.body] == ["add_rules", "add_rules"]
    assert "rule(" in str(selected)


def test_rule_comparison_memo_retains_only_successful_canonical_keys(monkeypatch):
    from collections import Counter  # noqa: PLC0415

    from egglog import _program as program  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Rule comparison accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    first = rule().then(set_(constant("memo_target", i64)).to(1)).__egg_ref__  # type: ignore[arg-type]
    equal = rule().then(set_(constant("memo_target", i64)).to(1)).__egg_ref__  # type: ignore[arg-type]
    different = rule().then(set_(constant("memo_target", i64)).to(2)).__egg_ref__  # type: ignore[arg-type]
    builder = Builder()
    alias = builder.import_ref(first)
    imported = builder.publish().ref("rules", alias)
    original = program._rule_key
    calls: Counter[Ref] = Counter()
    failure = ValueError("key did not complete")
    fail = True

    def counted(reference):
        calls[reference] += 1
        if reference == equal and fail:
            raise failure
        return original(reference)

    monkeypatch.setattr(program, "_rule_key", counted)
    keys: dict[Ref, tuple[object, ...]] = {}
    with pytest.raises(ValueError, match="key did not complete") as caught:
        program.rule_equal(first, equal, keys=keys)
    assert caught.value is failure
    assert set(keys) == {first}
    fail = False
    assert program.rule_equal(first, equal, keys=keys)
    assert program.rule_equal(imported, equal, keys=keys)
    assert not program.rule_equal(first, different, keys=keys)
    assert calls == {first: 1, equal: 2, different: 1}
    assert set(keys) == {first, equal, different}
    assert first != equal  # Reusing comparison data does not merge occurrences.


def test_rule_key_default_and_lambda_use_domains_without_engine():
    from egglog._program import _rule_key  # noqa: PLC0415

    class Value(Expr):
        def __init__(self) -> None: ...

    runtime_value, _value_context = _required_authored_class(Value)
    value_sort = runtime_value.__egg_sort__
    assert value_sort is not None
    builder = Builder()
    sort = builder.import_ref(value_sort)
    empty = builder.add("nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())))
    default = RuntimeExpr(builder.publish().ref("nodes", empty))

    @function
    def choose(left: Value = default, right: Value = default) -> Value: ...  # type: ignore[assignment]

    template = rule().then(choose(Value(), Value()))
    packed = pack([template.__egg_ref__])
    equal = _rule_key(Builder.from_program(packed.program).publish().ref("rules", packed.indices[0]))
    key = _rule_key(template.__egg_ref__)
    assert key == equal
    assert hash(key) == hash(equal)
    declaration = next(
        d for d in packed.program.declarations if d.bindings and _required_python_bindings(d).views[0].params
    )
    view = _required_python_bindings(declaration).views[0]
    original = view.params[0].default_expr
    assert view.params[1].default_expr == original
    view.params[1].default_expr = len(packed.program.nodes)
    packed.program.nodes.append(packed.program.nodes[original])
    independent = _rule_key(Builder.from_program(packed.program).publish().ref("rules", packed.indices[0]))
    assert key != independent  # Defaults of one presentation are one allocation domain.
    assert len({key, equal, independent}) == 2

    def lambdas(*, shared_body, captures=False, same_node=False):
        builder = Builder()
        sort = builder.import_ref(value_sort)
        func = builder.add("sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(result=sort))))
        bodies = [
            builder.add("nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())))
            for _ in range(2)
        ]
        functions = [
            builder.add(
                "nodes",
                pb.Node(
                    sort_id=func,
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                        "primitive_value",
                        pb.PrimitiveValue(
                            value=Oneof[Literal["lambda"], pb.Lambda](
                                "lambda",
                                pb.Lambda(captures=[bodies[index]] if captures else [], body=bodies[index]),
                            )
                        ),
                    ),
                ),
            )
            for index in (0, 0 if shared_body else 1)
        ]
        if same_node:
            functions[1] = functions[0]
        root = builder.add(
            "rules",
            pb.Rule(
                kind=Oneof[Literal["rule"], pb.RuleBody](
                    "rule",
                    pb.RuleBody(head=[pb.Action(kind=Oneof[Literal["term"], int]("term", node)) for node in functions]),
                )
            ),
        )
        return _rule_key(builder.publish().ref("rules", root))

    separate = lambdas(shared_body=False)
    shared = lambdas(shared_body=True)
    aliased = lambdas(shared_body=True, same_node=True)
    assert separate == shared == aliased  # Each deferred body has its own use domain.
    assert hash(separate) == hash(shared) == hash(aliased)
    assert lambdas(shared_body=True, captures=True) != lambdas(shared_body=False, captures=True)


@pytest.mark.parametrize("allocating", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_rule_key_shared_dag_work_is_bounded(monkeypatch, allocating, bidirectional):
    from egglog import _program  # noqa: PLC0415
    from egglog._program import _rule_key  # noqa: PLC0415

    depth = 14
    if allocating:

        class Value(Expr):
            def __init__(self, left: "Value", right: "Value") -> None: ...

        runtime_value, _value_context = _required_authored_class(Value)
        value_sort = runtime_value.__egg_sort__
        assert value_sort is not None
        builder = Builder()
        sort = builder.import_ref(value_sort)
        empty = builder.add("nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())))
        left = RuntimeExpr(builder.publish().ref("nodes", empty))
    else:
        left = i64(1)
    for _ in range(depth):
        left = Value(left, left) if allocating else left + left  # type: ignore[arg-type, operator]
    if allocating:
        parent = Value(left, left)  # type: ignore[arg-type]
        shared = _rule_key(rule().then(parent, parent).__egg_ref__)
        duplicated = _rule_key(rule().then(Value(left, left), Value(left, left)).__egg_ref__)  # type: ignore[arg-type]
        assert shared == duplicated
        assert hash(shared) == hash(duplicated)
    first = rule().then(left).__egg_ref__
    if bidirectional:
        expression = cast("Expr", left)
        first = birewrite(expression).to(expression).__egg_ref__
    packed = pack([first])
    second = Builder.from_program(packed.program).publish().ref("rules", packed.indices[0])
    original = _program._structural_parts
    visits = 0

    def counted(reference, *, identity=True):
        nonlocal visits
        visits += 1
        assert visits <= 16 * (depth + 20), "Comparison/hash expanded the shared expression DAG"
        return original(reference, identity=identity)

    monkeypatch.setattr(_program, "_structural_parts", counted)
    assert _program.rule_equal(first, second)
    assert hash(_rule_key(first)) == hash(_rule_key(second))


def test_rule_template_union_contexts_and_occurrence_identity(monkeypatch):
    from egglog import RewriteOrRule, _program  # noqa: PLC0415
    from egglog._program import _rule_key  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Template comparison accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)

    class Box(Expr):
        def __init__(self, value: i64) -> None: ...

    value = Box(0)  # type: ignore[arg-type]
    runtime_box, _box_context = _required_authored_class(Box)
    box_sort = runtime_box.__egg_sort__
    assert box_sort is not None

    def authored(*, shared, swap=False, empty=False):
        builder = Builder()
        sort = builder.import_ref(box_sort)
        member = builder.import_ref(value.__egg_ref__)
        nodes = [
            builder.add(
                "nodes",
                pb.Node(
                    sort_id=sort,
                    kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[] if empty else [member])),
                ),
            )
            for _ in range(2)
        ]
        head = [nodes[int(swap)], nodes[int(swap)] if shared else nodes[not swap]]
        index = builder.add(
            "rules",
            pb.Rule(
                kind=Oneof[Literal["rule"], pb.RuleBody](
                    "rule",
                    pb.RuleBody(
                        query=[] if empty else nodes,
                        head=[pb.Action(kind=Oneof[Literal["term"], int]("term", node)) for node in head],
                    ),
                ),
                eval_mode=pb.RuleEvalMode.SEMINAIVE,
            ),
        )
        return RewriteOrRule(builder.publish().ref("rules", index))

    left, right = authored(shared=True), authored(shared=True, swap=True)
    assert left == right  # Query visits must not seed the action allocation bijection.
    assert left.__egg_ref__ != right.__egg_ref__
    assert StructuralView(left.__egg_ref__) != StructuralView(right.__egg_ref__)
    assert authored(shared=True, empty=True) == authored(shared=True, empty=True)
    shared, independent = authored(shared=True, empty=True), authored(shared=False, empty=True)
    assert shared != independent
    assert not _program.rule_equal(shared.__egg_ref__, independent.__egg_ref__, keys={})
    assert hash(_rule_key(left.__egg_ref__)) == hash(_rule_key(right.__egg_ref__))
    assert len({_rule_key(left.__egg_ref__), _rule_key(right.__egg_ref__)}) == 1
    assert (
        len({
            _rule_key(authored(shared=True, empty=True).__egg_ref__),
            _rule_key(authored(shared=False, empty=True).__egg_ref__),
        })
        == 2
    )


def test_rule_template_comparison_uses_only_referenced_declarations(monkeypatch):
    from egglog import RewriteOrRule, _program  # noqa: PLC0415
    from egglog._program import _rule_key  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Rule closure comparison accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    first = rule().then(set_(constant("independent_target", i64)).to(1))  # type: ignore[arg-type]
    second = rule().then(set_(constant("independent_target", i64)).to(1))  # type: ignore[arg-type]
    assert first == second
    assert hash(_rule_key(first.__egg_ref__)) == hash(_rule_key(second.__egg_ref__))
    packed = pack(roots=[second.__egg_ref__])
    program = packed.program
    target = next(
        _required_oneof(declaration.kind, "function", pb.Function)
        for declaration in program.declarations
        if (
            declaration.kind.field == "function"
            if declaration.kind is not None
            else pytest.fail("Missing declaration kind")
        )
    )
    value = next(
        index
        for index, node in enumerate(program.nodes)
        if (node.kind.field == "primitive_value" if node.kind is not None else pytest.fail("Missing node kind"))
    )
    target.merge = value
    modified = RewriteOrRule(Builder.from_program(program).publish().ref("rules", packed.indices[0]))
    assert first != modified  # The command shape alone is identical.
    assert not _program.rule_equal(first.__egg_ref__, modified.__egg_ref__, keys={})
    assert len({_rule_key(first.__egg_ref__), _rule_key(modified.__egg_ref__)}) == 2
    builder = Builder()
    root = builder.import_ref(first.__egg_ref__)
    builder.import_ref(constant("unrelated", i64).__egg_ref__)
    assert first == RewriteOrRule(builder.publish().ref("rules", root))


def test_deferred_rulesets_keep_live_scope_order_and_failure_prefix(monkeypatch):
    from egglog import ruleset  # noqa: PLC0415
    from egglog.egraph import get_current_ruleset  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Deferred authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    selected = ruleset()
    events = []

    def older(value: "Late"):
        events.append(("older", get_current_ruleset()))
        return [rule(value, name="older").then()]

    def failing():
        events.append(("failing", get_current_ruleset()))
        selected.append(rule(name="side-effect").then())
        yield rule(name="not-appended").then()
        msg = "generator marker"
        raise ValueError(msg)

    selected.register(older)
    selected.register(failing)

    class Late(Expr):
        def __init__(self) -> None: ...

    with pytest.raises(ValueError, match="generator marker"):
        selected._snapshot()
    assert events == [("failing", selected)]
    assert get_current_ruleset() is None
    snapshot = selected._snapshot()
    assert events == [("failing", selected), ("older", selected)]
    assert [snapshot.owner.ref("rules", i).read().name for i in snapshot.read().kind.value.rules] == [
        "side-effect",
        "older",
    ]
    assert not selected.deferred_rule_gens


@pytest.mark.parametrize(
    ("name", "egg_name", "expected"),
    [
        ("x", None, "x"),
        ("shown", "backend", "backend"),
        ("shown", "x", "x"),
        ("shown", "_x", "_x"),
        ("shown", "0", "0"),
        ("shown", "@x", "@x"),
        ("shown", "_", "_"),
        ("x", "", "x"),
        ("", None, "_"),
        ("", "", "_"),
    ],
)
def test_direct_variables_are_canonical_without_an_engine(monkeypatch, name, egg_name, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("Variable authoring accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)

    class VariableBox(Expr):
        def __init__(self, value: i64) -> None: ...

    for sort in (i64, VariableBox, Vec[i64]):
        first, second = var(name, sort, egg_name=egg_name), var(name, sort, egg_name=egg_name)
        assert first.__egg_ref__ != second.__egg_ref__
        assert str(first) == expected
        assert get_var_name(first) == expected
        for expression in (first, second):
            record = expression.__egg_ref__.read()
            assert record.kind == Oneof[Literal["var"], str]("var", expected)
            assert StructuralView(expression.__egg_ref__.owner.ref("sorts", record.sort_id)) == StructuralView(
                sort.__egg_sort__  # type: ignore[union-attr]
            )
        packed = pack([first.__egg_ref__, second.__egg_ref__])
        decoded = pb.Program.from_binary(packed.program.to_binary())
        owner = Builder.from_program(decoded).publish()
        assert pack([owner.ref("nodes", index) for index in packed.indices]).program.to_binary() == decoded.to_binary()
        for index in packed.indices:
            restored = RuntimeExpr(owner.ref("nodes", index))
            assert str(restored) == expected
            assert get_var_name(restored) == expected
        detached = first.__egg_ref__.read()
        detached.kind = Oneof[Literal["var"], str]("var", "changed")
        assert str(first) == expected
    assert [str(value) for value in vars_("x y", i64)] == ["x", "y"]
    with pytest.raises(TypeError, match="concrete"):
        var("x", Vec)
    with pytest.raises(TypeError, match="concrete"):
        var("x", TypeVar("T"))  # type: ignore[arg-type]


@pytest.mark.parametrize("operation", [operator.eq, operator.ne])
def test_equality_dispatch_propagates_real_member_preparation_errors(operation):
    marker = AttributeError("real equality preparation failure")
    calls = []

    class Broken(Expr):
        def __eq__(self, other: "Broken") -> i64:  # type: ignore[override]
            calls.append("body")
            raise marker

    value = var("value", Broken)
    with pytest.raises(AttributeError, match="real equality preparation failure") as caught:
        operation(value, value)
    assert caught.value is marker
    assert calls == ["body"]


def test_numeric_planning_propagates_real_member_preparation_errors(class_authoring_without_engine):
    marker = AttributeError("real numeric preparation failure")
    calls = []

    class Broken(Expr):
        def __sub__(self, other: "Broken") -> i64:
            calls.append("body")
            raise marker

    value = var("value", Broken)
    with pytest.raises(AttributeError, match="real numeric preparation failure") as caught:
        value - value
    assert caught.value is marker
    assert calls == ["body"]


def test_numeric_exact_signature_does_not_prepare_an_unrelated_target(class_authoring_without_engine):
    calls = []

    class Unrelated(Expr):
        def __sub__(self, other: "Unrelated") -> i64:
            calls.append("body")
            msg = "Unrelated operator body was prepared"
            raise AssertionError(msg)

    converter(i64, Unrelated, lambda _: pytest.fail("Unselected converter ran"), cost=0)
    assert expr_parts(i64(5) - i64(2)) == expr_parts(i64.__sub__(i64(5), i64(2)))
    assert calls == []
    runtime_unrelated, _unrelated_context = _required_authored_class(Unrelated)
    assert runtime_unrelated.__egg_pending__ is not None


def test_numeric_mutating_member_keeps_its_receiver_and_aliases(class_authoring_without_engine):
    class Mutable(Expr):
        def __init__(self, value: i64) -> None: ...

        @method(mutates_self=True)
        def __add__(self, other: "Mutable") -> None: ...

    class Alternative(Expr):
        def __init__(self) -> None: ...
        def __add__(self, other: "Alternative") -> "Alternative": ...

    converter(Mutable, Alternative, lambda _: pytest.fail("Mutating receiver was converted"), cost=0)
    left, right = Mutable(1), Mutable(2)  # type: ignore[arg-type]
    alias, before = left, copy(left)
    right_ref = right.__egg_ref__
    assert left + right is None
    assert alias is left
    assert left.__egg_ref__ != before.__egg_ref__
    assert right.__egg_ref__ is right_ref
    assert get_callable_args(left, Mutable.__add__) == (to_runtime_expr(before), to_runtime_expr(right))


@pytest.mark.parametrize("strip_metadata", [False, True])
def test_detached_equality_uses_predicates_without_borrowing_members(class_authoring_without_engine, strip_metadata):
    class Shadow(Expr, egg_sort="i64"):
        def __init__(self, value: i64) -> None: ...

        @method(preserve=True)
        def __eq__(self, other) -> object:  # type: ignore[override]
            pytest.fail("Detached equality recovered a Python-only hook")

        @method(preserve=True)
        def __ne__(self, other) -> object:  # type: ignore[override]
            pytest.fail("Detached inequality recovered a Python-only hook")

    packed = pack([Shadow(3).__egg_ref__])  # type: ignore[arg-type]
    program = pb.Program.from_binary(packed.program.to_binary())
    if strip_metadata:
        for declaration in program.declarations:
            declaration.bindings = None
            assert declaration.kind is not None
            if declaration.kind.field == "eq_sort":
                declaration.kind.value.bindings = None
    detached = RuntimeExpr(Builder.from_program(program).publish().ref("nodes", packed.indices[0]))
    other = RuntimeExpr(detached.__egg_ref__)
    equality, inequality = detached == other, detached != other
    assert isinstance(equality, Fact)
    assert isinstance(inequality, Unit)
    assert equality
    assert expr_parts(inequality) == expr_parts(ne(detached).to(other))
    assert equality.__egg_ref__.read().kind.field == "union"
    assert inequality.__egg_ref__.read().kind.value.func == "egglog.core.ne"
    for name in ("__eq__", "__ne__", "__add__"):
        with pytest.raises(AttributeError, match="canonical Python member"):
            runtime._get_expr_member(detached, name, method_only=True)


@pytest.mark.parametrize("invalid", [None, False, 0, [], b"", object()])
def test_variable_names_and_aliases_validate_before_fallback(invalid):
    with pytest.raises(TypeError, match="Variable names must be strings"):
        var(invalid, i64, egg_name="valid")
    if invalid is not None:
        with pytest.raises(TypeError, match="Variable aliases must be strings"):
            var("valid", i64, egg_name=invalid)


def test_canonical_variable_authoring_is_thread_local(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Variable authoring accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: var("shown", i64, egg_name="shared"), range(16)))
    assert len({value.__egg_ref__ for value in values}) == 16
    assert all(StructuralView(value.__egg_ref__) == StructuralView(values[0].__egg_ref__) for value in values)
    detached = values[0].__egg_ref__.read()
    detached.kind = Oneof[Literal["var"], str]("var", "changed")
    assert [get_var_name(value) for value in values] == ["shared"] * 16


@pytest.mark.parametrize("bidirectional", [False, True])
def test_direct_rewrite_owns_records_without_an_engine(monkeypatch, bidirectional):
    from egglog import RewriteOrRule  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Rewrite authoring accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)

    class RewriteBox(Expr):
        def __init__(self, value: i64) -> None: ...

    lhs, rhs = RewriteBox(1), RewriteBox(2)  # type: ignore[arg-type]
    condition = eq(i64(3)).to(i64(3))
    builder = birewrite(lhs) if bidirectional else rewrite(lhs, subsume=True)
    kind = "birewrite" if bidirectional else "rewrite"
    assert str(builder) == f"{kind}(RewriteBox(1))"
    authored = builder.to(rhs, condition, i64(4), condition)
    assert authored.__egg_ref__.role == "rules"
    record = authored.__egg_ref__.read()
    assert record.eval_mode == pb.RuleEvalMode.SEMINAIVE
    assert not record.has_field("name")
    assert record.kind.field == kind
    body = record.kind.value
    if not bidirectional:
        assert body.subsume
    assert authored.__egg_ref__.owner.ref("nodes", body.lhs) == lhs.__egg_ref__
    assert authored.__egg_ref__.owner.ref("nodes", body.rhs) == rhs.__egg_ref__
    assert authored.__egg_ref__.owner.ref("nodes", body.conditions[0]) == condition.__egg_ref__
    assert body.conditions[0] == body.conditions[2]
    options = "" if bidirectional else ", subsume=True"
    expected = f"{kind}(RewriteBox(1){options}).to(RewriteBox(2), eq(i64(3)).to(i64(3)), i64(4), eq(i64(3)).to(i64(3)))"
    assert str(authored) == repr(authored) == expected
    with pytest.raises(ConvertError):
        rewrite(lhs).to(i64(2))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="destinations must be Rulesets"):
        rewrite(lhs, ruleset=object()).to(rhs)  # type: ignore[arg-type]
    condition_action = rule().then(condition)
    head = condition_action.__egg_ref__.read().kind.value.head
    assert len(head) == 1
    assert head[0].kind.field == "term"
    assert condition_action.__egg_ref__.owner.ref("nodes", head[0].kind.value) == condition.__egg_ref__
    lhs.__egg_ref__ = RewriteBox(9).__egg_ref__  # type: ignore[arg-type]
    rhs.__egg_ref__ = RewriteBox(10).__egg_ref__  # type: ignore[arg-type]
    body.conditions.clear()
    assert str(authored) == expected
    packed = pack([authored.__egg_ref__])
    assert len(packed.program.rules) == 1
    assert not packed.program.commands
    decoded = pb.Program.from_binary(packed.program.to_binary())
    owner = Builder.from_program(decoded).publish()
    restored = RewriteOrRule(owner.ref("rules", packed.indices[0]))
    assert str(restored) == expected
    assert restored == authored
    assert pack([restored.__egg_ref__]).program.to_binary() == decoded.to_binary()


def test_direct_birewrite_preserves_conversion_order_and_destination_on_failure():
    class Box(Expr):
        def __init__(self, value: i64) -> None: ...

    class Source:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

    events: list[Ruleset | None] = []

    def convert_source(value: Source) -> Box:
        events.append(get_current_ruleset())
        if value.fail:
            msg = "birewrite conversion marker"
            raise ValueError(msg)
        return Box(i64(2))

    converter(Source, Box, convert_source)
    destination, ambient = ruleset(), ruleset()
    original_destination = destination.__egg_ref__
    original_ambient = ambient.__egg_ref__
    lhs, rhs = Box(i64(1)), cast("Box", Source())
    with set_current_ruleset(ambient):
        with pytest.raises(TypeError, match="Expected a RuntimeExpr"):
            birewrite(cast("Box", object()), destination).to(rhs)
        with pytest.raises(TypeError, match="Expected a RuntimeExpr"):
            birewrite(lhs, destination).to(rhs, cast("Expr", object()))
        assert not events
        with pytest.raises(ValueError, match="birewrite conversion marker"):
            birewrite(lhs, destination).to(cast("Box", Source(fail=True)))
        assert destination.__egg_ref__ == original_destination
        with pytest.raises(ConvertError):
            birewrite(lhs, destination).to(cast("Box", i64(2)))
        assert destination.__egg_ref__ == original_destination
        unattached = birewrite(lhs).to(rhs)
        authored = birewrite(lhs, destination).to(rhs)
    assert events == [ambient, ambient, ambient]
    assert ambient.__egg_ref__ == original_ambient
    assert get_current_ruleset() is None
    record = destination.__egg_ref__.read()
    assert isinstance(record, pb.Command)
    assert record.kind is not None
    assert isinstance(record.kind.value, pb.AddRules)
    assert [destination.__egg_ref__.owner.ref("rules", index) for index in record.kind.value.rules] == [
        authored.__egg_ref__
    ]
    assert unattached == authored
    with pytest.raises(TypeError, match="destinations must be Rulesets"):
        birewrite(lhs, cast("Ruleset", object())).to(rhs)
    assert events == [ambient, ambient, ambient, None]


def test_direct_birewrite_keeps_occurrences_but_compares_structural_keys():
    from egglog._program import _rule_key  # noqa: PLC0415

    class Box(Expr):
        def __init__(self, value: i64) -> None: ...

    lhs, rhs = Box(i64(1)), Box(i64(2))
    first, second = birewrite(lhs).to(rhs), birewrite(lhs).to(rhs)
    assert first.__egg_ref__ != second.__egg_ref__
    assert first == second
    assert first != rewrite(lhs).to(rhs)
    assert first != birewrite(rhs).to(lhs)
    assert first != birewrite(lhs).to(rhs, eq(i64(1)).to(i64(1)))
    assert _rule_key(first.__egg_ref__) == _rule_key(second.__egg_ref__)
    assert len({_rule_key(first.__egg_ref__), _rule_key(second.__egg_ref__)}) == 1
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)
    selected = ruleset(first, first, second)
    packed = pack(commands=[selected.__egg_ref__])
    record = packed.program.commands[0]
    assert record.kind is not None
    assert isinstance(record.kind.value, pb.AddRules)
    occurrences = record.kind.value.rules
    assert len(occurrences) == 3
    assert occurrences[0] == occurrences[1] != occurrences[2]
    assert len(packed.program.rules) == 2


def test_birewrite_comparison_isolates_directional_union_allocations():
    from egglog._program import _rule_key  # noqa: PLC0415

    class Box(Expr):
        def __init__(self, value: i64) -> None: ...

    def authored(*, shared_left=True, shared_right=True, shared_across=False, condition_alias=False):
        builder = Builder()
        sort_ref = resolve_type_annotation(Box)
        assert isinstance(sort_ref, Ref)
        sort = builder.import_ref(sort_ref)
        member = builder.import_ref(to_runtime_expr(Box(i64(0))).__egg_ref__)
        nodes = [
            builder.add(
                "nodes",
                pb.Node(sort_id=sort, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[member]))),
            )
            for _ in range(5)
        ]
        left, right = nodes[0], nodes[0 if shared_across else 1]
        sides = [
            builder.add(
                "nodes",
                pb.Node(sort_id=sort, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=members))),
            )
            for members in ([left, left if shared_left else nodes[2]], [right, right if shared_right else nodes[3]])
        ]
        owner = builder.publish()
        lhs, rhs, condition = [
            cast("Box", RuntimeExpr(owner.ref("nodes", index)))
            for index in [*sides, nodes[0 if condition_alias else 4]]
        ]
        return birewrite(lhs).to(rhs, condition)

    separate, shared = authored(), authored(shared_across=True, condition_alias=True)
    assert separate == shared  # Sharing across directions/queries is not action allocation identity.
    assert hash(_rule_key(separate.__egg_ref__)) == hash(_rule_key(shared.__egg_ref__))
    assert separate != authored(shared_left=False)
    assert separate != authored(shared_right=False)


def test_direct_rule_reuses_query_variable_in_inline_set(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Rule variable authoring accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)

    @function
    def row(key: i64) -> i64: ...

    variable = var("x", i64)
    query = eq(row(variable)).to(variable)
    authored = rule(query).then(set_(row(variable)).to(variable))
    packed = pack([authored.__egg_ref__]).program
    variables = [
        index
        for index, node in enumerate(packed.nodes)
        if (node.kind.field == "var" if node.kind is not None else pytest.fail("Missing node kind"))
    ]
    assert len(variables) == 1
    index = variables[0]
    assert _required_oneof(packed.nodes[index].kind, "var", str) == "x"
    rule_record = _required_oneof(packed.rules[0].kind, "rule", pb.RuleBody)
    members = _required_oneof(packed.nodes[rule_record.query[0]].kind, "union", pb.Union).members
    assert members[1] == index
    assert _required_oneof(packed.nodes[members[0]].kind, "call", pb.Call).args == [index]
    assignment_record = _required_oneof(rule_record.head[0].kind, "set", pb.Set)
    assert assignment_record.target is not None
    assert assignment_record.target.args == [index]
    assert assignment_record.value == index
    assert str(authored) == "rule(eq(row(x)).to(x)).then(set_(row(x)).to(x))"


def test_direct_rule_relocates_inline_actions_without_an_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Rule authoring accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)

    class RuleBox(Expr):
        def __init__(self, value: i64) -> None: ...

    @function
    def rule_slot() -> i64: ...

    left, right = RuleBox(1), RuleBox(2)  # type: ignore[arg-type]
    shared, distinct = eq(left).to(right), eq(left).to(right)
    builder = Builder()
    builder.import_ref(i64(99).__egg_ref__)
    term = builder.import_ref(shared.__egg_ref__)
    assert term == 1  # Deliberately different from this query's first packed node.
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["term"], int]("term", term))
            )
        ),
    )
    shared_action = Action(builder.publish().ref("commands", command))
    assignment = set_(rule_slot()).to(7)  # type: ignore[arg-type]
    authored = rule(shared, distinct, name="direct", eval_mode="naive", no_decomp=True).then(
        assignment, shared_action, shared_action, left
    )
    record = authored.__egg_ref__.read()
    assert record.name == "direct"
    assert record.has_field("name")
    assert record.eval_mode == pb.RuleEvalMode.NAIVE
    assert record.no_decomp
    assert [action.kind.field for action in record.kind.value.head] == ["set", "term", "term", "term"]
    packed = pack([authored.__egg_ref__])
    assert not packed.program.commands
    body = _required_oneof(packed.program.rules[packed.indices[0]].kind, "rule", pb.RuleBody)
    assert body.query[0] != body.query[1]  # Distinct equal Union occurrences stay distinct.
    assert (
        _required_oneof(body.head[1].kind, "term", int)
        == _required_oneof(body.head[2].kind, "term", int)
        == body.query[0]
    )
    assignment_record = _required_oneof(body.head[0].kind, "set", pb.Set)
    assert assignment_record.target is not None
    assert assignment_record.target.args == []
    scalar_value = _required_oneof(
        packed.program.nodes[assignment_record.value].kind, "primitive_value", pb.PrimitiveValue
    )
    assert _required_oneof(scalar_value.value, "i64", int) == 7
    assert any(
        (declaration.kind.field == "function" and declaration.kind.value.name == assignment_record.target.func)
        if declaration.kind is not None
        else pytest.fail("Missing declaration kind")
        for declaration in packed.program.declarations
    )
    expected = (
        "rule(eq(RuleBox(1)).to(RuleBox(2)), eq(RuleBox(1)).to(RuleBox(2)), "
        "name='direct', eval_mode='naive', no_decomp=True).then(\n"
        "    set_(rule_slot()).to(i64(7)), union(RuleBox(1)).with_(RuleBox(2)), "
        "union(RuleBox(1)).with_(RuleBox(2)), RuleBox(1)\n"
        ")"
    )
    assert str(authored) == repr(authored) == expected
    record.kind.value.head.clear()
    left.__egg_ref__ = RuleBox(9).__egg_ref__  # type: ignore[arg-type]
    assert str(authored) == expected
    decoded = pb.Program.from_binary(packed.program.to_binary())
    owner = Builder.from_program(decoded).publish()
    assert pack([owner.ref("rules", packed.indices[0])]).program.to_binary() == decoded.to_binary()


def test_set_local_protocols_preserve_symbolic_order_and_empty_distinction(set_runtime):
    Set, integer = (set_runtime[name] for name in ("Set", "i64"))
    first, second = integer(2), integer(1)
    expression = Set(first, second, first)
    assert isinstance(expression, Set[integer])
    assert expression.value == {first, second}
    assert list(expression) == [first, second]
    assert len(expression) == 2
    assert first in expression
    assert integer(3) not in expression
    assert [arg.__egg_ref__ for arg in _required_callable_args(expression)] == [
        first.__egg_ref__,
        second.__egg_ref__,
        first.__egg_ref__,
    ]
    assert len(Set(integer(1) + 1, integer(2))) == 2
    assert Set[integer]().value == set()
    with pytest.raises(set_runtime["ExprValueError"]):
        _ = Set[integer].empty().value
    with pytest.warns(DeprecationWarning, match="use .value"):
        assert expression.eval() == expression.value
    match expression:
        case Set(children):
            assert children == {first, second}
        case _:
            pytest.fail("Set value matching failed")
    assert str(expression) == "Set[i64](2, 1, 2)"
    assert str(expression | expression) == "Set[i64](2, 1, 2) | Set[i64](2, 1, 2)"
    for operation, key in (
        (expression.insert(first), "insert"),
        (expression.remove(first), "remove"),
        (expression | expression, "union"),
        (expression - expression, "diff"),
        (expression & expression, "intersect"),
        (expression.contains(first), "contains"),
        (expression.not_contains(first), "not_contains"),
        (expression.length(), "length"),
    ):
        assert operation.__egg_ref__.read().kind.value.func == f"egglog.core.set.{key}"
        with pytest.raises(AttributeError):
            _ = operation.value
    with pytest.raises(AttributeError):
        _ = expression.get
    with pytest.raises(TypeError):
        Set[integer](args=(first,))
    with pytest.raises(TypeConstraintError):
        Set()


def test_set_conversion_and_typed_alias(set_runtime):
    Set, SetLike, integer = (set_runtime[name] for name in ("Set", "SetLike", "i64"))
    converted = convert({1, 2}, Set[integer])
    assert converted.value == {integer(1), integer(2)}
    empty = convert(set(), Set[integer])
    empty_function = _required_runtime_function(get_callable_fn(empty))
    assert empty_function.__egg_ref__ == Set[integer].empty.__egg_ref__
    assert isinstance(empty, Set[integer])
    with pytest.raises(set_runtime["ExprValueError"]):
        _ = empty.value
    for source in ([1], frozenset({1}), {"wrong"}):
        with pytest.raises(ConvertError):
            convert(source, Set[integer])
    assert get_type_args() == ()

    @function
    def size(value: SetLike[integer, int]) -> integer: ...  # type: ignore[valid-type]

    assert _required_callable_args(size({1, 2}))[0].value == converted.value
    nested_empty_function = _required_runtime_function(get_callable_fn(_required_callable_args(size(set()))[0]))
    assert nested_empty_function.__egg_ref__ == Set[integer].empty.__egg_ref__
    pair, string = (set_runtime[name] for name in ("Pair", "String"))
    nested = convert({(1, "a")}, Set[pair[integer, string]])
    assert next(iter(nested)).value == (integer(1), string("a"))


@pytest.mark.parametrize("empty", [False, True])
def test_set_formed_value_preserves_refs_and_pretty_roundtrip(set_runtime, empty, monkeypatch):
    Set, integer = (set_runtime[name] for name in ("Set", "i64"))
    builder = Builder()
    first = builder.import_ref(integer(2).__egg_ref__)
    second = builder.import_ref(integer(1).__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Set[integer].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["set"], pb.ValueList](
                        "set", pb.ValueList(items=[] if empty else [first, second, first])
                    )
                ),
            ),
        ),
    )
    owner = builder.publish()
    expression = RuntimeExpr(owner.ref("nodes", root))
    before = expression.__egg_ref__.read().to_binary()
    assert [item.__egg_ref__ for item in expression] == (  # type: ignore[attr-defined]
        [] if empty else [owner.ref("nodes", first), owner.ref("nodes", second)]
    )
    assert expression.value == (set() if empty else {integer(2), integer(1)})
    expected = "Set[i64]()" if empty else "Set[i64](i64(2), i64(1), i64(2))"

    def forbidden(*args, **kwargs):
        pytest.fail("Set inspection or pretty authored a Call")

    with monkeypatch.context() as inspection:
        inspection.setattr(runtime, "_author_call", forbidden)
        assert expression.value == (set() if empty else {integer(2), integer(1)})
        assert str(expression) == expected
    # Trusted display of our constructed values, not evaluation by an engine.
    authored = eval(expected, set_runtime)
    assert authored.__egg_ref__.read().kind.field == "call"
    assert list(authored) == list(expression)  # type: ignore[call-overload]
    assert get_callable_args(expression) is None
    copied = copy(expression)
    assert copied.__egg_ref__ == expression.__egg_ref__
    packed = pack([expression.__egg_ref__])
    binary = packed.program.to_binary()
    assert pb.Program.from_binary(binary).to_binary() == binary
    scalar_value = _required_oneof(packed.program.nodes[packed.indices[0]].kind, "primitive_value", pb.PrimitiveValue)
    items = _required_oneof(scalar_value.value, "set", pb.ValueList).items
    assert len(items) == (0 if empty else 3)
    if not empty:
        assert items[0] == items[2] != items[1]
        assert [
            _required_oneof(
                _required_oneof(packed.program.nodes[index].kind, "primitive_value", pb.PrimitiveValue).value,
                "i64",
                int,
            )
            for index in items
        ] == [2, 1, 2]
    assert expression.__egg_ref__.read().to_binary() == before


@pytest.mark.parametrize("same", [False, True])
def test_set_local_children_preserve_union_identity(set_runtime, same):
    Set, integer = (set_runtime[name] for name in ("Set", "i64"))
    first = RuntimeExpr(eq(integer(1)).to(integer(2)).__egg_ref__)
    second = first if same else RuntimeExpr(eq(integer(1)).to(integer(2)).__egg_ref__)
    expression = Set(first, second)
    assert len(expression) == (1 if same else 2)
    assert [child.__egg_ref__ for child in expression] == (
        [first.__egg_ref__] if same else [first.__egg_ref__, second.__egg_ref__]
    )


@pytest.mark.parametrize("malformed", ["wrong_sort", "invalid_ref", "cycle", "arity", "wrong_arm", "user_call"])
def test_set_formed_value_rejects_malformed_children(set_runtime, malformed):
    Set, integer, floating = (set_runtime[name] for name in ("Set", "i64", "f64"))
    builder = Builder()
    child = builder.import_ref((floating(1.0) if malformed == "wrong_sort" else integer(1)).__egg_ref__)
    sort_ref = Set[integer].__egg_sort__
    if malformed == "arity":
        record = sort_ref.read()
        record.kind.value.args.clear()
        sort = builder.add("sorts", record)
    else:
        sort = builder.import_ref(sort_ref)
    if malformed == "invalid_ref":
        child = 999
    elif malformed == "cycle":
        child = 1
    arm = "vec" if malformed == "wrong_arm" else "set"
    kind = Oneof[Literal["primitive_value"], pb.PrimitiveValue](
        "primitive_value",
        pb.PrimitiveValue(value=Oneof(arm, pb.ValueList(items=[child]))),  # type: ignore[arg-type]
    )
    if malformed == "user_call":
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function", pb.Function(name="egglog.core.set.of", output=sort)
                )
            ),
        )
        kind = Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.set.of", args=[child]))
    root = builder.add("nodes", pb.Node(sort_id=sort, kind=kind))
    expression = RuntimeExpr(builder.publish().ref("nodes", root))
    with pytest.raises((ValueError, IndexError, AttributeError)):
        _ = expression.value


def test_set_empty_converter_and_formed_pretty_use_retained_presentation(set_catalog_program, request):
    declarations = {d.kind.value.name: d for d in set_catalog_program.declarations}
    declarations["Set"].kind.value.bindings.python.path[-1] = "FiniteSet"
    declarations["egglog.core.set.empty"].bindings.python.views[0].path[-1] = "vacant"
    generated = request.getfixturevalue("set_runtime")
    Set, integer = generated["FiniteSet"], generated["i64"]
    assert str(convert(set(), Set[integer])) == "FiniteSet[i64].vacant()"
    builder = Builder()
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Set[integer].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["set"], pb.ValueList]("set", pb.ValueList()))
            ),
        ),
    )
    assert str(RuntimeExpr(builder.publish().ref("nodes", root))) == "FiniteSet[i64]()"


def test_map_from_pairs_rejects_set_even_with_forged_input_metadata(set_catalog_program, request):
    declaration = next(d for d in set_catalog_program.declarations if d.kind.value.name == "egglog.core.map.of")
    input_sort = set_catalog_program.sorts[declaration.kind.value.typing.value.inputs[0].sort]
    assert input_sort.kind.field == "family"
    assert input_sort.kind.value.name == "Vec"
    input_sort.kind.value.name = "Set"
    generated = request.getfixturevalue("set_runtime")
    Map, Set, Pair, integer, string = (generated[name] for name in ("Map", "Set", "Pair", "i64", "String"))
    expression = Map.from_pairs(Set(Pair(integer(1), string("a"))))
    assert expression.__egg_ref__.read().kind.field == "call"
    with pytest.raises(generated["ExprValueError"], match="Vec"):
        _ = expression.value


def test_exact_construction_and_conversions_stay_symbolic(rational_runtime):
    r, z, q, integer = (rational_runtime[name] for name in ("Rational", "BigInt", "BigRat", "i64"))
    symbolic = integer(1) + 2
    for expression, key in (
        (r(1, 2), "egglog.experimental.rational.make"),
        (z(10), "egglog.core.bigint.from_i64"),
        (z.from_string("123"), "egglog.core.bigint.from_string"),
        (q(1, 2), "egglog.core.bigrat.make"),
        (r(1, 2) + Fraction(1, 3), "egglog.experimental.rational.add"),
        (Fraction(1, 3) + r(1, 2), "egglog.experimental.rational.add"),
    ):
        assert expression.__egg_ref__.read().kind.field == "call"
        assert expression.__egg_ref__.read().kind.value.func == key
        packed = pack([expression.__egg_ref__])
        assert pb.Program.from_binary(packed.program.to_binary()).to_binary() == packed.program.to_binary()
    assert get_callable_fn(r(1, 2)) == r
    assert get_callable_fn(z.from_string("123")) == z.from_string
    assert [part.value for part in _required_callable_args(Fraction(1, 3) + r(1, 2))] == [
        Fraction(1, 3),
        Fraction(1, 2),
    ]
    for target in (r, z, q):
        converted = convert(symbolic, target)
        child = _required_callable_args(converted)[0]
        if target == q:
            child = _required_callable_args(child)[0]
        assert child.__egg_ref__ == symbolic.__egg_ref__
        with pytest.raises(rational_runtime["ExprValueError"]):
            _ = converted.value
    for target in (r, q):
        converted = convert(Fraction(1, 2), target)
        assert get_callable_fn(converted) == target
        numerator = _required_callable_args(converted)[0]
        if target == q:
            numerator = _required_callable_args(numerator)[0]
        assert numerator.__egg_ref__.read().kind.value.value.field == "i64"
    for target in (r, z, q):
        with pytest.raises(ConvertError):
            convert(1.5, target)
    assert str(r(symbolic, 2)) == "Rational(i64(1) + 2, 2)"


def test_bignum_generated_family_authors_exact_calls_without_engine(rational_runtime):
    z, q, integer = (rational_runtime[name] for name in ("BigInt", "BigRat", "i64"))
    a, b = z.from_string("123456789012345678901"), z.from_string("-17")
    x, y = q(a, z.from_string("2")), q(1, 3)
    expressions = [
        ("bigint.add", a + b, "BigInt"),
        ("bigint.sub", a - b, "BigInt"),
        ("bigint.mul", a * b, "BigInt"),
        ("bigint.div", a / b, "BigInt"),
        ("bigint.rem", a % b, "BigInt"),
        ("bigint.bitand", a & b, "BigInt"),
        ("bigint.bitor", a | b, "BigInt"),
        ("bigint.bitxor", a ^ b, "BigInt"),
        ("bigint.shl", a << integer(3), "BigInt"),
        ("bigint.shr", a >> 3, "BigInt"),
        ("bigint.not", ~a, "BigInt"),
        ("bigint.bits", a.bits(), "BigInt"),
        ("bigint.lt", a < b, "Unit"),
        ("bigint.gt", a > b, "Unit"),
        ("bigint.le", a <= b, "Unit"),
        ("bigint.ge", a >= b, "Unit"),
        ("bigint.bool_eq", a.bool_eq(b), "bool"),
        ("bigint.bool_lt", a.bool_lt(b), "bool"),
        ("bigint.bool_gt", a.bool_gt(b), "bool"),
        ("bigint.bool_le", a.bool_le(b), "bool"),
        ("bigint.bool_ge", a.bool_ge(b), "bool"),
        ("bigint.min", a.min(b), "BigInt"),
        ("bigint.max", a.max(b), "BigInt"),
        ("bigint.to_string", a.to_string(), "String"),
        ("bigrat.add", x + y, "BigRat"),
        ("bigrat.sub", x - y, "BigRat"),
        ("bigrat.mul", x * y, "BigRat"),
        ("bigrat.div", x / y, "BigRat"),
        ("bigrat.min", x.min(y), "BigRat"),
        ("bigrat.max", x.max(y), "BigRat"),
        ("bigrat.neg", -x, "BigRat"),
        ("bigrat.abs", abs(x), "BigRat"),
        ("bigrat.floor", x.floor(), "BigRat"),
        ("bigrat.ceil", x.ceil(), "BigRat"),
        ("bigrat.round", x.round(), "BigRat"),
        ("bigrat.pow", x**y, "BigRat"),
        ("bigrat.log", x.log(), "BigRat"),
        ("bigrat.sqrt", x.sqrt(), "BigRat"),
        ("bigrat.cbrt", x.cbrt(), "BigRat"),
        ("bigrat.numer", x.numer, "BigInt"),
        ("bigrat.denom", x.denom, "BigInt"),
        ("bigrat.to_f64", x.to_f64(), "f64"),
        ("bigrat.to_i64", x.to_i64(), "i64"),
        ("bigrat.lt", x < y, "Unit"),
        ("bigrat.gt", x > y, "Unit"),
        ("bigrat.le", x <= y, "Unit"),
        ("bigrat.ge", x >= y, "Unit"),
    ]
    assert len(expressions) == 47
    for suffix, expression, family in expressions:
        root = expression.__egg_ref__
        node = root.read()
        assert node.kind.field == "call"
        assert node.kind.value.func == f"egglog.core.{suffix}"
        assert root.owner.ref("sorts", node.sort_id).read().kind.value.name == family
        packed = pack([root]).program
        assert pb.Program.from_binary(packed.to_binary()).to_binary() == packed.to_binary()
    amount = integer(3)
    for expression, children in ((a << amount, (a, amount)), (x - y, (x, y)), (x.numer, (x,))):
        arguments = get_callable_args(expression)
        assert arguments is not None
        assert [child.__egg_ref__ for child in arguments] == [child.__egg_ref__ for child in children]
    assert x.value == Fraction(123456789012345678901, 2)
    for expression in (x.numer, x.denom, a >> 3, x + y):
        with pytest.raises(rational_runtime["ExprValueError"]):
            _ = expression.value
    assert str(x.numer).endswith(".numer")
    assert StructuralView((5 + a).__egg_ref__) == StructuralView((z(5) + a).__egg_ref__)
    with pytest.raises(ConvertError):
        _ = a << b


def test_closed_scalar_family_authors_exact_calls_without_engine(rational_runtime):
    integer, floating, ratio = (rational_runtime[name] for name in ("i64", "f64", "Rational"))
    count_matches = rational_runtime["count_matches"]
    a, b = integer(13), integer(2)
    x, y = floating(4.0), floating(2.0)
    r, s = ratio(3, 2), ratio(1, 2)
    expressions = [
        ("egglog.core.i64.bitor", a | b, "i64"),
        ("egglog.core.i64.bitxor", a ^ b, "i64"),
        ("egglog.core.i64.shl", a << b, "i64"),
        ("egglog.core.i64.shr", a >> b, "i64"),
        ("egglog.core.i64.not", ~a, "i64"),
        ("egglog.core.i64.log2", a.log2(), "i64"),
        ("egglog.core.i64.abs", abs(a), "i64"),
        ("egglog.core.i64.gt", a > b, "Unit"),
        ("egglog.core.i64.le", a <= b, "Unit"),
        ("egglog.core.i64.bool_lt", a.bool_lt(b), "bool"),
        ("egglog.core.i64.bool_gt", a.bool_gt(b), "bool"),
        ("egglog.core.i64.bool_le", a.bool_le(b), "bool"),
        ("egglog.core.i64.bool_ge", a.bool_ge(b), "bool"),
        ("egglog.core.f64.sub", x - y, "f64"),
        ("egglog.core.f64.mul", x * y, "f64"),
        ("egglog.core.f64.div", x / y, "f64"),
        ("egglog.core.f64.rem", x % y, "f64"),
        ("egglog.core.f64.pow", x**y, "f64"),
        ("egglog.core.f64.neg", -x, "f64"),
        ("egglog.core.f64.abs", abs(x), "f64"),
        ("egglog.core.f64.exp", x.exp(), "f64"),
        ("egglog.core.f64.log", x.log(), "f64"),
        ("egglog.core.f64.sqrt", x.sqrt(), "f64"),
        ("egglog.core.f64.lt", x < y, "Unit"),
        ("egglog.core.f64.gt", x > y, "Unit"),
        ("egglog.core.f64.le", x <= y, "Unit"),
        ("egglog.core.f64.ge", x >= y, "Unit"),
        ("egglog.core.f64.min", x.min(y), "f64"),
        ("egglog.core.f64.max", x.max(y), "f64"),
        ("egglog.core.f64.to_i64", x.to_i64(), "i64"),
        ("egglog.core.f64.from_i64", floating.from_i64(i=a), "f64"),
        ("egglog.core.f64.to_string", x.to_string(), "String"),
        ("egglog.experimental.f64.is_finite", x.is_finite(), "Unit"),
        ("egglog.experimental.rational.sub", r - s, "Rational"),
        ("egglog.experimental.rational.mul", r * s, "Rational"),
        ("egglog.experimental.rational.div", r / s, "Rational"),
        ("egglog.experimental.rational.min", r.min(s), "Rational"),
        ("egglog.experimental.rational.max", r.max(s), "Rational"),
        ("egglog.experimental.rational.neg", -r, "Rational"),
        ("egglog.experimental.rational.abs", abs(r), "Rational"),
        ("egglog.experimental.rational.floor", r.floor(), "Rational"),
        ("egglog.experimental.rational.ceil", r.ceil(), "Rational"),
        ("egglog.experimental.rational.round", r.round(), "Rational"),
        ("egglog.experimental.rational.numer", r.numer, "i64"),
        ("egglog.experimental.rational.denom", r.denom, "i64"),
        ("egglog.experimental.rational.to_f64", r.to_f64(), "f64"),
        ("egglog.experimental.rational.pow", r**s, "Rational"),
        ("egglog.experimental.rational.log", r.log(), "Rational"),
        ("egglog.experimental.rational.sqrt", r.sqrt(), "Rational"),
        ("egglog.experimental.rational.cbrt", r.cbrt(), "Rational"),
        ("egglog.experimental.rational.lt", r < s, "Unit"),
        ("egglog.experimental.rational.gt", r > s, "Unit"),
        ("egglog.experimental.rational.le", r <= s, "Unit"),
        ("egglog.experimental.rational.ge", r >= s, "Unit"),
        ("egglog.core.string.count_matches", count_matches(s="a\0a", pattern="a"), "i64"),
    ]
    assert len(expressions) == 55
    assert len({identity for identity, _, _ in expressions}) == 55
    for identity, expression, family in expressions:
        root = expression.__egg_ref__
        node = root.read()
        assert node.kind.field == "call"
        assert node.kind.value.func == identity
        assert root.owner.ref("sorts", node.sort_id).read().kind.value.name == family
        packed = pack([root])
        decoded = pb.Program.from_binary(packed.program.to_binary())
        restored = Builder.from_program(decoded).publish().ref("nodes", packed.indices[0])
        assert StructuralView(restored) == StructuralView(root)
        assert str(RuntimeExpr(restored)) == str(expression)
        if family != "Unit":
            with pytest.raises(rational_runtime["ExprValueError"]):
                _ = expression.value
    for expression, children in (
        (a << b, (a, b)),
        (x - y, (x, y)),
        (r - s, (r, s)),
        (r.numer, (r,)),
        (floating.from_i64(i=a), (a,)),
    ):
        actual_children = get_callable_args(expression)
        assert actual_children is not None
        assert [child.__egg_ref__ for child in actual_children] == [child.__egg_ref__ for child in children]
    for reflected, explicit in (
        (2 | a, integer(2) | a),
        (2 ^ a, integer(2) ^ a),
        (2 << a, integer(2) << a),
        (2 >> a, integer(2) >> a),
        (2 - x, floating(2.0) - x),
        (2 / x, floating(2.0) / x),
        (2**x, floating(2.0) ** x),
        (2 - r, ratio(2, 1) - r),
        (2 / r, ratio(2, 1) / r),
        (2**r, ratio(2, 1) ** r),
        (floating.from_i64(1), floating.from_i64(integer(1))),
    ):
        assert StructuralView(reflected.__egg_ref__) == StructuralView(explicit.__egg_ref__)
    assert str(r.numer) == "Rational(3, 2).numer"
    assert str(floating.from_i64(1)) == "f64.from_i64(1)"
    assert str(count_matches(s="aba", pattern="a")) == "count_matches('aba', 'a')"
    for operation, argument in (
        (a.bool_lt, x),
        (a.bool_gt, "2"),
        (a.bool_le, r),
        (a.bool_ge, x),
        (floating.from_i64, x),
        (r.min, x),
    ):
        with pytest.raises(ConvertError):
            operation(argument)
    with pytest.raises(TypeError):
        floating.from_i64()
    with pytest.raises(TypeError):
        floating.from_i64(1, 2)


def test_exact_legacy_local_inspection_and_protocols(rational_runtime):
    r, z, q = (rational_runtime[name] for name in ("Rational", "BigInt", "BigRat"))
    half = r(1, 2)
    assert half.value == Fraction(1, 2)
    assert float(half) == 0.5
    assert int(half) == 0
    assert r(1, -2).value == Fraction(-1, 2)
    with pytest.raises(ZeroDivisionError):
        _ = r(1, 0).value
    huge = 10**100 + 1
    formed_integer = z.from_string(str(huge))
    assert formed_integer.value == huge
    assert int(formed_integer) == operator.index(formed_integer) == huge
    formed_ratio = q(formed_integer, z.from_string("2"))
    assert formed_ratio.value == Fraction(huge, 2)
    assert int(formed_ratio) == huge // 2
    assert float(formed_ratio) == float(Fraction(huge, 2))
    for value in (half, formed_integer, formed_ratio):
        with pytest.warns(DeprecationWarning, match="use .value"):
            assert value.eval() == value.value
    match half:
        case r(value):
            assert value == Fraction(1, 2)
        case _:
            pytest.fail("Exact value match protocol missing")
    # Keep the old local form limits: from_i64 is a native call, not a literal.
    for symbolic in (z(10), q(1, 2), half + Fraction(1, 3)):
        with pytest.raises(rational_runtime["ExprValueError"]):
            _ = symbolic.value
    with pytest.raises(ValueError, match="invalid literal"):
        _ = z.from_string("invalid").value


def test_exact_pretty_uses_retained_classmethod_presentation(rational_runtime, rational_catalog_program):
    from egglog.pretty import pretty_ref  # noqa: PLC0415

    declaration = next(
        d for d in rational_catalog_program.declarations if d.kind.value.name == "egglog.core.bigint.from_string"
    )
    declaration.bindings.python.views[0].path[0] = "decimal"
    family = next(d.kind.value for d in rational_catalog_program.declarations if d.kind.value.name == "BigInt")
    family.bindings.python.path[-1] = "Integer"
    for name in ("egglog.core.bigrat.make", "egglog.experimental.rational.make"):
        declaration = next(d for d in rational_catalog_program.declarations if d.kind.value.name == name)
        declaration.bindings.python.views[0].params.reverse()
    builder = Builder.from_program(rational_catalog_program)
    for name, arm, payload, expected in (
        ("BigInt", "big_int", "123", "Integer.decimal('123')"),
        (
            "BigRat",
            "big_rat",
            pb.BigRat(numerator="1", denominator="2"),
            "BigRat(Integer.decimal('2'), Integer.decimal('1'))",
        ),
        ("Rational", "rational", pb.Rational(numerator=1, denominator=2), "Rational(2, 1)"),
    ):
        sort = next(
            i
            for i, s in enumerate(rational_catalog_program.sorts)
            if s.kind.field == "family" and s.kind.value.name == name
        )
        index = builder.add(
            "nodes",
            pb.Node(
                sort_id=sort,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(value=Oneof(arm, payload)),  # type: ignore[arg-type]
                ),
            ),
        )
        # The ambient generated classes still have the original BigInt name and
        # from_string method. Only this owned record contains the changed view.
        assert pretty_ref(builder.publish().ref("nodes", index)) == expected
        builder = Builder.from_program(rational_catalog_program)


@pytest.mark.parametrize(
    ("family", "arm", "payload", "expected"),
    [
        ("BigInt", "big_int", "-123456789012345678901234567890", -123456789012345678901234567890),
        (
            "BigRat",
            "big_rat",
            pb.BigRat(numerator="100000000000000000001", denominator="3"),
            Fraction(100000000000000000001, 3),
        ),
        ("Rational", "rational", pb.Rational(numerator=-1, denominator=2), Fraction(-1, 2)),
        ("Rational", "rational", pb.Rational(numerator=0, denominator=1), Fraction(0)),
    ],
)
def test_exact_inert_values_are_owned_inspectable_and_printable(rational_runtime, family, arm, payload, expected):
    from egglog.deconstruct import get_literal_value  # noqa: PLC0415

    cls = rational_runtime[family]
    builder = Builder()
    primitive = pb.PrimitiveValue(value=Oneof(arm, payload))
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(cls.__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue]("primitive_value", primitive),
        ),
    )
    expression = RuntimeExpr(builder.publish().ref("nodes", index))
    if not isinstance(payload, str):
        payload.denominator = "0" if arm == "big_rat" else 0
    assert get_literal_value(expression) == expression.value == expected
    assert get_callable_args(expression) is None
    assert copy(expression).__egg_ref__ == expression.__egg_ref__
    # Display uses actual constructor/classmethod presentations, never f64 or
    # frontend evaluation. Reauthoring is symbolic, not an inert-record clone.
    displayed = eval(str(expression), rational_runtime)
    assert displayed.value == expected
    assert displayed.__egg_ref__.read().kind.field == "call"
    packed = pack([expression.__egg_ref__])
    detached = Builder.from_program(pb.Program.from_binary(packed.program.to_binary())).publish()
    assert RuntimeExpr(detached.ref("nodes", packed.indices[0])).value == expected


@pytest.mark.parametrize("access", ["literal", "value", "display", "binary", "display_value"])
@pytest.mark.parametrize("form", ["integer", "numerator", "denominator", "both"])
def test_exact_inert_decimal_magnitude_ignores_process_digit_limit(rational_runtime, access, form):
    from egglog.deconstruct import get_literal_value  # noqa: PLC0415

    limit = sys.get_int_max_str_digits()
    digits = max(5000, limit + 1)
    text, huge = "9" * digits, 10**digits - 1
    if form == "integer":
        family, arm, payload, expected = "BigInt", "big_int", "-" + text, -huge
        display = f"BigInt.from_string('-{text}')"
    else:
        numerator, denominator, expected = {
            "numerator": ("-" + text, "2", Fraction(-huge, 2)),
            "denominator": ("1", text, Fraction(1, huge)),
            "both": (text, "1" + "0" * (digits - 1), Fraction(huge, 10 ** (digits - 1))),
        }[form]
        family, arm = "BigRat", "big_rat"
        payload = pb.BigRat(numerator=numerator, denominator=denominator)
        display = f"BigRat(BigInt.from_string({numerator!r}), BigInt.from_string({denominator!r}))"
    builder = Builder()
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(rational_runtime[family].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(value=Oneof(arm, payload)),  # type: ignore[arg-type]
            ),
        ),
    )
    expression = RuntimeExpr(builder.publish().ref("nodes", index))
    try:
        if access == "literal":
            assert get_literal_value(expression) == expected
        elif access == "value":
            assert expression.value == expected
        elif access == "display":
            assert (
                str(expression)
                == black.format_str(display, mode=black.Mode(line_length=180, string_normalization=False)).strip()
            )
        elif access == "binary":
            packed = pack([expression.__egg_ref__])
            encoded = packed.program.to_binary()
            decoded = pb.Program.from_binary(encoded)
            assert decoded.to_binary() == encoded
            detached = Builder.from_program(decoded).publish()
            assert RuntimeExpr(detached.ref("nodes", packed.indices[0])).value == expected
        else:
            # Independent of inert decoding: the existing display contract
            # reauthors a symbolic from-string Call that remains inspectable.
            reauthored = eval(display, rational_runtime)
            assert reauthored.__egg_ref__.read().kind.field == "call"
            assert reauthored.value == expected
    finally:
        assert sys.get_int_max_str_digits() == limit


def test_exact_from_string_inspection_preserves_noncanonical_int_behavior(rational_runtime):
    cls = rational_runtime["BigInt"]
    for text, expected in (("  +001_234 ", 1234), ("-0", 0), ("+1", 1), ("01", 1)):
        assert cls.from_string(text).value == expected
    for text in ("1e3", "1.5", "NaN", "Infinity"):
        with pytest.raises(ValueError, match="invalid literal"):
            _ = cls.from_string(text).value


@pytest.mark.parametrize(
    ("family", "arm", "payload"),
    [
        ("BigInt", "big_int", "01"),
        ("BigInt", "big_int", "-0"),
        ("BigInt", "big_int", "+1"),
        ("BigInt", "big_int", "1_0"),
        ("BigInt", "big_int", " 1"),
        ("BigInt", "big_int", "1e3"),
        ("BigInt", "big_int", "1.5"),
        ("BigInt", "big_int", "NaN"),
        ("BigRat", "big_rat", pb.BigRat(numerator="1", denominator="2.0")),
        ("BigInt", "rational", pb.Rational(numerator=1, denominator=2)),
        ("Rational", "i64", 1),
        ("BigRat", "big_rat", pb.BigRat(numerator="2", denominator="4")),
        ("BigRat", "big_rat", pb.BigRat(numerator="0", denominator="2")),
        ("BigRat", "big_rat", pb.BigRat(numerator="1", denominator="-2")),
        ("BigRat", "big_rat", pb.BigRat(numerator="+1", denominator="2")),
        ("Rational", "rational", pb.Rational(numerator=2, denominator=4)),
        ("Rational", "rational", pb.Rational(numerator=1, denominator=0)),
        ("Rational", "rational", pb.Rational(numerator=0, denominator=2)),
        ("Rational", "rational", pb.Rational(numerator=2**63, denominator=1)),
        ("Rational", "rational", pb.Rational(numerator=1, denominator=2**63)),
    ],
)
def test_exact_inert_values_reject_malformed_data(rational_runtime, family, arm, payload):
    from egglog.deconstruct import get_literal_value  # noqa: PLC0415

    builder = Builder()
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(rational_runtime[family].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof(arm, payload))
            ),
        ),
    )
    expression = RuntimeExpr(builder.publish().ref("nodes", index))
    with pytest.raises(ValueError, match=r"canonical|requires|reduced"):
        get_literal_value(expression)
    with pytest.raises(ValueError, match=r"canonical|requires|reduced"):
        _ = expression.value


def test_scalar_wave_local_protocols_and_matching_never_evaluate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Scalar authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    text, false, unit = String("a\0b\n☃"), Bool(False), Unit()
    assert text.value == "a\0b\n☃"
    assert false.value is False
    assert bool(false) is False
    assert bool(Bool(True)) is True
    assert convert("x", String).value == "x"
    assert convert(False, Bool).value is False
    with pytest.warns(DeprecationWarning, match="use .value"):
        assert false.eval() is False
    with pytest.warns(DeprecationWarning, match="use .value"):
        assert text.eval() == text.value
    match text, false:
        case String(payload), Bool(value):
            assert payload == "a\0b\n☃"
            assert value is False
        case _:
            pytest.fail("Scalar literal matching failed")
    for computed in (Bool(True) | False, ~Bool(False)):
        with pytest.raises(ExprValueError):
            bool(computed)
    with pytest.raises(ExprValueError):
        _ = join("a", "b").value
    with pytest.raises(ValueError, match="Can only check equality"):
        bool(unit)
    assert not hasattr(unit, "value")
    assert not hasattr(unit, "eval")
    assert str(unit) == "Unit()"


@pytest.mark.parametrize(
    ("constructor", "arguments"), [(String, (1,)), (String, ()), (Bool, (1,)), (Bool, ()), (Unit, (None,))]
)
def test_scalar_wave_literal_constructors_reject_wrong_type_or_arity(constructor, arguments):
    with pytest.raises(TypeError):
        constructor(*arguments)


def test_scalar_wave_canonical_calls_and_reverse_operand_order():
    calls = [
        (join(), "string.concat", ()),
        (join("a"), "string.concat", ("a",)),
        (join("a", "b", "c"), "string.concat", ("a", "b", "c")),
        (String("a") + "b", "string.concat", ("a", "b")),
        ("left" + String("right"), "string.concat", ("left", "right")),
        (String("abc").replace(new="x", old="b"), "string.replace", ("abc", "b", "x")),
        (i64(-42).to_string(), "i64.to_string", (-42,)),
        (~Bool(False), "bool.not", (False,)),
        (Bool(True) & False, "bool.and", (True, False)),
        (True & Bool(False), "bool.and", (True, False)),
        (Bool(True) | False, "bool.or", (True, False)),
        (True | Bool(False), "bool.or", (True, False)),
        (Bool(True) ^ False, "bool.xor", (True, False)),
        (True ^ Bool(False), "bool.xor", (True, False)),
        (Bool(True).implies(False), "bool.implies", (True, False)),
    ]
    for expression, name, values in calls:
        call = expression.__egg_ref__.read().kind.value
        assert call.func == f"egglog.core.{name}"
        assert tuple(child.value for child in _required_callable_args(expression)) == values
    assert str(String("a") + "b") == "join('a', 'b')"
    assert str(~Bool(False)) == "~Bool(False)"
    assert str(~(Bool(True) | False)) == "~(Bool(True) | False)"
    assert str((~Bool(True)) & False) == "(~Bool(True)) & False"


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_scalar_wave_canonical_bool_pretty_preserves_every_argument(count):
    or_function = _required_runtime_function(Bool.__or__)
    canonical = RuntimeFunction(or_function.__egg_ref__, None)
    values = [index % 2 == 0 for index in range(count)]
    expression = canonical(*values)
    assert [child.value for child in _required_callable_args(expression)] == values
    expected = (
        "Bool(True) | False"
        if count == 2
        else "egglog.core.bool.or(" + ", ".join(f"Bool({value})" for value in values) + ")"
    )
    assert str(expression) == expected


@pytest.mark.parametrize(
    ("target", "member"),
    [
        pytest.param(String("a").__add__, String.__add__, id="invoke0"),
        pytest.param(Bool(False).__or__, Bool.__or__, id="invoke1"),
    ],
)
def test_scalar_wave_binary_view_rejects_wrong_arity_and_sort(target, member):
    for args in ((), ("a", "b")):
        with pytest.raises(TypeError):
            target(*args)
    # Operator dispatch may promote both operands; validate this exact member's sort.
    with pytest.raises(ConvertError):
        member(target.__self__, i64(1))
    with pytest.raises(TypeError, match="unexpected keyword"):
        target(other=True, name="implies")


def test_scalar_wave_specialized_view_permutation_and_zero_count():
    packet = pack([(String("left") + "right").__egg_ref__])
    declaration = next(
        d
        for d in packet.program.declarations
        if (
            d.kind.value.name == "egglog.core.string.concat"
            if d.kind is not None
            else pytest.fail("Missing declaration kind")
        )
    )
    view = _required_python_bindings(declaration).views[1]
    view.receiver, view.params[0].core_input = 1, 0
    owner = Builder.from_program(packet.program).publish()
    target = RuntimeFunction(
        owner.ref("declarations", packet.program.declarations.index(declaration)), 1, to_runtime_expr(String("right"))
    )
    result = target("left")
    assert [child.value for child in _required_callable_args(result)] == ["left", "right"]
    assert str(result) == "join('left', 'right')"
    view = _required_python_bindings(declaration).views[0]
    view.tail_repetitions = 0
    view.params.clear()
    owner = Builder.from_program(packet.program).publish()
    empty = RuntimeFunction(owner.ref("declarations", packet.program.declarations.index(declaration)), 0)
    assert get_callable_args(empty()) == ()
    with pytest.raises(TypeError):
        empty("extra")


def test_scalar_wave_specialized_bool_deconstruction_uses_effective_slots():
    packet = pack([(Bool(True) | False).__egg_ref__])
    declaration = next(
        d
        for d in packet.program.declarations
        if (
            d.kind.value.name == "egglog.core.bool.or"
            if d.kind is not None
            else pytest.fail("Missing declaration kind")
        )
    )
    view = _required_python_bindings(declaration).views[0]
    view.receiver, view.params[0].core_input = 1, 0
    owner = Builder.from_program(packet.program).publish()
    method = RuntimeFunction(
        owner.ref("declarations", packet.program.declarations.index(declaration)), 0, to_runtime_expr(Bool(False))
    )
    result = method(True)
    assert result is not None
    call = result.__egg_ref__.read().kind.value
    assert [RuntimeExpr(result.__egg_ref__.owner.ref("nodes", i)).value for i in call.args] == [True, False]
    assert [argument.value for argument in _required_callable_args(result)] == [False, True]
    assert str(result) == "Bool(False) | True"
    selected_fn = get_callable_fn(result)
    assert isinstance(selected_fn, RuntimeFunction)
    reconstructed = selected_fn(*_required_callable_args(result))
    assert reconstructed is not None
    assert expr_parts(reconstructed) == expr_parts(result)


@pytest.mark.parametrize(
    "mutation", ["fixed", "missing", "missing_input", "duplicate", "default", "mutation", "huge", "owner", "property"]
)
def test_scalar_wave_authoring_rejects_invalid_specialization(mutation):
    packet = pack([(Bool(True) | False).__egg_ref__])
    declaration = next(
        d
        for d in packet.program.declarations
        if (
            d.kind.value.name == "egglog.core.bool.or"
            if d.kind is not None
            else pytest.fail("Missing declaration kind")
        )
    )
    primitive = _required_oneof(declaration.kind, "host_primitive", pb.HostPrimitive)
    signature = _required_oneof(primitive.typing, "signature", pb.GenericSignature)
    view = _required_python_bindings(declaration).views[0]
    if mutation == "fixed":
        signature.inputs = signature.varargs * 2
        signature.varargs.clear()
    elif mutation == "missing":
        view.params.clear()
    elif mutation == "missing_input":
        view.params[0].clear_field("core_input")
    elif mutation == "duplicate":
        view.params[0].core_input = 0
    elif mutation == "default":
        view.params[0].default_expr = packet.indices[0]
    elif mutation == "mutation":
        view.mutates = 0
    elif mutation == "owner":
        packet.program.sorts.append(
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="String")))
        )
        assert view.owner is not None
        view.owner.kind = Oneof[Literal["sort"], int]("sort", len(packet.program.sorts) - 1)
    elif mutation == "property":
        view.kind = pb.PythonCallKind.PROPERTY
    else:
        view.tail_repetitions = 2**32 - 1
    owner = Builder.from_program(packet.program).publish()
    method = RuntimeFunction(
        owner.ref("declarations", packet.program.declarations.index(declaration)), 0, to_runtime_expr(Bool(True))
    )
    with pytest.raises((TypeError, ValueError), match=r"signature|input|owner|maps|Sort mismatch|views"):
        method(False)


def test_scalar_wave_concrete_defaults_and_nested_containers():
    default = String("a") + "b"

    @function
    def message(prefix: StringLike = default, enabled: BoolLike = False) -> String: ...

    @function
    def flag(value: BoolLike = Bool(False)) -> Bool: ...

    first, second = message(), message()
    assert str(first) == "message()"
    first_args, second_args = _required_callable_args(first), _required_callable_args(second)
    assert first_args[0].__egg_ref__ != second_args[0].__egg_ref__
    assert expr_parts(first_args[0]) == expr_parts(default)
    assert first_args[1].value is False
    assert _required_callable_args(flag())[0].value is False
    assert [part.value for part in _required_callable_args(message("x", True))] == ["x", True]
    pair = Pair[String, Bool]("key", False)  # type: ignore[arg-type]
    mapping = Map[String, Bool].empty().insert("key", False)  # type: ignore[arg-type]
    assert tuple(part.value for part in pair.value) == ("key", False)
    assert mapping.value[String("key")].value is False
    assert isinstance(mapping[String("key")], Bool)
    with pytest.raises(ConvertError):
        Pair[String, Bool](False, "key")  # type: ignore[arg-type]
    with pytest.raises(ConvertError):
        message(i64(1))  # type: ignore[arg-type]


def test_runtime_instance_checks_preserve_concrete_sorts_without_engine(monkeypatch):
    from egglog import MultiSet  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime instance checking accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)

    class Number(Expr):
        def __init__(self) -> None: ...

    class Other(Expr):
        def __init__(self) -> None: ...

    monomial = Map[Number, i64]
    polynomial = Map[monomial, f64]

    @function
    def choose(values: Map[i64, f64]) -> Number: ...

    checks = [
        (polynomial.empty(), polynomial, True),
        (polynomial.empty(), monomial, False),
        (monomial.empty(), polynomial, False),
        (polynomial.empty(), Map[Map[Number, f64], f64], False),
        (polynomial.empty(), Map[Map[Other, i64], f64], False),
        (Map[i64, f64].empty(), Map[f64, i64], False),
        (polynomial.empty(), Map, True),
        (MultiSet[Number](), MultiSet[Number], True),
        (MultiSet[Number](), MultiSet[Other], False),
        (MultiSet[Number](), MultiSet, True),
        (UnstableFn(choose), UnstableFn[Number, Map[i64, f64]], True),
        (UnstableFn(choose), UnstableFn[Number, Map[f64, i64]], False),
        (UnstableFn(choose), UnstableFn[Other, Map[i64, f64]], False),
        (UnstableFn(choose), UnstableFn, True),
        (Number(), Number, True),
        (Number(), Other, False),
        (i64(1), i64, True),
        (i64(1), f64, False),
        (None, Map, False),
        (1, i64, False),
    ]
    for value, selected, expected in checks:
        assert isinstance(value, selected) is expected, (value, selected)


def test_map_authoring_symbolic_children_and_local_protocols(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Map authoring/inspection accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    key, first, newest = i64(1) + 2, f64(3.0), f64(4.0)
    empty = Map[i64, f64].empty()
    assert empty.value == {}
    expression = empty.insert(key, first).insert(i64(1) + 2, newest)
    assert list(expression.value) == [key]
    assert next(iter(expression.value)).__egg_ref__ == key.__egg_ref__
    assert expression.value[key].__egg_ref__ == newest.__egg_ref__
    assert list(expression) == [key]
    assert len(expression) == 1
    assert key in expression
    assert i64(9) not in expression
    with pytest.warns(DeprecationWarning, match="use .value"):
        assert expression.eval() == expression.value
    match expression:
        case Map(items):
            assert items == expression.value
        case _:
            pytest.fail("Map value matching failed")
    selected = expression[key]
    assert isinstance(selected, f64)
    assert selected.__egg_ref__.read().kind.value.func == "egglog.core.map.get"
    assert str(selected).endswith("[i64(1) + 2]")
    with pytest.raises(ExprValueError):
        _ = selected.value


def test_map_fixed_from_pairs_and_computed_vec_authoring():
    first, newest = f64(2.0), f64(3.0)
    entries = Vec(Pair(i64(1), first), Pair(i64(1), newest))
    expression = Map.from_pairs(entries)
    assert isinstance(expression, Map[i64, f64])  # type: ignore[misc]
    assert _required_callable_args(expression)[0].__egg_ref__ == entries.__egg_ref__
    assert expression.value[i64(1)].__egg_ref__ == newest.__egg_ref__
    assert Map[i64, f64].from_pairs(Vec[Pair[i64, f64]].empty()).value == {}
    with pytest.raises(TypeError):
        Map[i64, f64].from_pairs()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Map[i64, f64].from_pairs(items=entries)  # type: ignore[call-arg]
    with pytest.raises(ConvertError):
        Map[i64, f64].from_pairs(Pair(i64(1), f64(2.0)))  # type: ignore[arg-type]
    with pytest.raises(ConvertError):
        Map[i64, f64].from_pairs(Vec(Pair(f64(1.0), i64(2))))  # type: ignore[arg-type]

    @function
    def computed() -> Vec[Pair[i64, f64]]: ...

    symbolic = computed()
    mapping = Map.from_pairs(symbolic)
    assert _required_callable_args(mapping)[0].__egg_ref__ == symbolic.__egg_ref__
    assert len(mapping.__egg_ref__.read().kind.value.args) == 1
    assert "computed" in str(mapping)
    with pytest.raises(ExprValueError):
        _ = mapping.value
    assert isinstance(mapping[i64(1)], f64)
    assert "Map" in {
        d.kind.value.name if d.kind is not None else pytest.fail("Missing declaration kind")
        for d in pack([mapping.__egg_ref__]).program.declarations
    }


@pytest.mark.parametrize("empty", [False, True])
def test_map_value_payload_order_ref_zero_and_pretty(empty):
    builder = Builder()
    key = builder.import_ref(i64(2).__egg_ref__)
    assert key == 0
    value = builder.import_ref(f64(1.0).__egg_ref__)
    later = builder.import_ref(f64(3.0).__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Map[i64, f64].__egg_sort__),  # type: ignore[attr-defined]
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["map"], pb.MapValue](
                        "map",
                        pb.MapValue(
                            entries=[]
                            if empty
                            else [pb.MapEntry(key=key, value=value), pb.MapEntry(key=key, value=later)]
                        ),
                    )
                ),
            ),
        ),
    )
    owner = builder.publish()
    expression = RuntimeExpr(owner.ref("nodes", root))
    assert get_callable_args(expression) is None
    if empty:
        assert expression.value == {}
        assert str(expression) == "Map[i64, f64].empty()"
    else:
        mapping = expression.value
        assert isinstance(mapping, dict)
        first_key = next(iter(mapping))
        assert isinstance(first_key, RuntimeExpr)
        selected = mapping[i64(2)]
        assert isinstance(selected, RuntimeExpr)
        assert first_key.__egg_ref__ == owner.ref("nodes", key)
        assert selected.__egg_ref__ == owner.ref("nodes", later)
        assert str(expression) == "Map[i64, f64].empty().insert(i64(2), f64(1.0)).insert(i64(2), f64(3.0))"


def test_map_dict_conversion_context_and_exact_target_precedence():
    expression = convert({1: 2.0, 3: 4.0}, Map[i64, f64])
    assert [(key.value, value.value) for key, value in expression.value.items()] == [(1, 2.0), (3, 4.0)]
    assert expression.__egg_ref__.read().kind.value.func == "egglog.core.map.insert"
    assert convert({}, Map[i64, f64]).__egg_ref__.read().kind.value.func == "egglog.core.map.empty"
    assert get_type_args() == ()
    with pytest.raises(ConvertError):
        convert({f64(1.0): 2.0}, Map[i64, f64])
    with pytest.raises(ConvertError):
        convert({1: i64(2)}, Map[i64, f64])
    sentinel = Map[i64, f64].empty()
    converter(object, Map[i64, f64], lambda source: sentinel, cost=0)
    assert convert({1: 2.0}, Map[i64, f64]) is sentinel


def test_map_like_annotation_nested_conversion_and_order():
    @function
    def size(value: MapLike[i64, f64, int, float]) -> i64: ...

    converted = _required_callable_args(size({1: 2.0}))[0]
    assert isinstance(converted, Map[i64, f64])  # type: ignore[misc]
    assert converted.value[i64(1)].value == 2.0
    with pytest.raises(TypeError):
        _ = MapLike[i64, f64, int]  # type: ignore[type-arg]
    with pytest.raises(ConvertError):
        size({1: i64(2)})  # type: ignore[dict-item]
    nested = convert({1: [(2, 3.0)]}, Map[i64, Vec[Pair[i64, f64]]])
    assert nested.value[i64(1)].value[0].value[1].value == 3.0
    assert get_type_args() == ()

    class Key:
        pass

    class Value:
        pass

    events = []

    def key(source):
        events.append("key")
        assert get_type_args() == ()
        return i64(1)

    def value(source):
        events.append("value")
        assert get_type_args() == ()
        return f64(2.0)

    converter(Key, i64, key)
    converter(Value, f64, value)
    assert convert({Key(): Value(), Key(): Value()}, Map[i64, f64]).value[i64(1)].value == 2.0
    assert events == ["key", "value", "key", "value"]
    sentinel = Map[i64, f64].empty()
    converter(dict, Map[i64, f64], lambda source: sentinel, cost=0)
    assert convert({}, Map[i64, f64]) is sentinel


@pytest.mark.parametrize("same", [False, True])
def test_map_union_key_identity_and_nested_children(same):
    first = RuntimeExpr(eq(i64(1)).to(i64(2)).__egg_ref__)
    second = first if same else RuntimeExpr(eq(i64(1)).to(i64(2)).__egg_ref__)
    value = Vec(Pair(i64(3), f64(4.0)))
    mapping = Map.from_pairs(Vec(Pair(first, value), Pair(second, value)))
    children = mapping.value
    assert len(children) == (1 if same else 2)
    assert [key.__egg_ref__ for key in children] == (
        [first.__egg_ref__] if same else [first.__egg_ref__, second.__egg_ref__]
    )
    assert all(child.__egg_ref__ == value.__egg_ref__ for child in children.values())


@pytest.mark.parametrize(
    "invalid",
    [
        "key_sort",
        "value_sort",
        "missing_key",
        "missing_value",
        "oob",
        "cycle",
        "leaf_cycle",
        "union_oob",
        "union_cycle",
        "arity",
        "wrong_family",
        "user_call",
        "call_arity",
        "nominal",
    ],
)
def test_map_invalid_records_are_explicit_local_inspection_errors(invalid):
    if invalid == "nominal":

        class NominalMap(Expr, egg_sort="Map"):
            def __init__(self) -> None: ...

        with pytest.raises(ExprValueError):
            _map_entries(NominalMap())  # type: ignore[arg-type]
        with pytest.raises(ConvertError):
            convert({}, NominalMap)
        return
    builder = Builder()
    key = builder.import_ref((f64(1.0) if invalid == "key_sort" else i64(1)).__egg_ref__)
    value = builder.import_ref((i64(2) if invalid == "value_sort" else f64(2.0)).__egg_ref__)
    sort = builder.import_ref((Vec[i64] if invalid == "wrong_family" else Map[i64, f64]).__egg_sort__)  # type: ignore[attr-defined]
    if invalid == "arity":
        record = Map[i64, f64].__egg_sort__.read()  # type: ignore[attr-defined]
        record.kind.value.args = [builder.import_ref(i64.__egg_sort__)]  # type: ignore[attr-defined]
        sort = builder.add("sorts", record)
    root = builder.reserve("nodes")
    if invalid == "oob":
        value = 999
    elif invalid == "cycle":
        value = root
    elif invalid == "leaf_cycle":
        value = builder.reserve("nodes")
        builder.fill(
            "nodes",
            value,
            pb.Node(
                sort_id=builder.import_ref(f64.__egg_sort__),  # type: ignore[attr-defined]
                kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.f64.add", args=[value, value])),
            ),
        )
    elif invalid in {"union_oob", "union_cycle"}:
        value = builder.reserve("nodes")
        builder.fill(
            "nodes",
            value,
            pb.Node(
                sort_id=builder.import_ref(f64.__egg_sort__),  # type: ignore[attr-defined]
                kind=Oneof[Literal["union"], pb.Union](
                    "union", pb.Union(members=[999 if invalid == "union_oob" else value])
                ),
            ),
        )
    entry = pb.MapEntry(
        key=None if invalid == "missing_key" else key, value=None if invalid == "missing_value" else value
    )
    kind = Oneof[Literal["primitive_value"], pb.PrimitiveValue](
        "primitive_value",
        pb.PrimitiveValue(value=Oneof[Literal["map"], pb.MapValue]("map", pb.MapValue(entries=[entry]))),
    )
    if invalid == "user_call":
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function", pb.Function(name="egglog.core.map.empty", output=sort)
                )
            ),
        )
        kind = Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.map.empty"))
    elif invalid == "call_arity":
        kind = Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.map.of", args=[key, value]))
    builder.fill("nodes", root, pb.Node(sort_id=sort, kind=kind))
    with pytest.raises(ExprValueError):
        _map_entries(RuntimeExpr(builder.publish().ref("nodes", root)))


@pytest.mark.parametrize("invalid", ["pair_missing", "vec_oob", "pair_user"])
def test_map_from_pairs_rejects_invalid_nested_provenance(invalid):
    builder = Builder()
    key = builder.import_ref(i64(1).__egg_ref__)
    value = builder.import_ref(f64(2.0).__egg_ref__)
    pair_sort = builder.import_ref(Pair[i64, f64].__egg_sort__)  # type: ignore[attr-defined]
    pair_kind = Oneof[Literal["primitive_value"], pb.PrimitiveValue](
        "primitive_value",
        pb.PrimitiveValue(
            value=Oneof[Literal["pair"], pb.PairValue](
                "pair", pb.PairValue(first=key, second=None if invalid == "pair_missing" else value)
            )
        ),
    )
    if invalid == "pair_user":
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function",
                    pb.Function(
                        name="egglog.core.pair.make",
                        inputs=[
                            pb.Arg(sort=builder.import_ref(i64.__egg_sort__)),  # type: ignore[attr-defined]
                            pb.Arg(sort=builder.import_ref(f64.__egg_sort__)),  # type: ignore[attr-defined]
                        ],
                        output=pair_sort,
                    ),
                )
            ),
        )
        pair_kind = Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.pair.make", args=[key, value]))
    pair = builder.add("nodes", pb.Node(sort_id=pair_sort, kind=pair_kind))
    vector = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Vec[Pair[i64, f64]].__egg_sort__),  # type: ignore[attr-defined]
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["vec"], pb.ValueList](
                        "vec", pb.ValueList(items=[999 if invalid == "vec_oob" else pair])
                    )
                ),
            ),
        ),
    )
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Map[i64, f64].__egg_sort__),  # type: ignore[attr-defined]
            kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.map.of", args=[vector])),
        ),
    )
    with pytest.raises(ExprValueError):
        RuntimeExpr(builder.publish().ref("nodes", root)).value


@pytest.mark.parametrize("source", ["dict", "value"])
@pytest.mark.parametrize("same_sort", [False, True])
def test_map_insert_view_parameter_permutation_preserves_core_order(vec_catalog_program, request, source, same_sort):
    insert = next(d for d in vec_catalog_program.declarations if d.kind.value.name == "egglog.core.map.insert")
    view = insert.bindings.python.views[0]
    view.path = ["put"]
    view.params.reverse()
    view.params[0].name, view.params[1].name = "item", "at"
    request.getfixturevalue("vec_runtime")
    value_type = i64 if same_sort else f64
    value = i64(2) if same_sort else f64(2.0)
    target = runtime._BASE_CLASSES["Map"][i64, value_type]
    assert target.__egg_sort__ is not None
    if source == "dict":
        expression: RuntimeExpr = convert({1: 2 if same_sort else 2.0}, target)  # type: ignore[arg-type]
        assert expression.value == {i64(1): value}
        assert str(expression) == f"Map[i64, {value_type}].empty().put({value.value!r}, 1)"
        assert [argument.value for argument in _required_callable_args(expression)[1:]] == [value.value, 1]
        args = [
            RuntimeExpr(expression.__egg_ref__.owner.ref("nodes", index))
            for index in expression.__egg_ref__.read().kind.value.args
        ]
        assert args[1].value == 1
        assert args[2].value == value.value
    else:
        builder = Builder()
        key = builder.import_ref(i64(1).__egg_ref__)
        item = builder.import_ref(value.__egg_ref__)
        root = builder.add(
            "nodes",
            pb.Node(
                sort_id=builder.import_ref(target.__egg_sort__),
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(
                        value=Oneof[Literal["map"], pb.MapValue](
                            "map", pb.MapValue(entries=[pb.MapEntry(key=key, value=item)])
                        )
                    ),
                ),
            ),
        )
        expression = RuntimeExpr(builder.publish().ref("nodes", root))
        assert str(expression) == f"Map[i64, {value_type}].empty().put({value}, i64(1))"
        assert expression.value == {i64(1): value}


def test_map_converter_and_value_pretty_follow_renamed_views(vec_catalog_program, request):
    declarations = {d.kind.value.name: d for d in vec_catalog_program.declarations}
    declarations["Map"].kind.value.bindings.python.path[-1] = "Mapping"
    declarations["egglog.core.map.empty"].bindings.python.views[0].path = ["vacant"]
    declarations["egglog.core.map.insert"].bindings.python.views[0].path = ["put"]
    declarations["egglog.core.map.of"].bindings.python.views[0].path = ["from_entries"]
    declarations["egglog.core.map.get"].bindings.python.views[0].path = ["lookup"]
    request.getfixturevalue("vec_runtime")
    # This scoped catalog is not the module-global actual catalog.
    target = runtime._BASE_CLASSES["Map"][i64, f64]
    assert target.__egg_sort__ is not None
    expression: RuntimeExpr = convert({1: 2.0}, target)  # type: ignore[arg-type]
    assert str(expression) == "Mapping[i64, f64].vacant().put(1, 2.0)"
    mapping = expression.value
    assert isinstance(mapping, dict)
    value = mapping[i64(1)]
    assert isinstance(value, RuntimeExpr)
    assert value.value == 2.0
    from_entries = _required_runtime_function(target.from_entries)(Vec(Pair(i64(1), f64(2.0))))
    assert from_entries is not None
    formed_mapping = from_entries.value
    assert isinstance(formed_mapping, dict)
    formed_value = formed_mapping[i64(1)]
    assert isinstance(formed_value, RuntimeExpr)
    assert formed_value.value == 2.0
    assert ".from_entries(" in str(_required_runtime_function(from_entries.lookup)(i64(1)))
    assert str(_required_runtime_function(from_entries.lookup)(i64(1))).endswith(".lookup(1)")
    with pytest.raises(AttributeError):
        _ = target.from_pairs
    key, value = next(iter(_map_entries(expression)))
    builder = Builder()
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(target.__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["map"], pb.MapValue](
                        "map",
                        pb.MapValue(
                            entries=[pb.MapEntry(key=builder.import_ref(key), value=builder.import_ref(value))]
                        ),
                    )
                ),
            ),
        ),
    )
    assert str(RuntimeExpr(builder.publish().ref("nodes", root))) == "Mapping[i64, f64].vacant().put(i64(1), f64(2.0))"


@pytest.mark.parametrize("equality_output", [False, True])
def test_vec_map_callback_capture_core_order_and_binary_roundtrip(equality_output, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Vec.map authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)

    class Item(Expr):
        def __init__(self, value: i64) -> None: ...

    captured = i64(10)
    calls = []

    def callback(value):
        calls.append(1)
        result = value + captured
        return Item(result) if equality_output else result

    source = i64(3).range()
    result = source.map(callback)
    assert isinstance(result, Vec[Item] if equality_output else Vec[i64])
    reference = result.__egg_ref__
    call = reference.read().kind.value
    assert call.func == "egglog.core.vec.map"
    assert len(call.args) == 2
    function = reference.owner.ref("nodes", call.args[0])
    assert function.owner.ref("sorts", function.read().sort_id).read().kind.field == "func"
    assert StructuralView(reference.owner.ref("nodes", call.args[1])) == StructuralView(source.__egg_ref__)
    assert calls == [1]
    encoded = pack([reference])
    decoded = Builder.from_program(pb.Program.from_binary(encoded.program.to_binary())).publish()
    restored = RuntimeExpr(decoded.ref("nodes", encoded.indices[0]))
    assert StructuralView(restored.__egg_ref__) == StructuralView(reference)
    assert str(restored) == str(result)
    assert calls == [1]


@pytest.mark.parametrize("equality_output", [False, True])
def test_vec_map_executes_captured_callback_and_replays_source(equality_output):
    class Item(Expr):
        def __init__(self, value: i64) -> None: ...

    captured = i64(10)
    calls = []

    def callback(value):
        calls.append(1)
        result = value + captured
        return Item(result) if equality_output else result

    expression = i64(3).range().map(callback)
    expected = Vec(*(Item(i64(i)) if equality_output else i64(i) for i in (10, 11, 12)))
    graph = EGraph(save_egglog_string=True)
    replay = EGraph()
    try:
        # Eq-producing callbacks can mint constructors only in Full context.
        actual = graph.let("mapped", expression)
        graph.check(eq(actual).to(expected))
        if not equality_output:
            graph.check(eq(expression).to(expected))
        source = bindings._ProtoEngine.source(graph.as_egglog_string)
        submissions = 0
        while (submission := source.next_submission()) is not None:
            kind, encoded, _count = submission
            assert kind == "program"
            assert encoded is not None
            assert replay._state.handle is not None
            response, error = replay._state.transport.run(
                pb.RunProgramRequest(
                    egraph_id=replay._state.handle, program=pb.Program.from_binary(encoded)
                ).to_binary()
            )
            assert error is None
            assert pb.RunProgramResponse.from_binary(response).error is None
            submissions += 1
        assert submissions > 0
        assert calls == [1]
    finally:
        graph._state.destroy()
        replay._state.destroy()


def test_vec_generic_authoring_local_protocols_and_pretty(vec_runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Vec authoring/inspection accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    vector = vec_runtime
    symbolic, literal = i64(1) + 2, i64(3)
    expression = vector(symbolic, literal)
    assert isinstance(expression, vector[i64])
    assert [item.__egg_ref__ for item in expression.value] == [symbolic.__egg_ref__, literal.__egg_ref__]
    assert list(expression) == [symbolic, literal]
    assert len(expression) == 2
    assert symbolic in expression
    assert i64(4) not in expression
    with pytest.warns(DeprecationWarning, match="use .value"):
        assert expression.eval() == expression.value
    assert str(vector[i64].empty()) == "Vec[i64].empty()"
    assert vector[i64]().value == vector[i64].empty().value == ()
    assert str(vector(i64(1), i64(3))) == "Vec[i64](1, 3)"
    selected = vector(i64(1), i64(3))[0]
    assert isinstance(selected, i64)
    assert selected.__egg_ref__.read().kind.value.func == "egglog.core.vec.get"
    assert str(selected) == "Vec[i64](1, 3)[0]"
    with pytest.raises(ExprValueError):
        selected.value
    with pytest.raises(TypeError):
        vector[i64](args=(i64(1),))
    with pytest.raises(ConvertError):
        vector[i64](f64(1.0))
    with pytest.raises(TypeConstraintError):
        vector()

    ranged = i64(5).range()
    assert isinstance(ranged, vector[i64])
    assert str(ranged) == "i64(5).range()"
    with pytest.raises(ExprValueError):
        ranged.value
    selected = ranged[0]
    assert isinstance(selected, i64)
    assert str(selected) == "i64(5).range()[0]"
    reference = to_runtime_expr(ranged).__egg_ref__
    node = reference.read()
    assert node.kind is not None
    assert isinstance(node.kind.value, pb.Call)
    assert node.kind.value.func == "egglog.core.vec.range"
    assert len(node.kind.value.args) == 1
    argument = reference.owner.ref("nodes", node.kind.value.args[0])
    assert StructuralView(argument) == StructuralView(to_runtime_expr(i64(5)).__egg_ref__)
    assert StructuralView(reference.owner.ref("sorts", node.sort_id)) == StructuralView(vector[i64].__egg_sort__)
    packed = pack([reference])
    assert (
        sum(d.kind is not None and d.kind.value.name == "egglog.core.vec.range" for d in packed.program.declarations)
        == 1
    )
    decoded = Builder.from_program(pb.Program.from_binary(packed.program.to_binary())).publish()
    assert StructuralView(decoded.ref("nodes", packed.indices[0])) == StructuralView(reference)
    range_method = to_runtime_expr(i64(5)).range
    assert isinstance(range_method, RuntimeFunction)
    with pytest.raises(TypeError):
        range_method(1)
    with pytest.raises(AttributeError):
        to_runtime_expr(f64(5.0)).range


@pytest.mark.parametrize("source", [[1, 2], (1, 2), [], ()])
def test_vec_contextual_sequence_conversion_and_matching(vec_runtime, source):
    Vec = vec_runtime
    expression = convert(source, Vec[i64])
    assert tuple(item.value for item in expression.value) == tuple(source)
    assert get_type_args() == ()
    match expression:
        case Vec(children):
            assert tuple(item.value for item in children) == tuple(source)
        case _:
            pytest.fail("Vec value matching failed")
    nested = convert([[(1, 2.0)]], Vec[Vec[Pair[i64, f64]]])
    assert nested.value[0].value[0].value[1].value == 2.0


def test_vec_sequence_exact_target_precedence_and_context(vec_runtime):
    target = vec_runtime[i64]
    sentinel = target(i64(9))
    converter(object, target, lambda source: sentinel, cost=0)
    assert convert([1], target) is sentinel
    assert convert((1,), target) is sentinel
    with pytest.raises(ConvertError):
        convert([1], Pair[i64, i64])


def test_vec_like_annotation_context_and_element_conversion(vec_runtime):
    @function
    def size(value: VecLike[i64, int]) -> i64: ...

    expression = size([1, 2])
    argument = _required_callable_args(expression)[0]
    assert isinstance(argument, vec_runtime[i64])
    assert tuple(child.value for child in argument.value) == (1, 2)
    assert _required_callable_args(size(()))[0].value == ()
    assert get_type_args() == ()


def test_vec_annotation_protocol_preserves_lookup_and_pending_classes(vec_runtime):
    class Token(Expr):
        def __init__(self, value: i64) -> None: ...

    _runtime_token, token_context = _required_authored_class(Token)
    pending = token_context.prepare
    assert _runtime_token.__args__ == ()
    assert token_context.prepare is pending
    assert i64.__args__ == ()  # type: ignore[attr-defined]
    assert vec_runtime[i64].__args__ == (i64,)
    assert Pair[i64, f64].__args__ == (i64, f64)  # type: ignore[attr-defined]
    assert len(vec_runtime.__args__) == 1

    def annotated(value: Pair[i64, f64]) -> VecLike[i64, int]: ...

    hints = get_type_hints(annotated)
    assert hints["value"] == Pair[i64, f64]
    _: object
    with pytest.raises(TypeError):
        _ = VecLike[i64, int, float]  # type: ignore[misc, type-arg]

    def wrong(value: VecLike[int, int]) -> i64: ...  # type: ignore[type-var]

    with pytest.raises(TypeError, match="Type arguments must resolve"):
        get_type_hints(wrong)
    with pytest.raises(AttributeError, match="no generated Python member"):
        _ = vec_runtime[i64].missing
    with pytest.raises(AttributeError):
        _ = Token.missing  # type: ignore[attr-defined]
    assert token_context.prepare is None  # Ordinary lookup still prepares.


def test_vec_empty_conversion_uses_supplied_presentation(vec_catalog_program, request):
    empty = next(d for d in vec_catalog_program.declarations if d.kind.value.name == "egglog.core.vec.empty")
    empty.bindings.python.views[0].path = ["vacant"]
    Vec = request.getfixturevalue("vec_runtime")
    expression = convert([], Vec[i64])
    assert expression.value == ()
    assert str(expression) == "Vec[i64].vacant()"
    builder = Builder()
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Vec[i64].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["vec"], pb.ValueList]("vec", pb.ValueList()))
            ),
        ),
    )
    assert str(RuntimeExpr(builder.publish().ref("nodes", root))) == "Vec[i64].vacant()"


@pytest.mark.parametrize("same", [False, True])
def test_vec_children_preserve_union_identity(vec_runtime, same):
    union = eq(i64(1)).to(i64(2))
    left = RuntimeExpr(union.__egg_ref__)
    right = left if same else RuntimeExpr(eq(i64(1)).to(i64(2)).__egg_ref__)
    children = vec_runtime(left, right).value
    assert children[0].__egg_ref__ == left.__egg_ref__
    assert children[1].__egg_ref__ == right.__egg_ref__
    assert (expr_parts(children[0]) == expr_parts(children[1])) is same


@pytest.mark.parametrize("malformed", ["wrong_sort", "invalid_ref", "cycle", "nominal", "user_call"])
def test_vec_value_rejects_invalid_payload_and_provenance(vec_runtime, malformed):
    if malformed == "nominal":

        class NominalVec(Expr, egg_sort="Vec"):
            def __init__(self) -> None: ...

        with pytest.raises(AttributeError):
            _container_items(NominalVec())  # type: ignore[arg-type]
        return
    builder = Builder()
    child = builder.import_ref((f64(1.0) if malformed == "wrong_sort" else i64(1)).__egg_ref__)
    sort = builder.import_ref(vec_runtime[i64].__egg_sort__)
    if malformed == "invalid_ref":
        child = 100
    elif malformed == "cycle":
        child = 1
    kind = Oneof[Literal["primitive_value"], pb.PrimitiveValue](
        "primitive_value",
        pb.PrimitiveValue(value=Oneof[Literal["vec"], pb.ValueList]("vec", pb.ValueList(items=[child]))),
    )
    if malformed == "user_call":
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function", pb.Function(name="egglog.core.vec.of", output=sort)
                )
            ),
        )
        kind = Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.vec.of", args=[child]))
    root = builder.add("nodes", pb.Node(sort_id=sort, kind=kind))
    with pytest.raises((ValueError, IndexError, AttributeError)):
        RuntimeExpr(builder.publish().ref("nodes", root)).value


def test_vec_computed_value_is_not_locally_evaluated(vec_runtime):
    @function
    def computed() -> Vec[i64]: ...

    expression = computed()
    assert isinstance(expression, vec_runtime[i64])
    with pytest.raises(ExprValueError):
        expression.value
    selected = expression[0]
    assert isinstance(selected, i64)
    assert selected.__egg_ref__.read().kind.value.args[0] == 0

    @function
    def computed_pair() -> Pair[i64, f64]: ...

    pair = computed_pair()
    assert isinstance(pair, Pair[i64, f64])  # type: ignore[misc]
    assert resolve_callable(computed_pair).read().kind.field == "function"
    assert {
        d.kind.value.name if d.kind is not None else pytest.fail("Missing declaration kind")
        for d in pack([pair.__egg_ref__]).program.declarations
    } >= {"Pair", "i64", "f64"}
    with pytest.raises(ExprValueError):
        pair.value


@pytest.mark.parametrize("empty", [False, True])
def test_vec_value_payload_direct_children_and_pretty(vec_runtime, empty):
    builder = Builder()
    first = builder.import_ref(i64(1).__egg_ref__)
    assert first == 0
    second = builder.import_ref(i64(2).__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(vec_runtime[i64].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["vec"], pb.ValueList](
                        "vec", pb.ValueList(items=[] if empty else [first, second, first])
                    )
                ),
            ),
        ),
    )
    owner = builder.publish()
    value = RuntimeExpr(owner.ref("nodes", root))
    children = value.value
    assert isinstance(children, tuple)
    references = []
    for child in children:
        assert isinstance(child, RuntimeExpr)
        references.append(child.__egg_ref__)
    assert references == ([] if empty else [owner.ref("nodes", i) for i in [first, second, first]])
    assert str(value) == ("Vec[i64].empty()" if empty else "Vec[i64](i64(1), i64(2), i64(1))")
    assert get_callable_args(value) is None


def test_vec_getitem_pretty_uses_public_syntax(vec_runtime):
    assert str(vec_runtime[i64](i64(1))[0]) == "Vec[i64](1)[0]"


def test_pair_proposed_authoring_and_symbolic_inspection(pair_runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pair authoring accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    pair = pair_runtime
    left, right = i64(1) + 2, f64(3.0)
    expression = pair[i64, f64](right=right, left=left)
    children = expression.value
    assert children[0].__egg_ref__ == left.__egg_ref__
    assert children[1].__egg_ref__ == right.__egg_ref__
    assert expression.left.__egg_ref__.read().kind.value.func == "egglog.core.pair.first"
    assert expression.right.__egg_ref__.read().kind.value.func == "egglog.core.pair.second"
    assert str(expression.left).endswith(".left")
    assert pair.__match_args__ == ("value",)
    with pytest.raises(ExprValueError):
        expression.left.value
    with pytest.raises(TypeError):
        pair[i64, f64](i64(1))
    with pytest.raises(TypeError):
        pair[i64, f64](i64(1), f64(2), f64(3))
    with pytest.raises(TypeError):
        pair[i64, f64](left=i64(1), right=f64(2), unknown=3)
    with pytest.raises(ConvertError):
        pair[i64, f64](f64(1), f64(2))


def test_pair_tuple_conversion_context_exact_override_and_wrong_families(pair_runtime):
    pair = pair_runtime
    converted = convert((1, 2.5), pair[i64, f64])
    assert [child.value for child in converted.value] == [1, 2.5]
    assert [str(child) for child in converted.value] == ["i64(1)", "f64(2.5)"]
    assert get_type_args() == ()
    marker = pair[i64, f64](i64(9), f64(8))
    contexts = []

    def exact(value):
        contexts.append(get_type_args())
        return marker

    converter(tuple, pair[i64, f64], exact, cost=0)
    assert convert((1, 2), pair[i64, f64]) is marker
    assert contexts == [(i64, f64)]  # type: ignore[comparison-overlap]
    assert get_type_args() == ()
    # Unrelated families and nominal same-spelled sorts must not reuse Pair's
    # host conversion registration.
    with pytest.raises(ConvertError):
        convert((i64(1), f64(2.0)), Vec[i64])

    class NominalPair(Expr, egg_sort="Pair"): ...

    with pytest.raises(ConvertError):
        convert((i64(1), f64(2)), NominalPair)


@pytest.mark.parametrize("subclass", [False, True])
def test_pair_exact_target_precedes_generic_fallback_across_source_mro(pair_runtime, subclass):
    class TupleSubclass(tuple):
        __slots__ = ()

    target = pair_runtime[i64, f64]
    sentinel = target(i64(9), f64(8.0))
    calls = []

    def exact(value):
        calls.append((value, get_type_args()))
        return sentinel

    converter(object, target, exact)
    value = TupleSubclass((1, 2.0)) if subclass else (1, 2.0)
    assert convert(value, target) is sentinel
    assert calls == [(value, (i64, f64))]  # type: ignore[comparison-overlap]
    assert get_type_args() == ()


@pytest.mark.parametrize("value", [(), (1,), (1, 2, 3)])
def test_pair_tuple_width_is_checked_before_conversion(pair_runtime, value):
    with pytest.raises(ValueError, match="Expected a tuple of length 2"):
        convert(value, pair_runtime[i64, f64])
    assert get_type_args() == ()


def test_pair_nested_matching_and_retained_children(pair_runtime):
    Pair = pair_runtime
    expression = convert(((1, 2.0), 3), Pair[Pair[i64, f64], i64])
    match expression:
        case Pair((Pair((left, right)), last)):
            assert (left.value, right.value, last.value) == (1, 2.0, 3)
        case _:
            pytest.fail("Pair value matching failed")
    children = expression.value
    owner = weakref(expression.__egg_ref__.owner)
    child_owner = weakref(children[0].__egg_ref__.owner)
    del expression
    gc.collect()
    assert children[0].value[1].value == 2.0
    assert owner() is None  # Imported refs retain their actual child owner only.
    assert child_owner() is not None
    del children
    # Extracted pattern children above intentionally retain their own owners.
    del left, right, last
    gc.collect()
    assert child_owner() is None


@pytest.mark.parametrize("same", [False, True])
def test_pair_value_payload_preserves_child_refs_and_union_identity(pair_runtime, same):
    builder = Builder()
    scalar = builder.import_ref(i64.__egg_sort__)  # type: ignore[attr-defined]
    literal = builder.import_ref(i64(1).__egg_ref__)
    first = builder.add(
        "nodes", pb.Node(sort_id=scalar, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[literal])))
    )
    second = (
        first
        if same
        else builder.add(
            "nodes",
            pb.Node(sort_id=scalar, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[literal]))),
        )
    )
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(pair_runtime[i64, i64].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["pair"], pb.PairValue]("pair", pb.PairValue(first=first, second=second))
                ),
            ),
        ),
    )
    owner = builder.publish()
    expression = RuntimeExpr(owner.ref("nodes", root))
    pair_value = expression.value
    assert isinstance(pair_value, tuple)
    left, right = pair_value
    assert isinstance(left, RuntimeExpr)
    assert isinstance(right, RuntimeExpr)
    assert left.__egg_ref__ == owner.ref("nodes", first)
    assert right.__egg_ref__ == owner.ref("nodes", second)
    assert (left.__egg_ref__ == right.__egg_ref__) == same
    assert (StructuralView(left.__egg_ref__) == StructuralView(right.__egg_ref__)) == same
    authored = pair_runtime(left, right)
    assert authored.value[0].__egg_ref__ == left.__egg_ref__
    assert authored.value[1].__egg_ref__ == right.__egg_ref__
    encoded = pack([expression.__egg_ref__]).program
    assert any(
        (
            n.kind.field == "primitive_value"
            and (
                n.kind.value.value.field == "pair"
                if n.kind.value.value is not None
                else pytest.fail("Missing primitive value arm")
            )
        )
        if n.kind is not None
        else pytest.fail("Missing node kind")
        for n in encoded.nodes
    )


def test_pair_value_ref_zero_and_pretty_uses_direct_payload(pair_runtime):
    builder = Builder()
    first = builder.import_ref(i64(0).__egg_ref__)
    assert first == 0
    second = builder.import_ref(f64(2.0).__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(pair_runtime[i64, f64].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["pair"], pb.PairValue]("pair", pb.PairValue(first=first, second=second))
                ),
            ),
        ),
    )
    expression = RuntimeExpr(builder.publish().ref("nodes", root))
    children = expression.value
    assert isinstance(children, tuple)
    values = []
    for child in children:
        assert isinstance(child, RuntimeExpr)
        values.append(child.value)
    assert values == [0, 2.0]
    assert str(expression) == "Pair[i64, f64](i64(0), f64(2.0))"
    assert expression.__egg_ref__.read().kind.value.value.field == "pair"
    assert get_callable_args(expression) is None  # No fabricated make Call.


@pytest.mark.parametrize("malformed", ["missing_first", "missing_second", "wrong_sort", "invalid_ref", "cycle"])
def test_pair_value_rejects_invalid_local_payloads(pair_runtime, malformed):
    builder = Builder()
    first = builder.import_ref(i64(1).__egg_ref__)
    second = builder.import_ref(f64(2.0).__egg_ref__)
    value = pb.PairValue(first=first, second=second)
    if malformed.startswith("missing_"):
        value.clear_field(malformed.removeprefix("missing_"))
    elif malformed == "wrong_sort":
        value.second = first
    elif malformed == "invalid_ref":
        value.first = 100
    else:
        value.first = 2  # The node being authored, not a well-typed i64 child.
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(pair_runtime[i64, f64].__egg_sort__),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["pair"], pb.PairValue]("pair", value))
            ),
        ),
    )
    with pytest.raises((ValueError, IndexError)):
        RuntimeExpr(builder.publish().ref("nodes", root)).value


def test_pair_value_rejects_nominal_and_user_call_provenance(pair_runtime):
    class NominalPair(Expr, egg_sort="Pair"):
        @method(egg_fn="pair")
        def __init__(self, left: i64, right: f64) -> None: ...

    with pytest.raises(ExprValueError):
        _pair_value(NominalPair(1, 2.0))  # type: ignore[arg-type]
    builder = Builder()
    output = builder.import_ref(pair_runtime[i64, f64].__egg_sort__)
    builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["constructor"], pb.Constructor](
                "constructor", pb.Constructor(name="egglog.core.pair.make", output=output)
            )
        ),
    )
    root = builder.add(
        "nodes",
        pb.Node(sort_id=output, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.pair.make"))),
    )
    with pytest.raises(ExprValueError):
        _pair_value(RuntimeExpr(builder.publish().ref("nodes", root)))


def test_pair_contextual_tuple_conversion_uses_generic_registration():
    # A custom converter isolates generic family lookup from initializer views.
    target = Pair[i64, f64]
    builder = Builder()
    node = builder.add(
        "nodes",
        pb.Node(sort_id=builder.import_ref(target.__egg_sort__), kind=Oneof[Literal["var"], str]("var", "pair")),  # type: ignore[attr-defined]
    )
    sentinel = RuntimeExpr(builder.publish().ref("nodes", node))
    converter(tuple, Pair, lambda _: sentinel, cost=0)
    result = convert((i64(1), f64(2.0)), Pair[i64, f64])
    assert to_runtime_expr(result) is sentinel


@pytest.mark.parametrize("doc", [None, "", "function documentation"])
def test_function_source_records_defaults_and_metadata(doc, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Function authoring accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    default = i64(1)

    def score(x: i64, offset: i64 = default) -> i64: ...

    score.__doc__ = doc
    score = function(egg_fn="exact score")(score)
    default.__egg_ref__ = i64(2).__egg_ref__
    assert score.__module__ == __name__
    assert type(score).__module__ == "egglog.runtime"
    assert score.__doc__ == doc
    assert str(score) == repr(score) == "score"
    reference = resolve_callable(score)
    record = reference.read()
    assert record.doc == (doc or "")
    assert record.has_field("doc") == (doc is not None)
    assert record.kind.field == "function"
    assert record.kind.value.name == "exact score"
    expression = score(1)  # type: ignore[arg-type]
    assert str(expression) == "score(1)"
    assert str(score(offset=3, x=1)) == "score(1, 3)"  # type: ignore[arg-type]
    assert str(set_(score(1, 2)).to(7)) == "set_(score(1)).to(i64(7))"  # type: ignore[arg-type]
    assert str(set_(score(1, 3)).to(7)) == "set_(score(1, 3)).to(i64(7))"  # type: ignore[arg-type]
    default.__egg_ref__ = i64(9).__egg_ref__
    assert [a.value for a in _required_callable_args(score(1))] == [1, 2]  # type: ignore[arg-type]
    recovered = get_callable_fn(expression)
    assert recovered.__module__ == __name__
    assert recovered.__doc__ == doc
    assert recovered == score
    assert get_callable_args(expression, score) is not None
    assert _required_callable_args(score(offset=2, x=1))[1].value == 2
    with pytest.raises(TypeError):
        score()
    with pytest.raises(TypeError):
        score(1, x=2)
    with pytest.raises(TypeError):
        score(1, unknown=2)


@pytest.mark.parametrize("doc", [None, "", "class and initializer documentation"])
def test_class_and_initializer_documentation_presence(doc):
    class Documented(Expr):
        __doc__ = doc

        def __init__(self, value: i64) -> None: ...

        __init__.__doc__ = doc

    assert Documented.__doc__ == doc
    runtime_documented, _documented_context = _required_authored_class(Documented)
    definition = runtime_documented.__egg_definition__.read()
    assert definition.has_field("doc") == (doc is not None)
    initializer = resolve_callable(Documented)
    assert initializer.read().has_field("doc") == (doc is not None)
    assert RuntimeFunction(initializer, 0, runtime_documented).__doc__ == doc


@pytest.mark.parametrize("name", ["", "__egglog_lookup_shared"])
@pytest.mark.parametrize("later", ["none", "distinct", "conflicting"])
def test_owned_definition_lookup_keeps_exact_refs_without_detaching(monkeypatch, name, later):
    requested = pb.Program(
        sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
        declarations=[
            pb.Declaration(
                kind=Oneof[Literal["relation"], pb.Relation]("relation", pb.Relation(name=name)),
                doc="retained documentation",
            ),
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name=name)
                )
            ),
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name=name))),
        ],
        nodes=[pb.Node(kind=Oneof[Literal["var"], str]("var", "origin"))],
    )
    original = Builder.from_program(requested).publish()
    origin = original.ref("nodes", 0)
    expected = {
        namespace: original.ref("declarations", index)
        for index, namespace in enumerate(("callable", "host_sort_family", "eq_sort"))
    }
    _required_oneof(requested.declarations[0].kind, "relation", pb.Relation).name = "changed caller input"
    detached = expected["callable"].read()
    detached.kind.value.name = "changed detached record"
    detached.doc = "changed detached documentation"
    assert expected["callable"].read().doc == "retained documentation"
    missing = "__egglog_lookup_later"
    with pytest.raises(LookupError, match="No canonical definition"):
        runtime._definition_at(origin, "callable", missing)

    extended = Builder()
    for reference in expected.values():
        extended.import_ref(reference)
    # Repeated imported slots still name one canonical definition. The import
    # cache is disposable, as in test_import_cache_is_not_authoritative.
    extended._imports.clear()
    extended.import_ref(expected["callable"])
    if later != "none":
        added = extended.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["relation"], pb.Relation](
                    "relation", pb.Relation(name=name if later == "conflicting" else missing)
                )
            ),
        )
    scalar = extended.import_ref(original.ref("sorts", 0))
    query_index = extended.add("nodes", pb.Node(sort_id=scalar, kind=Oneof[Literal["var"], str]("var", "new origin")))
    extended_owner = extended.publish()
    query = extended_owner.ref("nodes", query_index)
    fallback = builtin_catalog().definitions["host_sort_family", "i64"]

    def forbidden_read(self):
        pytest.fail("Definition key lookup detached a full published record")

    with monkeypatch.context() as patch:
        patch.setattr(Ref, "read", forbidden_read)
        for namespace, reference in expected.items():
            assert runtime._definition_at(origin, namespace, name) == reference
            if later == "conflicting" and namespace == "callable":
                with pytest.raises(ValueError, match="Ambiguous local definition"):
                    runtime._definition_at(query, namespace, name)
            else:
                assert runtime._definition_at(query, namespace, name) == reference
        assert runtime._definition_at(query, "host_sort_family", "i64") == fallback
        if later == "distinct":
            assert runtime._definition_at(query, "callable", missing) == extended_owner.ref("declarations", added)
        else:
            with pytest.raises(LookupError, match="No canonical definition"):
                runtime._definition_at(query, "callable", missing)
    assert expected["callable"].read().kind.value.name == name


def test_free_function_identity_uses_python_name_not_signature_or_owner():
    class Box(Expr): ...

    def first(x: i64 = i64(1)) -> i64: ...

    def second(x: f64 = f64(2.0)) -> Box: ...

    second.__name__ = first.__name__
    left, right = function(first), function(egg_fn="different native name")(second)
    runtime_left, runtime_right = _required_runtime_function(left), _required_runtime_function(right)
    assert runtime_left == runtime_right
    assert hash(runtime_left) == hash(runtime_right)
    assert {runtime_left: "same callable"}[runtime_right] == "same callable"
    assert _required_callable_args(left(), right)[0].value == 1
    second.__module__ = "different_module"
    other_module = function(second)
    assert runtime_left != _required_runtime_function(other_module)
    first.__name__ = "different_name"
    assert runtime_left != _required_runtime_function(function(first))


def test_member_deconstruction_filter_ignores_receiver_binding():
    value = i64(1)
    expression = value + 2
    # Numeric special methods are Python forwarders, not callable records.
    assert get_callable_args(expression, value.__add__) is None
    bound_member = RuntimeExpr.__getattr__(to_runtime_expr(value), "__add__")
    assert get_callable_fn(expression) != bound_member
    for member in (bound_member, i64.__add__):
        children = get_callable_args(expression, member)
        assert children is not None
        assert [child.value for child in children] == [1, 2]


@pytest.mark.parametrize("raises", [False, True])
def test_runtime_function_comparison_calls_identical_bound_expression_hook(raises):
    calls = []

    class Token(Expr):
        @method(preserve=True)
        def __eq__(self, other) -> bool:  # type: ignore[override]
            calls.append(1)
            if raises:
                msg = "bound equality marker"
                raise ValueError(msg)
            return False

    token = to_runtime_expr(constant("token", Token))
    member = i64.__add__
    runtime_member = _required_runtime_function(member)
    # Comparison only: deliberately do not invoke an i64 member with Token.
    left = RuntimeFunction(runtime_member.__egg_ref__, runtime_member.__egg_view__, token)
    right = RuntimeFunction(runtime_member.__egg_ref__, runtime_member.__egg_view__, token)
    if raises:
        with pytest.raises(ValueError, match="bound equality marker"):
            assert left == right
    else:
        assert (left == right) is False
    assert calls == [1]


def test_function_late_local_annotations_and_concurrent_materialization():
    def make():
        @function
        def leaf() -> "Later": ...

        class Later(Expr): ...

        return leaf, Later

    leaf, later = make()
    calls = []
    prepare = leaf.__egg_prepare__

    def delayed_prepare():
        calls.append(1)
        sleep(0.02)
        return prepare()

    leaf.__egg_prepare__ = delayed_prepare
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: leaf(), range(8)))
    assert calls == [1]
    assert all(isinstance(value, later) for value in values)
    assert all(
        _required_runtime_function(get_callable_fn(value)).__egg_ref__ == resolve_callable(leaf) for value in values
    )
    assert leaf.__egg_prepare__ is None


def test_function_annotation_locals_shadow_globals():
    i64 = f64  # noqa: F841 -- get_type_hints resolves the quoted local name

    @function
    def measure(value: "i64") -> "i64": ...  # type: ignore[valid-type]

    expression = measure(1.5)
    assert isinstance(expression, f64)
    assert _required_callable_args(expression)[0].value == 1.5


def test_bodyless_none_and_docstring_do_not_execute_python():
    @function
    def plain() -> i64:
        return None  # type: ignore[return-value]

    @function
    def documented() -> f64:
        """Only documentation."""

    assert resolve_callable(plain).read().kind.field == "function"
    assert resolve_callable(documented).read().kind.field == "function"
    assert str(plain()) == "plain()"
    assert documented.__doc__ == "Only documentation."


def test_named_eager_body_is_authored_once_without_engine():
    calls = 0

    @function
    def body() -> i64:
        nonlocal calls
        calls += 1
        return i64(1)

    assert calls == 0
    for _ in range(2):
        assert str(body()) == "body()"
    assert calls == 1
    runtime_body = _required_runtime_function(body)
    assert runtime_body.__egg_prepare__ is None
    declaration = resolve_callable(body)
    assert declaration.read().kind.field == "primitive"
    assert declaration.owner.ref("nodes", declaration.read().kind.value.body).read().kind.value.value.value == 1

    @function(cost=1)  # type: ignore[type-var]
    def invalid() -> i64: ...

    with pytest.raises(ValueError, match="Primitive-returning callables cannot use cost"):
        invalid()


def test_named_eager_binders_keep_slots_and_callee_closure():
    @function
    def seed(value: i64) -> i64:
        return value

    @function
    def inner(second: i64, first: i64 = seed(i64(7))) -> i64:
        return second + first

    @function
    def outer(unused: i64, second: i64, first: i64) -> i64:
        return inner(first, second) + inner(second)

    reference = resolve_callable(outer)
    primitive = reference.read().kind.value
    assert isinstance(primitive, pb.Primitive)
    assert [argument.name for argument in primitive.inputs] == ["unused", "second", "first"]
    body = RuntimeExpr(reference.owner.ref("nodes", primitive.body))
    assert str(body) == "inner(_2, _1) + inner(_1)"
    inner_reference = resolve_callable(inner)
    before = pack([inner_reference]).program.to_binary()
    program = pack([reference]).program
    inner_copy = next(
        declaration
        for declaration in program.declarations
        if declaration.kind is not None and declaration.kind.value.name == inner_reference.read().kind.value.name
    )
    assert inner_copy.kind is not None
    assert isinstance(inner_copy.kind.value, pb.Primitive)
    assert pack([inner_reference]).program.to_binary() == before
    inner_body = RuntimeExpr(inner_reference.owner.ref("nodes", inner_reference.read().kind.value.body))
    assert str(inner_body) == "_0 + _1"
    assert inner_copy.bindings is not None
    assert inner_copy.bindings.python is not None
    default = inner_copy.bindings.python.views[0].params[1]
    assert default.has_field("default_expr")
    default_kind = program.nodes[default.default_expr].kind
    assert default_kind is not None
    assert isinstance(default_kind.value, pb.Call)
    assert default_kind.value.func == resolve_callable(seed).read().kind.value.name
    assert inner_reference in {
        reference.owner.ref("declarations", index) for index in range(len(reference.owner._slots["declarations"]))
    }


@pytest.mark.parametrize("spelling", ["_0", "value", "external"])
@pytest.mark.parametrize("aliased", [False, True])
def test_named_eager_body_rejects_free_variable_by_identity(spelling, aliased):
    external = var("shown" if aliased else spelling, i64, egg_name=spelling if aliased else None)

    @function
    def body(value: i64) -> i64:
        return value + external

    with pytest.raises(ValueError, match=r"Free variable.*named eager body"):
        resolve_callable(body)
    assert cast("RuntimeFunction", body).__egg_declaration__ is None


def test_named_eager_partial_call_captures_bind_formals_only():
    @function
    def add(left: i64, right: i64) -> i64:
        return left + right

    @function
    def capture(unused: i64, value: i64) -> UnstableFn[i64, i64]:
        return UnstableFn(add, value)

    reference = resolve_callable(capture)
    body = reference.owner.ref("nodes", reference.read().kind.value.body)
    value = RuntimeExpr(body)
    assert str(value) == "partial(add, _1)"
    assert body.read().kind.value.value.field == "partial_call"
    partial_value = get_literal_value(value)
    assert isinstance(partial_value, partial)
    assert partial_value.func == add
    assert pack([reference]).program.to_binary()


@pytest.mark.parametrize("outcome", ["none", "wrong", "raise", "recursive"])
def test_named_eager_body_failure_publishes_no_declaration(outcome):
    calls = []

    def body(value: i64) -> object:
        calls.append(value)
        if outcome == "none":
            return None
        if outcome == "wrong":
            return String("wrong")
        if outcome == "recursive":
            return wrapped(value)
        msg = "body marker"
        raise RuntimeError(msg)

    body.__annotations__["return"] = i64
    wrapped = cast("RuntimeFunction", function(body))
    error, message = {
        "none": (ValueError, "Callable bodies must return a value"),
        "wrong": (ConvertError, "Cannot convert"),
        "raise": (RuntimeError, "body marker"),
        "recursive": (ValueError, "Recursively resolving"),
    }[outcome]
    for _ in range(2):
        with pytest.raises(error, match=message):
            resolve_callable(wrapped)
    assert len(calls) == 1
    assert wrapped.__egg_declaration__ is None
    assert wrapped.__egg_prepare__ is None


def test_named_eager_eqsort_result_and_ambient_ruleset_isolation():
    class Box(Expr):
        def __init__(self, value: i64) -> None: ...

    calls = []
    ambient = ruleset()

    @function
    def body(value: i64) -> Box:
        calls.append(get_current_ruleset())
        rewrite(Box(value)).to(Box(value + 1))
        return Box(value)

    with set_current_ruleset(ambient):
        result = body(i64(2))
        assert get_current_ruleset() is ambient
    assert calls == [None]
    assert isinstance(result, Box)
    assert resolve_callable(body).read().kind.field == "primitive"
    assert not ambient._snapshot().read().kind.value.rules

    def empty(value: i64): ...

    empty.__annotations__["return"] = Box
    assert resolve_callable(function(empty)).read().kind.field == "constructor"


def test_named_eager_result_conversion_has_no_ambient_ruleset():
    class Box(Expr):
        def __init__(self, value: i64) -> None: ...

    class Source:
        pass

    calls = []

    def convert_source(value):
        calls.append(get_current_ruleset())
        rewrite(Box(i64(1))).to(Box(i64(2)))
        return Box(i64(3))

    converter(Source, Box, convert_source)

    def body() -> object:
        calls.append(get_current_ruleset())
        return Source()

    body.__annotations__["return"] = Box
    wrapped = function(body)
    ambient = ruleset()
    with set_current_ruleset(ambient):
        resolve_callable(wrapped)
        assert get_current_ruleset() is ambient
    assert calls == [None, None]
    assert not ambient._snapshot().read().kind.value.rules


@pytest.mark.parametrize("subsume_body", [False, True])
def test_free_ruleset_body_is_a_deferred_constructor_rewrite(subsume_body):
    class FreeRewriteBox(Expr, egg_sort=f"FreeRewriteBox_{subsume_body}"):
        def __init__(self, value: i64) -> None: ...

    destination, ambient = ruleset(), ruleset()
    calls = []

    @function(ruleset=destination, subsume=subsume_body)
    def body(unused: i64, value: i64 = i64(4)) -> FreeRewriteBox:
        calls.append((get_current_ruleset(), str(unused), str(value)))
        return FreeRewriteBox(value)

    assert not destination._snapshot().read().kind.value.rules
    with set_current_ruleset(ambient):
        expression = body(i64(9))
        assert get_current_ruleset() is ambient
    assert calls == [(destination, "unused", "value")]
    assert resolve_callable(body).read().kind.field == "constructor"
    arguments = get_callable_args(expression)
    assert arguments is not None
    assert [argument.value for argument in arguments] == [9, 4]
    body(i64(8), i64(5))
    assert len(calls) == 1
    snapshot = destination._snapshot()
    assert len(snapshot.read().kind.value.rules) == 1
    assert not ambient._snapshot().read().kind.value.rules
    reference = snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])
    record = reference.read().kind.value
    assert record.subsume is subsume_body
    lhs = reference.owner.ref("nodes", record.lhs)
    assert [lhs.owner.ref("nodes", argument).read().kind.value for argument in lhs.read().kind.value.args] == [
        "_0",
        "_1",
    ]
    rhs = RuntimeExpr(reference.owner.ref("nodes", record.rhs))
    rhs_arguments = get_callable_args(rhs, FreeRewriteBox)
    assert rhs_arguments is not None
    assert rhs_arguments[0].__egg_ref__.read().kind.value == "_1"
    encoded = pack(commands=[snapshot]).program.to_binary()
    program = pb.Program.from_binary(encoded)
    assert program.to_binary() == encoded
    assert {
        declaration.kind.value.name
        for declaration in program.declarations
        if declaration.kind is not None and declaration.kind.field == "constructor"
    } == {resolve_callable(body).read().kind.value.name, resolve_callable(FreeRewriteBox).read().kind.value.name}


@pytest.mark.parametrize(("name", "expected"), [("value", "_value"), ("0", "_0"), ("_0", "__0")])
def test_free_ruleset_body_preserves_original_positional_variable_relation(name, expected):
    class ExternalRewriteBox(Expr, egg_sort=f"ExternalRewriteBox_{name}"):
        def __init__(self, value: i64) -> None: ...

    @function
    def inner(value: i64) -> i64:
        return value + 1

    external = var(name, i64)
    destination = ruleset()

    @function(ruleset=destination)
    def body(value: i64) -> ExternalRewriteBox:
        return ExternalRewriteBox(inner(value) + external)

    body(i64(2))
    snapshot = destination._snapshot()
    reference = snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])
    record = reference.read().kind.value
    rhs = RuntimeExpr(reference.owner.ref("nodes", record.rhs))
    rhs_arguments = get_callable_args(rhs, ExternalRewriteBox)
    assert rhs_arguments is not None
    sum_arguments = get_callable_args(rhs_arguments[0])
    assert sum_arguments is not None
    call, free = sum_arguments
    call_arguments = get_callable_args(call, inner)
    assert call_arguments is not None
    assert call_arguments[0].__egg_ref__.read().kind.value == "_0"
    assert free.__egg_ref__.read().kind.value == expected
    program = pack(commands=[snapshot]).program
    declaration = next(
        item.kind.value
        for item in program.declarations
        if item.kind is not None and isinstance(item.kind.value, pb.Primitive)
    )
    inner_body = program.nodes[declaration.body]
    assert inner_body.kind is not None
    assert isinstance(inner_body.kind.value, pb.Call)
    assert program.nodes[inner_body.kind.value.args[0]].kind == Oneof("var", "_0")
    assert to_runtime_expr(external).__egg_ref__.read().kind.value == name


def test_free_ruleset_body_lhs_keeps_original_wrapper_binders():
    class ReplacedRewriteBox(Expr):
        def __init__(self, value: i64) -> None: ...

    destination = ruleset()

    @function(ruleset=destination)
    def body(value: i64) -> ReplacedRewriteBox:
        value.__replace_expr__(i64(7))
        return ReplacedRewriteBox(value)

    body(i64(2))
    snapshot = destination._snapshot()
    reference = snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])
    record = reference.read().kind.value
    lhs = RuntimeExpr(reference.owner.ref("nodes", record.lhs))
    lhs_arguments = get_callable_args(lhs, body)
    assert lhs_arguments is not None
    assert lhs_arguments[0].__egg_ref__.read().kind.value == "_0"
    rhs = RuntimeExpr(reference.owner.ref("nodes", record.rhs))
    rhs_arguments = get_callable_args(rhs, ReplacedRewriteBox)
    assert rhs_arguments is not None
    assert rhs_arguments[0].value == 7


def test_free_ruleset_body_result_conversion_restores_outer_context():
    class ConvertedRewriteBox(Expr):
        def __init__(self, value: i64) -> None: ...

    class Source:
        pass

    destination, ambient = ruleset(), ruleset()
    calls = []

    def convert_source(value):
        calls.append(get_current_ruleset())
        return ConvertedRewriteBox(i64(3))

    converter(Source, ConvertedRewriteBox, convert_source)

    def body() -> object:
        calls.append(get_current_ruleset())
        return Source()

    body.__annotations__["return"] = ConvertedRewriteBox
    wrapped = function(ruleset=destination)(cast("Callable[[], ConvertedRewriteBox]", body))
    with set_current_ruleset(ambient):
        resolve_callable(wrapped)
        assert get_current_ruleset() is ambient
    assert calls == [destination, ambient]
    assert len(destination._snapshot().read().kind.value.rules) == 1
    assert not ambient._snapshot().read().kind.value.rules


@pytest.mark.parametrize("outcome", ["none", "wrong", "raise", "recursive"])
def test_free_ruleset_body_failure_is_cached_without_default_publication(outcome):
    class FailedRewriteBox(Expr, egg_sort=f"FailedRewriteBox_{outcome}"):
        def __init__(self, value: i64) -> None: ...

    destination, ambient = ruleset(), ruleset()
    calls = []

    def body(value: i64) -> object:
        calls.append(get_current_ruleset())
        rewrite(FailedRewriteBox(i64(1)), ruleset=destination).to(FailedRewriteBox(i64(2)))
        if outcome == "none":
            return None
        if outcome == "wrong":
            return String("wrong")
        if outcome == "recursive":
            return wrapped(value)
        msg = "rewrite body marker"
        raise RuntimeError(msg)

    body.__annotations__["return"] = FailedRewriteBox
    wrapped = cast("RuntimeFunction", function(ruleset=destination)(cast("Callable[[i64], FailedRewriteBox]", body)))
    error, message = {
        "none": (ValueError, "Callable bodies must return a value"),
        "wrong": (ConvertError, "Cannot convert"),
        "raise": (RuntimeError, "rewrite body marker"),
        "recursive": (ValueError, "Recursively resolving"),
    }[outcome]
    with set_current_ruleset(ambient):
        for _ in range(2):
            with pytest.raises(error, match=message):
                resolve_callable(wrapped)
            assert get_current_ruleset() is ambient
    assert calls == [destination]
    assert wrapped.__egg_declaration__ is None
    assert wrapped.__egg_prepare__ is None
    assert len(destination._snapshot().read().kind.value.rules) == 1
    assert not ambient._snapshot().read().kind.value.rules


def test_free_ruleset_body_invalid_destination_precedes_callback():
    class InvalidRulesetBox(Expr): ...

    @function(ruleset=cast("Ruleset", object()))
    def body() -> InvalidRulesetBox:
        pytest.fail("Invalid destination executed the body")

    with pytest.raises(TypeError, match="Rewrite destinations must be Rulesets"):
        resolve_callable(body)


@pytest.mark.parametrize("options", [{"cost": 1}, {"unextractable": True}])
def test_free_ruleset_body_preserves_constructor_options(options):
    class DeferredRulesetBox(Expr):
        def __init__(self) -> None: ...

    destination = ruleset()
    calls = []

    @function(ruleset=destination, **options)
    def body() -> DeferredRulesetBox:
        calls.append(get_current_ruleset())
        return DeferredRulesetBox()

    reference = resolve_callable(body)
    record = reference.read()
    assert record.kind.field == "constructor"
    assert record.kind.value.unextractable is options.get("unextractable", False)
    if "cost" in options:
        assert record.kind.value.has_field("cost")
        cost = reference.owner.ref("nodes", record.kind.value.cost).read()
        assert cost.kind.value.value == Oneof[Literal["i64"], int]("i64", options["cost"])
    else:
        assert not record.kind.value.has_field("cost")
    assert calls == [destination]
    assert len(destination._snapshot().read().kind.value.rules) == 1
    expression = body()
    graph = EGraph(expression)
    graph.run(destination)
    graph.check(eq(expression).to(DeferredRulesetBox()))
    assert resolve_callable(body) is reference
    assert calls == [destination]


@pytest.mark.parametrize(
    ("equality", "options", "error", "message"),
    [
        (
            False,
            {"ruleset": object()},
            ValueError,
            "Primitive-returning callables with bodies cannot use an explicit ruleset",
        ),
        (
            False,
            {"merge": lambda old, new: old},
            ValueError,
            "Primitive-returning callables with bodies cannot use merge",
        ),
        (False, {"cost": 1}, ValueError, "Primitive-returning callables cannot use cost"),
        (True, {"ruleset": object()}, TypeError, "Rewrite destinations must be Rulesets"),
        (True, {"cost": 1}, ValueError, "Eqsort-returning eager bodies cannot use cost"),
        (True, {"unextractable": True}, ValueError, "Eqsort-returning eager bodies cannot be unextractable"),
    ],
)
def test_named_eager_static_options_precede_callback(equality, options, error, message):
    class Box(Expr): ...

    def body(value: i64):
        pytest.fail("Invalid or deferred body was executed")

    body.__annotations__["return"] = Box if equality else i64
    wrapped = function(**options)(body)
    with pytest.raises(error, match=message):
        resolve_callable(wrapped)
    assert wrapped.__egg_declaration__ is None


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"unextractable": True}, ValueError, "Primitive-returning callables cannot be unextractable"),
        ({"ruleset": object()}, ValueError, "Explicit rulesets require a body"),
        ({"subsume": True}, ValueError, "Primitive-returning callables cannot use subsume"),
        ({"builtin": True, "merge": lambda old, new: old}, ValueError, "Builtin callables cannot use merge"),
        ({"builtin": True, "ruleset": object()}, ValueError, "Builtin callables cannot use an explicit ruleset"),
        ({"mutates_first_arg": True}, ValueError, "must be annotated to return None"),
    ],
)
def test_free_function_options_do_not_silently_drop_behavior(options, error, message):
    @function(**options)
    def unsupported(x: i64) -> i64: ...

    with pytest.raises(error, match=message):
        unsupported(1)
    assert unsupported.__egg_declaration__ is None
    assert unsupported.__egg_prepare__ is None


def test_free_function_invalid_parameter_kinds_and_unknown_options():
    @function
    def positional(x: i64, /) -> i64: ...

    @function
    def keyword(*, x: i64) -> i64: ...

    @function
    def variadic(*values: i64) -> i64: ...

    for value in (positional, keyword):
        with pytest.raises(ValueError, match="positional or keyword args"):
            resolve_callable(value)
    with pytest.raises(NotImplementedError, match="variadic"):
        variadic(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unexpected keyword"):
        function(unknown_option=True)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="either a function or keyword options"):
        function(positional, unknown_option=True)  # type: ignore[call-overload]


def test_function_source_recursion_and_terminal_failure_release_callback():
    calls = []

    def factory():
        def prepare():
            calls.append(1)
            return resolve_callable(value)

        value = RuntimeFunction(prepare, 0)
        return value

    recursive = factory()
    for _ in range(2):
        with pytest.raises(ValueError, match="Recursively resolving function"):
            resolve_callable(recursive)
    assert calls == [1]
    assert recursive.__egg_prepare__ is None
    assert recursive.__egg_declaration__ is None
    observed = weakref(recursive)
    del recursive
    gc.collect()
    assert observed() is None


def test_free_function_retains_local_output_hooks_then_collects():
    def factory():
        class Box(Expr):
            @method(preserve=True)
            def tag(self):
                return "local output"

        @function
        def leaf(value: i64 = i64(1)) -> Box: ...

        expression = leaf()
        assert resolve_callable(leaf).read().kind.field == "constructor"
        return expression, weakref(Box), weakref(resolve_callable(leaf).owner), weakref(leaf)

    expression, cls, owner, callable_ref = factory()
    _CLASS_CACHE.clear()
    gc.collect()
    assert expression.tag() == "local output"
    assert get_callable_fn(expression).__doc__ is None
    assert cls() is not None
    assert owner() is not None
    del expression
    gc.collect()
    assert cls() is None
    assert owner() is None
    assert callable_ref() is None


def test_public_union_action_freezes_canonical_operands_and_conversion(class_authoring_without_engine):
    class UnionNode(Expr):
        def __init__(self, value: i64) -> None: ...

    converted = []

    def from_int(value):
        converted.append(value)
        return UnionNode(value)

    converter(int, UnionNode, from_int)
    lhs = UnionNode(1)  # type: ignore[arg-type]
    action = union(lhs).with_(2)  # type: ignore[arg-type]
    assert converted == [2]
    command = action.__egg_ref__.read()
    assert command.kind.field == "action"
    assert command.kind.value.kind.field == "term"
    root = action.__egg_ref__.owner.ref("nodes", command.kind.value.kind.value)
    node = root.read()
    assert node.kind.field == "union"
    assert len(node.kind.value.members) == 2
    runtime_union_node, _union_context = _required_authored_class(UnionNode)
    union_sort = runtime_union_node.__egg_sort__
    assert union_sort is not None
    assert StructuralView(root.owner.ref("sorts", node.sort_id)) == StructuralView(union_sort)
    for index in node.kind.value.members:
        assert root.owner.ref("nodes", index).read().sort_id == node.sort_id
    lhs.__replace_expr__(UnionNode(9))  # type: ignore[arg-type]
    assert str(action) == "union(UnionNode(1)).with_(UnionNode(2))"
    assert str(union(lhs)) == "union(UnionNode(9))"
    with pytest.raises(ConvertError):
        union(lhs).with_(String("wrong sort"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unhashable type: 'Action'"):
        hash(action)
    packed = pack(commands=[action.__egg_ref__]).program
    assert pb.Program.from_binary(packed.to_binary()) == packed


def test_public_union_reuse_and_separate_constructions_retain_identity(class_authoring_without_engine):
    from egglog._program import _rule_key  # noqa: PLC0415

    class UnionNode(Expr):
        def __init__(self, value: i64) -> None: ...

    lhs, rhs = UnionNode(1), UnionNode(2)  # type: ignore[arg-type]
    action = union(lhs).with_(rhs)
    wrapped, cloned = Action(action.__egg_ref__), copy(action)
    separate = union(lhs).with_(rhs)
    assert action == wrapped == cloned
    assert action == separate
    assert cloned.__egg_ref__ == wrapped.__egg_ref__ == action.__egg_ref__
    packed = pack(commands=[item.__egg_ref__ for item in (action, action, cloned, separate)]).program
    roots = [
        _required_oneof(_required_oneof(command.kind, "action", pb.Action).kind, "term", int)
        for command in packed.commands
    ]
    assert roots[0] == roots[1] == roots[2]
    assert roots[0] != roots[3]
    authored = rule().then(action, action, cloned, separate)
    head = authored.__egg_ref__.read().kind.value.head
    assert head[0].kind.value == head[1].kind.value == head[2].kind.value
    assert head[0].kind.value != head[3].kind.value
    assert _rule_key(rule().then(action).__egg_ref__) == _rule_key(rule().then(separate).__egg_ref__)
    assert hash(_rule_key(rule().then(action).__egg_ref__)) == hash(_rule_key(rule().then(separate).__egg_ref__))
    assert _rule_key(rule().then(action, action).__egg_ref__) != _rule_key(rule().then(action, separate).__egg_ref__)


def test_public_union_contextual_rule_and_fact_presentation(class_authoring_without_engine):
    class UnionNode(Expr):
        def __init__(self, value: i64) -> None: ...

    x, y = vars_("x y", UnionNode)
    action = union(x).with_(y)
    command = action.__egg_ref__.read()
    query = Fact(action.__egg_ref__.owner.ref("nodes", command.kind.value.kind.value))
    assert str(query) == "eq(x).to(y)"
    assert str(action) == "union(x).with_(y)"
    authored = rule(query, name="union-context").then(action)
    expected = "rule(eq(x).to(y), name='union-context').then(union(x).with_(y))"
    assert str(authored) == expected
    packed = pack([authored.__egg_ref__]).program
    canonical = _required_oneof(packed.rules[0].kind, "rule", pb.RuleBody)
    assert canonical.query == [_required_oneof(canonical.head[0].kind, "term", int)]
    assert pb.Program.from_binary(packed.to_binary()) == packed


def test_fact_action_reuses_authored_union_nodes_without_engine(class_authoring_without_engine):
    class UnionNode(Expr):
        def __init__(self, value: i64) -> None: ...

    lhs, rhs = UnionNode(1), UnionNode(2)  # type: ignore[arg-type]
    fact = eq(lhs).to(rhs)
    separate = eq(lhs).to(rhs)
    original = fact.__egg_ref__.read().to_binary()
    authored = rule(fact).then(fact, fact, separate)
    actions = [expr_action(item) for item in (fact, fact, separate)]
    packet = pack(
        [fact.__egg_ref__, separate.__egg_ref__, authored.__egg_ref__],
        commands=[action.__egg_ref__ for action in actions],
    )
    program = pb.Program.from_binary(packet.program.to_binary())
    first, second, rule_index = packet.indices
    assert first != second
    _required_oneof(program.nodes[first].kind, "union", pb.Union)
    _required_oneof(program.nodes[second].kind, "union", pb.Union)
    assert (
        sum(
            node.kind.field == "union" if node.kind is not None else pytest.fail("Missing node kind")
            for node in program.nodes
        )
        == 2
    )
    action_data = [_required_oneof(command.kind, "action", pb.Action) for command in program.commands]
    assert [_required_oneof(command.kind, "term", int) for command in action_data] == [first, first, second]
    canonical = _required_oneof(program.rules[rule_index].kind, "rule", pb.RuleBody)
    assert canonical.query == [first]
    assert [_required_oneof(action.kind, "term", int) for action in canonical.head] == [first, first, second]
    assert fact.__egg_ref__.read().to_binary() == original


def test_set_and_term_authoring_own_commands_without_evaluating(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Action authoring cannot create an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    builder = Builder()
    output = builder.import_ref(i64.__egg_sort__)  # type: ignore[attr-defined]
    index = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function]("function", pb.Function(name="slot", output=output))
        ),
    )
    target = RuntimeFunction(builder.publish().ref("declarations", index))()
    assert target is not None
    rhs = to_runtime_expr(i64(4))
    action = set_(target).to(rhs)
    with pytest.raises(TypeError, match="unhashable type: 'Action'"):
        hash(action)
    term = expr_action(target)
    rhs.__egg_ref__ = i64(9).__egg_ref__
    target.__egg_ref__ = i64(5).__egg_ref__
    packed = pack(commands=[action.__egg_ref__, term.__egg_ref__]).program
    assignment = _required_oneof(_required_oneof(packed.commands[0].kind, "action", pb.Action).kind, "set", pb.Set)
    assert assignment.target is not None
    assert assignment.target.func == "slot"
    scalar_value = _required_oneof(packed.nodes[assignment.value].kind, "primitive_value", pb.PrimitiveValue)
    assert _required_oneof(scalar_value.value, "i64", int) == 4
    _required_oneof(_required_oneof(packed.commands[1].kind, "action", pb.Action).kind, "term", int)
    assert str(action) == "set_(slot()).to(i64(4))"
    assert str(term) == "slot()"
    assert str(set_(i64(4))) == "set_(i64(4))"
    with pytest.raises(TypeError, match="function calls"):
        set_(i64(4)).to(i64(5))
    with pytest.raises(TypeError, match="table-backed"):
        set_(i64(4) + 1).to(i64(5))


def test_action_equality_uses_commands_and_independent_declarations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Action comparison accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    left = constant("score", i64, egg_name="score")
    right = constant("score", i64, egg_name="score")
    first, second = set_(left).to(1), set_(right).to(1)  # type: ignore[arg-type]
    assert first == second
    assert (first != second) is False
    assert first != set_(right).to(2)  # type: ignore[arg-type]
    assert first != expr_action(right)
    assert first != set_(constant("other", i64, egg_name="other")).to(1)  # type: ignore[arg-type]
    assert first.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError, match="unhashable type: 'Action'"):
        hash(first)


def test_action_equality_ignores_retained_python_helper_identity():
    def make():
        class Token(Expr):
            @method(preserve=True)
            def tag(self):
                return "retained"

        token = constant("token", Token)
        runtime_token, _token_context = _required_authored_class(Token)
        return expr_action(token), runtime_token.__egg_hooks__["tag"]

    first, first_hook = make()
    second, second_hook = make()
    assert first_hook is not second_hook
    assert first == second


def test_public_f64_structural_identity_and_signed_zero():
    source = float("nan")
    left, same_source, fresh = f64(source), f64(source), f64(float("nan"))
    assert expr_parts(left) == expr_parts(same_source)
    assert bool(left == same_source)
    assert hash(left) == hash(same_source)
    assert expr_parts(left) == expr_parts(fresh)
    assert bool(left == fresh)
    assert hash(left) == hash(fresh)
    assert expr_parts(left) == expr_parts(copy(left))
    assert hash(left) == hash(copy(left))
    assert math.isnan(left.value)
    assert left.__egg_ref__.read().kind.value.value.value == int.from_bytes(struct.pack("!d", source))
    assert struct.pack("!d", left.value) == struct.pack("!d", source)
    assert math.isnan(fresh.value)
    assert struct.pack("!d", left.value) == struct.pack("!d", fresh.value)

    positive, negative = f64(0.0), f64(-0.0)
    assert expr_parts(positive) == expr_parts(negative)
    assert bool(positive == negative)
    assert hash(positive) == hash(negative)
    assert struct.pack("!d", positive.value) != struct.pack("!d", negative.value)


def test_action_equality_preserves_numeric_zero_and_nan_payloads():
    positive, negative = f64(0.0), f64(-0.0)
    assert positive.__egg_ref__.read().kind.value.value.value != negative.__egg_ref__.read().kind.value.value.value
    assert expr_action(positive) == expr_action(negative)
    assert expr_action(f64(1.0)) != expr_action(f64(2.0))
    assert expr_action(i64(1)) != expr_action(f64(1.0))
    value = f64(float("nan"))
    assert expr_action(value) == expr_action(copy(value))
    assert expr_action(value) == expr_action(f64(float("nan")))


def test_action_equality_compares_nonempty_union_syntax_in_order():
    union = eq(i64(1)).to(i64(2))
    assert expr_action(RuntimeExpr(union.__egg_ref__)) == expr_action(RuntimeExpr(union.__egg_ref__))
    independent = eq(i64(1)).to(i64(2))
    assert expr_action(RuntimeExpr(union.__egg_ref__)) == expr_action(RuntimeExpr(independent.__egg_ref__))
    reversed_union = eq(i64(2)).to(i64(1))
    assert expr_action(RuntimeExpr(union.__egg_ref__)) != expr_action(RuntimeExpr(reversed_union.__egg_ref__))
    left = i64(1)
    assert expr_action(left + left) == expr_action(i64(1) + i64(1))
    assert expr_action(i64(1) + i64(2)) != expr_action(i64(2) + i64(1))


@pytest.mark.parametrize(
    ("default", "options", "message"),
    [
        (False, {"ruleset": object()}, "Explicit rulesets require a default"),
        (True, {"ruleset": object()}, "Primitive-returning defaults cannot use an explicit ruleset"),
        (True, {"merge": lambda old, new: pytest.fail("merge evaluated")}, "Primitive-returning callables with bodies"),
    ],
)
def test_invalid_constant_options_reject_before_authored_records(default, options, message):
    args = (i64(1),) if default else ()
    with pytest.raises(ValueError, match=message):
        constant("bad", i64, *args, **options)


def test_constant_eager_defaults_are_closed_nullary_primitives(monkeypatch):
    from egglog import Maybe  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Constant authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)

    class ConstantBox(Expr):
        def __init__(self, value: i64) -> None: ...

    @function
    def seed(value: i64) -> i64:
        return value + 1

    scalar, box = seed(i64(3)), ConstantBox(i64(4))
    before = pack([scalar.__egg_ref__, box.__egg_ref__]).program.to_binary()
    for value, expected in [
        (constant("scalar", i64, scalar), scalar),
        (constant("box", ConstantBox, box), box),
        (constant("missing", Maybe[i64], None), Maybe[i64].none()),
    ]:
        declaration = resolve_callable(value)
        primitive = declaration.read().kind.value
        assert isinstance(primitive, pb.Primitive)
        assert not primitive.inputs
        assert expr_parts(RuntimeExpr(declaration.owner.ref("nodes", primitive.body))) == expr_parts(expected)
        with pytest.raises(TypeError, match="table-backed Function"):
            set_(value).to(expected)
        encoded = pack([value.__egg_ref__]).program.to_binary()
        assert pb.Program.from_binary(encoded).to_binary() == encoded
    assert pack([scalar.__egg_ref__, box.__egg_ref__]).program.to_binary() == before


def test_constant_merges_are_nullary_functions_without_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Constant authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    calls = []

    def maximum(old, new):
        calls.append(get_current_ruleset())
        return old.max(new)

    class MergedToken(Expr):
        def __init__(self) -> None: ...

    ambient = ruleset()
    with set_current_ruleset(ambient):
        maximum_value = constant("maximum", i64, merge=maximum)
        token = constant("token", MergedToken, merge=lambda old, _new: old)
        assert get_current_ruleset() is ambient
    assert calls == [None]
    for value, expected in [(maximum_value, "old.max(new)"), (token, "old")]:
        declaration = resolve_callable(value)
        merged = declaration.read().kind.value
        assert isinstance(merged, pb.Function)
        assert not merged.inputs
        assert str(RuntimeExpr(declaration.owner.ref("nodes", merged.merge))) == expected
    assert set_(maximum_value).to(i64(2)).__egg_ref__.read().kind.value.kind.field == "set"
    assert set_(token).to(MergedToken()).__egg_ref__.read().kind.value.kind.field == "set"
    assert not ambient._snapshot().read().kind.value.rules


def test_constant_ruleset_default_is_a_destination_constructor_rewrite(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Constant authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)

    class DefaultToken(Expr):
        def __init__(self, value: i64) -> None: ...

    destination, ambient = ruleset(), ruleset()
    previous = destination._snapshot()
    default = DefaultToken(i64(3))
    with set_current_ruleset(ambient):
        value = constant("token", DefaultToken, default, ruleset=destination)
        assert get_current_ruleset() is ambient
    assert resolve_callable(value).read().kind.field == "constructor"
    assert not previous.read().kind.value.rules
    assert not ambient._snapshot().read().kind.value.rules
    snapshot = destination._snapshot()
    assert len(snapshot.read().kind.value.rules) == 1
    reference = snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])
    record = reference.read().kind.value
    assert isinstance(record, pb.Rewrite)
    assert not record.conditions
    assert not record.subsume
    assert expr_parts(RuntimeExpr(reference.owner.ref("nodes", record.lhs))) == expr_parts(value)
    assert expr_parts(RuntimeExpr(reference.owner.ref("nodes", record.rhs))) == expr_parts(default)
    encoded = pack(commands=[snapshot]).program.to_binary()
    assert pb.Program.from_binary(encoded).to_binary() == encoded
    with pytest.raises(TypeError, match="Rewrite destinations must be Rulesets"):
        constant("invalid", DefaultToken, default, ruleset=object())  # type: ignore[call-overload]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("spelling", ["external", "_0"])
def test_constant_defaults_reject_unbound_query_variables(explicit, spelling):
    class QueryDefaultToken(Expr):
        def __init__(self, value: i64) -> None: ...

    destination = ruleset() if explicit else None
    default = QueryDefaultToken(var(spelling, i64))
    with pytest.raises(ValueError, match=r"Free variable.*named eager body"):
        constant("invalid", QueryDefaultToken, default, ruleset=destination)
    if destination is not None:
        assert not destination._snapshot().read().kind.value.rules


@pytest.mark.parametrize("egg_name", [None, ""])
def test_constant_generated_name_preserves_percent_prefix(egg_name):
    value = constant("current", i64, egg_name=egg_name)
    call = value.__egg_ref__.read().kind.value
    assert call.func == f"%{__name__.replace('.', '_')}_current"


def test_constants_own_distinct_canonical_presentation_and_declaration_kinds(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Constant authoring cannot create an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)

    class Token(Expr):
        @method(preserve=True)
        def tag(self):
            return "retained"

    table = constant("current", i64, egg_name="exact table name")
    token = constant("token", Token, egg_name="exact token name")
    assert resolve_callable(table).read().kind.field == "function"
    assert resolve_callable(token).read().kind.field == "constructor"
    assert str(table) == "current"
    assert str(token) == "token"
    assert token.tag() == "retained"
    constant_name = get_constant_name(table)
    assert constant_name is not None
    assert constant_name.name == "current"
    constant_name = get_constant_name(table)
    assert constant_name is not None
    assert constant_name.module == __name__
    assert get_constant_name(i64(1)) is None
    assert get_constant_name(i64(1) + 2) is None
    with pytest.raises(TypeError, match="Expected Expression"):
        get_constant_name(1)
    with pytest.raises(TypeError, match="table-backed"):
        set_(token).to(token)
    packed = pack([table.__egg_ref__])
    program = pb.Program.from_binary(packed.program.to_binary())
    declaration = next(
        d
        for d in program.declarations
        if (d.kind.field == "function" if d.kind is not None else pytest.fail("Missing declaration kind"))
    )
    _required_python_bindings(declaration).views[0].kind = pb.PythonCallKind.FUNCTION
    reconstructed = RuntimeExpr(Builder.from_program(program).publish().ref("nodes", packed.indices[0]))
    assert str(reconstructed) == "current()"
    assert get_constant_name(reconstructed) is None
    assert str(table) == "current"


def test_constant_keeps_local_sort_hooks_alive_without_a_global_registry():
    def make():
        class Token(Expr):
            @method(preserve=True)
            def tag(self):
                return "local token"

        token = constant("token", Token)
        runtime_token, _token_context = _required_authored_class(Token)
        return token, weakref(Token), weakref(runtime_token.__egg_definition__.owner)

    token, class_ref, owner_ref = make()
    _CLASS_CACHE.clear()
    copied = copy(token)
    del token
    gc.collect()
    assert copied.tag() == "local token"
    assert class_ref() is not None
    del copied
    gc.collect()
    assert class_ref() is None
    assert owner_ref() is None


def test_user_constructor_defaults_keywords_and_names_are_records(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Class authoring cannot create an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    default = i64(5)

    class Num(Expr, egg_sort="exact sort"):
        @method(egg_fn="exact constructor")
        def __init__(self, value: i64 = default) -> None: ...

    runtime_num, num_context = _required_authored_class(Num)
    first = Num()
    declaration = resolve_callable(Num)
    record = declaration.read()
    assert record.kind.value.name == "exact constructor"
    assert runtime_num.__egg_definition__.read().kind.value.name == "exact sort"
    assert record.bindings.python.views[0].params[0].name == "value"
    assert _required_callable_args(first, Num)[0].value == 5
    default.__replace_expr__(i64(9))
    assert _required_callable_args(Num(), Num)[0].value == 5
    assert expr_parts(Num(value=3)) == expr_parts(Num(3))  # type: ignore[arg-type]
    assert num_context.prepare is None
    runtime_num.__egg_attr_cache__.clear()
    assert num_context.catalog is not None
    num_context.catalog.members.clear()
    num_context.catalog.reindex()
    assert resolve_callable(Num) == declaration
    assert _required_callable_args(Num(), Num)[0].value == 5
    with pytest.raises(TypeError):
        Num(1, value=2)  # type: ignore[arg-type, misc]
    with pytest.raises(TypeError):
        Num(unknown=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Num(1, 2)  # type: ignore[arg-type, call-arg]
    with pytest.raises(ConvertError, match="Cannot convert"):
        Num("wrong sort")  # type: ignore[arg-type]


def test_user_preserved_protocols_and_match_args():
    class Local(Expr):
        def __init__(self) -> None: ...

        __match_args__ = ("value",)

        @method(preserve=True)  # type: ignore[prop-decorator]
        @property
        def value(self):
            return 4

        @method(preserve=True)
        def tag(self):
            return "tag"

        def __str__(self) -> str:
            return "local str"

        def __repr__(self) -> str:
            return "local repr"

        def __hash__(self) -> int:
            return 42

        @method(preserve=True)
        def __eq__(self, other) -> bool:  # type: ignore[override]
            return other == "sentinel"

        @method(preserve=True)
        def __ne__(self, other) -> bool:  # type: ignore[override]
            return other != "sentinel"

    expression = Local()
    assert expression.value == 4
    assert expression.tag() == "tag"
    assert str(expression) == "local str"
    assert repr(expression) == "local repr"
    assert hash(expression) == 42
    assert expression == "sentinel"
    assert (expression == "other") is False
    assert expression != "other"
    assert (expression != "sentinel") is False
    assert bool(eq(expression).to(copy(expression)))
    match expression:
        case Local(4):
            pass
        case _:
            pytest.fail("Explicit match_args and preserved property were lost")


def test_local_class_hooks_survive_caches_but_not_all_roots():
    def make(label):
        class Local(Expr):
            def __init__(self) -> None: ...

            @method(preserve=True)
            def tag(self):
                return label

        runtime_local, local_context = _required_authored_class(Local)
        expression = Local()
        return expression, weakref(Local), weakref(runtime_local.__egg_definition__.owner), weakref(local_context)

    left, cls, owner, context = make("left")
    right, other_cls, other_owner, other_context = make("right")
    retained = copy(left)
    _CLASS_CACHE.clear()
    gc.collect()
    assert left.tag() == retained.tag() == "left"
    assert right.tag() == "right"
    assert cls() is not other_cls()
    del left, right, retained
    gc.collect()
    assert cls() is owner() is context() is None
    assert other_cls() is other_owner() is other_context() is None


def test_recursive_class_sorts_resolve_without_recursive_member_compilation():
    class Left(Expr):
        def __init__(self, right: "Right") -> None: ...

    class Right(Expr):
        def __init__(self, left: Left) -> None: ...

    left = resolve_callable(Left)
    right = resolve_callable(Right)
    runtime_left, _left_context = _required_authored_class(Left)
    runtime_right, _right_context = _required_authored_class(Right)
    left_sort, right_sort = runtime_left.__egg_sort__, runtime_right.__egg_sort__
    assert left_sort is not None
    assert right_sort is not None
    assert StructuralView(left.owner.ref("sorts", left.read().kind.value.inputs[0].sort)) == StructuralView(right_sort)
    assert StructuralView(right.owner.ref("sorts", right.read().kind.value.inputs[0].sort)) == StructuralView(left_sort)


def test_type_factory_retains_canonical_sort_and_formals(class_authoring_without_engine):
    def init(self, value: i64) -> None: ...

    def choose(self, value: i64) -> i64:
        return value

    namespace = {"__module__": __name__, "__init__": init, "choose": choose}
    dynamic = type("Factory", (Expr,), namespace)
    expression = dynamic(3)
    runtime_dynamic, _dynamic_context = _required_authored_class(dynamic)
    dynamic_sort = runtime_dynamic.__egg_sort__
    assert dynamic_sort is not None
    assert isinstance(expression, dynamic)
    assert runtime._class_for_sort(dynamic_sort) is dynamic
    assert namespace == {"__module__": __name__, "__init__": init, "choose": choose}
    reference = resolve_callable(runtime_dynamic.choose)
    declaration = reference.read()
    assert declaration.kind.field == "primitive"
    assert [argument.name for argument in declaration.kind.value.inputs] == ["self", "value"]
    body = reference.owner.ref("nodes", declaration.kind.value.body)
    assert body.read().kind == Oneof[Literal["var"], str]("var", "_1")
    assert StructuralView(reference.owner.ref("sorts", declaration.kind.value.inputs[0].sort)) == StructuralView(
        dynamic_sort
    )
    with pytest.raises(TypeError):
        runtime.RuntimeClass("Spoof", (Expr,), {})  # type: ignore[arg-type, call-arg]
    with pytest.raises(TypeError):
        runtime.RuntimeClass(dynamic.__egg_definition__, object())  # type: ignore[arg-type, call-arg]
    with pytest.raises(TypeError):
        runtime.RuntimeClass(dynamic.__egg_definition__, unsupported=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="requires a canonical sort declaration"):
        runtime.RuntimeClass(resolve_callable(dynamic))


def test_constructor_scope_publishes_once_across_authoring_threads():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    _runtime_num, num_context = _required_authored_class(Num)
    calls = []
    prepare = num_context.prepare
    assert prepare is not None

    def delayed_prepare(cls):
        calls.append(cls)
        sleep(0.02)  # Hold first compilation open while other authors enter.
        return prepare(cls)

    num_context.prepare = delayed_prepare
    with ThreadPoolExecutor(max_workers=4) as executor:
        expressions = list(executor.map(Num, range(8)))
    assert calls == [Num]
    assert [_required_callable_args(expression, Num)[0].value for expression in expressions] == list(range(8))


@pytest.mark.parametrize(
    "options",
    [
        {"cost": 1},
        {"merge": lambda old, new: old},
        {"mutates_self": True},
        {"unextractable": True},
        {"subsume": True},
        {"preserve": True},
    ],
)
def test_unmigrated_constructor_options_never_publish_members(options):
    class Invalid(Expr):
        @method(**options)
        def __init__(self) -> None: ...

    if "cost" in options:
        expression = Invalid()
        reference = resolve_callable(Invalid)
        record = reference.read()
        assert record.kind.field == "constructor"
        assert record.bindings.python.views[0].kind == pb.PythonCallKind.INITIALIZER
        cost = reference.owner.ref("nodes", record.kind.value.cost).read()
        assert cost.kind.value.value == Oneof[Literal["i64"], int]("i64", 1)
        assert get_callable_fn(expression) == Invalid
        assert resolve_callable(Invalid) is reference
        assert {name for _, name in Invalid.__egg_context__.catalog.members} == {"__init__"}
        return
    with pytest.raises(NotImplementedError, match="options"):
        Invalid()
    assert not Invalid.__egg_context__.catalog.members
    assert not Invalid.__egg_attr_cache__


def test_reverse_args_members_map_heterogeneous_inputs_and_mutation(class_authoring_without_engine):
    class ReversedResult(Expr): ...

    class ReversedSource(Expr):
        def __init__(self, value: i64) -> None: ...

        @method(reverse_args=True)
        def make(self, label: StringLike, count: i64 = i64(4)) -> ReversedResult: ...

        @method(reverse_args=True)
        @classmethod
        def create(cls, label: StringLike, count: i64) -> "ReversedSource": ...

        @method(reverse_args=True, mutates_self=True)
        def replace(self, label: StringLike) -> None: ...

    source = ReversedSource(i64(3))
    expression = source.make("tag")
    declaration = resolve_callable(ReversedSource.make)
    record = declaration.read()
    assert [argument.name for argument in record.kind.value.inputs] == ["count", "label", "self"]
    view = record.bindings.python.views[0]
    assert view.receiver == 2
    assert [(parameter.name, parameter.core_input) for parameter in view.params] == [("label", 1), ("count", 0)]
    assert view.params[1].has_field("default_expr")
    call = expression.__egg_ref__.read().kind.value
    arguments = [expression.__egg_ref__.owner.ref("nodes", index) for index in call.args]
    assert [RuntimeExpr(argument) for argument in arguments] == [i64(4), String("tag"), source]
    assert get_callable_fn(expression) == ReversedSource.make
    assert get_callable_args(expression) == (
        to_runtime_expr(source),
        to_runtime_expr(String("tag")),
        to_runtime_expr(i64(4)),
    )
    assert expr_parts(ReversedSource.make(source, count=i64(4), label="tag")) == expr_parts(expression)
    with pytest.raises(ConvertError, match="Cannot convert"):
        source.make(i64(9))  # type: ignore[arg-type]
    with pytest.raises(ConvertError, match="Cannot convert"):
        source.make("tag", String("wrong"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="too many positional"):
        source.make("tag", i64(4), i64(5))  # type: ignore[call-arg]
    packed = pack([expression.__egg_ref__])
    program = pb.Program.from_binary(packed.program.to_binary())
    detached = RuntimeExpr(Builder.from_program(program).publish().ref("nodes", packed.indices[0]))
    assert expr_parts(detached) == expr_parts(expression)
    assert str(detached) == str(expression)
    assert get_callable_args(detached) == get_callable_args(expression)

    created = ReversedSource.create("tag", i64(4))
    create = resolve_callable(ReversedSource.create).read()
    assert [argument.name for argument in create.kind.value.inputs] == ["count", "label"]
    assert not create.bindings.python.views[0].has_field("receiver")
    assert get_callable_args(created) == (to_runtime_expr(String("tag")), to_runtime_expr(i64(4)))

    snapshot = copy(source)
    assert source.replace("next") is None
    replace = resolve_callable(ReversedSource.replace).read()
    assert replace.bindings.python.views[0].receiver == replace.bindings.python.views[0].mutates == 1
    assert get_callable_args(source, ReversedSource.replace) == (
        to_runtime_expr(snapshot),
        to_runtime_expr(String("next")),
    )


def test_reverse_args_initializer_retains_python_parameter_order(class_authoring_without_engine):
    class ReversedInitializer(Expr):
        @method(reverse_args=True)
        def __init__(self, label: StringLike, count: i64) -> None: ...

    expression = ReversedInitializer("tag", i64(4))
    declaration = resolve_callable(ReversedInitializer).read()
    assert [argument.name for argument in declaration.kind.value.inputs] == ["count", "label"]
    view = declaration.bindings.python.views[0]
    assert not view.has_field("receiver")
    assert [(parameter.name, parameter.core_input) for parameter in view.params] == [("label", 1), ("count", 0)]
    assert get_callable_fn(expression) == ReversedInitializer
    assert get_callable_args(expression) == (to_runtime_expr(String("tag")), to_runtime_expr(i64(4)))
    assert expr_parts(ReversedInitializer(count=i64(4), label="tag")) == expr_parts(expression)


def test_reverse_args_rewrite_body_receives_python_order(class_authoring_without_engine):
    destination = ruleset()
    calls = []

    class ReversedRewrite(Expr, ruleset=destination):
        def __init__(self, value: i64) -> None: ...

        @method(reverse_args=True)
        def select(self, label: StringLike, count: i64) -> "ReversedRewrite":
            calls.append((str(self), str(label), str(count)))
            return self

    ReversedRewrite(i64(1)).select("tag", i64(2))
    assert calls == [("self", "label", "count")]
    snapshot = destination._snapshot()
    assert len(snapshot.read().kind.value.rules) == 1
    reference = snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])
    rewrite = reference.read().kind.value
    lhs = reference.owner.ref("nodes", rewrite.lhs)
    assert [lhs.owner.ref("nodes", index).read().kind.value for index in lhs.read().kind.value.args] == [
        "count",
        "label",
        "self",
    ]
    assert reference.owner.ref("nodes", rewrite.rhs) == lhs.owner.ref("nodes", lhs.read().kind.value.args[-1])
    assert [str(argument) for argument in _required_callable_args(RuntimeExpr(lhs))] == ["self", "label", "count"]


@pytest.mark.parametrize("options", [{"cost": 1}, {}])
def test_wrapped_ignored_name_rejects_but_retains_prior_members(options):
    class Num(Expr):
        def __init__(self) -> None: ...

        @method(**options)
        def __radd__(self, other: i64) -> i64: ...

    runtime_num, num_context = _required_authored_class(Num)
    with pytest.raises(NotImplementedError, match="__radd__"):
        Num()
    assert num_context.catalog is not None
    assert {name for _, name in num_context.catalog.members} == {"__init__"}
    assert not runtime_num.__egg_attr_cache__
    assert isinstance(Num(), Num)
    with pytest.raises(AttributeError):
        _ = Num.__radd__


def test_preserved_wrapped_ignored_name_remains_a_local_hook():
    class Num(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __radd__(self, other: int) -> int:
            return other + 1

    _runtime_num, num_context = _required_authored_class(Num)
    expression = Num()
    assert Num.__radd__(expression, 2) == 3
    assert num_context.catalog is not None
    assert {name for _, name in num_context.catalog.members} == {"__init__"}


def test_unsupported_class_members_and_parameter_kinds_are_not_ignored():
    class HasBody(Expr):
        def __init__(self) -> None: ...

        def other(self) -> i64:
            return i64(1)

    class KeywordOnly(Expr):
        def __init__(self, *, value: i64) -> None: ...

    runtime_keyword_only, keyword_only_context = _required_authored_class(KeywordOnly)
    # Eager method bodies are supported; fixed member parameter kinds still apply.
    assert isinstance(HasBody().other(), i64)
    assert resolve_callable(HasBody.other).read().kind.field == "primitive"
    with pytest.raises(NotImplementedError, match="positional-or-keyword"):
        KeywordOnly()  # type: ignore[call-arg]
    assert keyword_only_context.catalog is not None
    assert not keyword_only_context.catalog.members
    assert not runtime_keyword_only.__egg_attr_cache__


def test_concrete_member_records_defaults_and_unbound_views(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Class authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    default = i64(4)

    class Node(Expr, egg_sort="exact Node sort"):
        ZERO: ClassVar["Node"]
        COUNT: ClassVar[i64]

        def __init__(self) -> None: ...

        @method(egg_fn="exact member")
        def score(self, x: i64 = default) -> i64: ...

        @property
        def weight(self) -> i64: ...

        @classmethod
        def make(cls, x: i64 = default) -> "Node": ...

    runtime_node, node_context = _required_authored_class(Node)
    assert runtime_node.__egg_sort__ is not None
    expression = Node()
    assert expr_parts(Node.score(expression, 2)) == expr_parts(expression.score(x=2))  # type: ignore[arg-type]
    assert expr_parts(Node.weight(expression)) == expr_parts(expression.weight)
    assert str(expression.score()) == "Node().score()"
    assert str(expression.weight) == "Node().weight"
    assert str(Node.make()) == "Node.make()"
    assert str(Node.ZERO) == "Node.ZERO"
    assert str(Node.COUNT) == "Node.COUNT"
    score = resolve_callable(Node.score)
    score_record = score.read()
    assert score_record.kind.field == "function"
    assert score_record.kind.value.name == "exact member"
    view = score_record.bindings.python.views[0]
    assert view.receiver == 0
    assert [(p.name, p.core_input) for p in view.params] == [("x", 1)]
    assert StructuralView(score.owner.ref("sorts", score_record.kind.value.inputs[0].sort)) == StructuralView(
        runtime_node.__egg_sort__
    )
    assert resolve_callable(Node.make).read().kind.field == "constructor"
    assert resolve_callable(Node.ZERO).read().kind.field == "constructor"
    assert resolve_callable(Node.COUNT).read().kind.field == "function"
    default.__replace_expr__(i64(9))
    assert _required_callable_args(expression.score())[1].value == 4
    assert _required_callable_args(Node.make())[0].value == 4
    runtime_node.__egg_attr_cache__.clear()
    assert node_context.catalog is not None
    node_context.catalog.members.clear()
    node_context.catalog.reindex()
    assert resolve_callable(Node.score) == score
    assert _required_callable_args(expression.score())[1].value == 4
    with pytest.raises(TypeError):
        expression.score(1, x=2)  # type: ignore[arg-type, misc]
    with pytest.raises(TypeError):
        Node.make(unknown=1)  # type: ignore[call-arg]
    with pytest.raises(ConvertError):
        expression.score(f64(1))  # type: ignore[arg-type]
    with pytest.raises(ConvertError):
        Node.score(i64(1), i64(2))  # type: ignore[arg-type]


@pytest.mark.parametrize("future", [False, True])
def test_classvar_annotations_are_late_and_owner_scoped(future):
    namespace = {"Expr": Expr, "ClassVar": ClassVar, "i64": i64, "__name__": f"{__name__}.annotations_{future}"}
    exec(
        ("from __future__ import annotations\n" if future else "")
        + "class First(Expr):\n"
        + "    VALUE: ClassVar[Later]\n"
        + "    def choose(self, other: Later) -> Later: ...\n"
        + "class Later(Expr):\n"
        + "    ROOT: ClassVar[First]\n",
        namespace,
    )
    first, later = namespace["First"], namespace["Later"]
    runtime_first, _first_context = _required_authored_class(first)
    runtime_later, later_context = _required_authored_class(later)
    first_matches: bool = isinstance(runtime_first.VALUE, runtime_later)
    assert first_matches
    root_matches: bool = isinstance(runtime_later.ROOT, runtime_first)
    assert root_matches
    later_root = runtime_later.ROOT
    assert isinstance(later_root, RuntimeExpr)
    choose = _required_runtime_function(later_root.choose)
    first_value = runtime_first.VALUE
    chosen_matches: bool = isinstance(choose(first_value), runtime_later)
    assert chosen_matches
    assert later_context.catalog is not None


@pytest.mark.parametrize("descriptor", [property, classmethod])
def test_member_decorator_order_preserves_value_error(descriptor):
    class Invalid(Expr):
        def __init__(self) -> None: ...

        @descriptor
        @method(egg_fn="wrong order")
        def member(self) -> i64: ...

    runtime_invalid, invalid_context = _required_authored_class(Invalid)
    with pytest.raises(ValueError, match="above @classmethod or @property"):
        Invalid()
    assert invalid_context.catalog is not None
    assert {name for _, name in invalid_context.catalog.members} == {"__init__"}
    assert not runtime_invalid.__egg_attr_cache__
    assert isinstance(Invalid(), Invalid)
    with pytest.raises(AttributeError):
        _ = Invalid.member


@pytest.mark.parametrize("doc", [None, "", "member documentation"])
def test_concrete_member_documentation_presence(doc):
    class Documented(Expr):
        def __init__(self) -> None: ...
        def value(self) -> i64: ...

        value.__doc__ = doc

        @classmethod
        def named(cls) -> i64: ...

        named.__func__.__doc__ = doc  # type: ignore[attr-defined]

        @property
        def weight(self) -> i64: ...

        weight.fget.__doc__ = doc  # type: ignore[attr-defined]

    for fn in (Documented.value, Documented.named, Documented.weight):
        assert fn.__doc__ == doc
        assert fn.__module__ == __name__
        assert resolve_callable(fn).read().has_field("doc") == (doc is not None)


def test_member_scope_failure_keeps_prefix_without_executing_body():
    calls = []

    class Invalid(Expr):
        VALUE: ClassVar[i64]

        def __init__(self) -> None: ...
        def valid(self) -> i64: ...

        def body(self) -> i64:
            calls.append(1)
            return i64(2)

        @method(cost=1)  # type: ignore[type-var]
        def invalid(self) -> i64: ...

    runtime_invalid, invalid_context = _required_authored_class(Invalid)
    # The supported eager header is private until complete. A genuinely invalid
    # later header must still fail before callbacks or publication of that body.
    with pytest.raises(ValueError, match="cannot use cost"):
        Invalid.valid
    assert calls == []
    assert invalid_context.catalog is not None
    assert {name for _, name in invalid_context.catalog.members} == {"VALUE", "__init__", "valid"}
    assert not runtime_invalid.__egg_attr_cache__
    assert isinstance(Invalid.VALUE, i64)
    assert isinstance(Invalid().valid(), i64)
    with pytest.raises(AttributeError):
        _ = Invalid.body
    assert calls == []


def test_eager_member_forward_completion_owns_only_final_definitions(class_authoring_without_engine):
    calls = []
    captured = []

    class Eager(Expr):
        def __init__(self) -> None: ...

        def first(self, value: i64 = i64(3)) -> i64:
            calls.append("first")
            captured.append(Eager.later)
            return self.later(value) + 1

        def later(self, value: i64) -> i64:
            calls.append("later")
            return value + 2

        @property
        def number(self) -> i64:
            calls.append("number")
            return i64(4)

        @classmethod
        def make(cls) -> "Eager":
            calls.append("make")
            return cls()

    expression = Eager()
    first, later = resolve_callable(Eager.first), resolve_callable(Eager.later)
    assert captured == [Eager.later]
    assert resolve_callable(captured[0]) is later
    assert calls == ["first", "later", "number", "make"]
    assert first.read().kind.field == later.read().kind.field == "primitive"
    body = RuntimeExpr(first.owner.ref("nodes", first.read().kind.value.body))
    assert str(body) == "_0.later(_1) + 1"
    assert isinstance(expression.first(), i64)
    assert isinstance(expression.number, i64)
    assert isinstance(Eager.make(), Eager)
    packed = pack([to_runtime_expr(expression.first()).__egg_ref__, first, later, resolve_callable(captured[0])])
    names = []
    for record in packed.program.declarations:
        kind = record.kind
        assert kind is not None
        names.append(kind.value.name)
    assert names.count(first.read().kind.value.name) == names.count(later.read().kind.value.name) == 1
    assert pb.Program.from_binary(packed.program.to_binary()) == packed.program
    assert calls == ["first", "later", "number", "make"]


def test_eager_member_reverse_args_keep_callback_and_core_order(class_authoring_without_engine):
    class Reverse(Expr):
        def __init__(self, value: i64) -> None: ...

        @method(reverse_args=True)
        def ordered(self, other: i64) -> "Pair[Reverse, i64]":
            return Pair(self, other)

    reference = resolve_callable(Reverse.ordered)
    primitive = reference.read().kind.value
    view = reference.read().bindings.python.views[0]
    assert isinstance(primitive, pb.Primitive)
    assert [argument.name for argument in primitive.inputs] == ["other", "self"]
    assert view.receiver == 1
    assert [parameter.core_input for parameter in view.params] == [0]
    assert str(RuntimeExpr(reference.owner.ref("nodes", primitive.body))) == "Pair[Reverse, i64](_1, _0)"
    assert str(Reverse(1).ordered(2)) == "Reverse(1).ordered(2)"  # type: ignore[arg-type]
    pack([to_runtime_expr(Reverse(1).ordered(2)).__egg_ref__])  # type: ignore[arg-type]


def test_eager_member_eqsort_views_and_later_bodyless_member(class_authoring_without_engine):
    class Result(Expr):
        def __init__(self) -> None: ...

    class Views(Expr):
        def __init__(self) -> None: ...

        def value(self) -> Result:
            return Result()

        @property
        def own(self) -> Result:
            return Result()

        @classmethod
        def make(cls) -> Result:
            return Result()

        def early(self) -> Result:
            return self.later()

        def later(self) -> Result: ...

    value = Views()
    for member, expression in (
        (Views.value, value.value()),
        (Views.own, value.own),
        (Views.make, Views.make()),
        (Views.early, value.early()),
    ):
        reference = resolve_callable(member)
        assert isinstance(reference.read().kind.value, pb.Primitive)
        assert isinstance(expression, Result)
        pack([to_runtime_expr(expression).__egg_ref__])
    assert resolve_callable(Views.later).read().kind.field == "constructor"


def test_eager_initializer_gets_symbolic_self_and_rejects_retained_recursion(class_authoring_without_engine):
    seen = []

    class Initialized(Expr):
        def __init__(self, value: i64) -> None:
            node = self.__egg_ref__.read()
            seen.append((node.kind.value.func, str(value), self.__egg_ref__.owner.ref("sorts", node.sort_id)))
            return Initialized.wrap(value)  # type: ignore[return-value]  # noqa: PLE0101 -- symbolic initializer returns its replacement

        @classmethod
        def wrap(cls, value: i64) -> "Initialized": ...

    value = Initialized(2)  # type: ignore[arg-type]
    reference = resolve_callable(Initialized)
    primitive = reference.read().kind.value
    assert isinstance(primitive, pb.Primitive)
    runtime_initialized, _initialized_context = _required_authored_class(Initialized)
    assert seen == [(primitive.name, "value", runtime_initialized.__egg_sort__)]
    assert str(RuntimeExpr(reference.owner.ref("nodes", primitive.body))) == "Initialized.wrap(_0)"
    pack([to_runtime_expr(value).__egg_ref__])

    class Recursive(Expr):
        def __init__(self) -> None:
            return self  # type: ignore[return-value]  # noqa: PLE0101 -- witness a recursive symbolic initializer result

    runtime_recursive, recursive_context = _required_authored_class(Recursive)
    with pytest.raises(ValueError, match="Recursively resolving function declarations"):
        Recursive()
    assert recursive_context.catalog is not None
    assert not recursive_context.catalog.members
    assert not runtime_recursive.__egg_attr_cache__

    class Replaced(Expr):
        def __init__(self, value: i64) -> None:
            self.__replace_expr__(Replaced.wrap(value))
            return self  # type: ignore[return-value]  # noqa: PLE0101 -- symbolic initializer returns its replacement

        @classmethod
        def wrap(cls, value: i64) -> "Replaced": ...

    pack([to_runtime_expr(Replaced(3)).__egg_ref__])  # type: ignore[arg-type]
    replacement = resolve_callable(Replaced)
    assert str(RuntimeExpr(replacement.owner.ref("nodes", replacement.read().kind.value.body))) == "Replaced.wrap(_0)"


@pytest.mark.parametrize("mutual", [False, True])
def test_eager_member_cycles_fail_without_publishing_pending_headers(class_authoring_without_engine, mutual):
    calls = []

    class Cyclic(Expr):
        def __init__(self) -> None: ...

        def first(self) -> i64:
            calls.append("first")
            return self.later() if mutual else self.first()

        def later(self) -> i64:
            calls.append("later")
            return self.first()

    runtime_cyclic, cyclic_context = _required_authored_class(Cyclic)
    with pytest.raises(ValueError, match="Recursively resolving function declarations"):
        Cyclic.first
    assert calls == (["first", "later"] if mutual else ["first"])
    assert cyclic_context.catalog is not None
    assert {name for _, name in cyclic_context.catalog.members} == {"__init__"}
    assert not runtime_cyclic.__egg_attr_cache__
    with pytest.raises(AttributeError):
        Cyclic.first


def test_eager_member_failure_keeps_completed_dependency_prefix(class_authoring_without_engine):
    calls = []

    class Partial(Expr):
        def __init__(self) -> None: ...

        def first(self) -> i64:
            calls.append("first")
            self.later()
            msg = "body sentinel"
            raise ValueError(msg)

        def later(self) -> i64:
            calls.append("later")
            return i64(2)

    runtime_partial, partial_context = _required_authored_class(Partial)
    with pytest.raises(ValueError, match="body sentinel"):
        Partial.first
    assert calls == ["first", "later"]
    assert partial_context.catalog is not None
    assert {name for _, name in partial_context.catalog.members} == {"__init__", "later"}
    assert set(runtime_partial.__egg_attr_cache__) == {"later"}
    assert isinstance(Partial().later(), i64)
    with pytest.raises(AttributeError):
        Partial.first
    assert calls == ["first", "later"]


@pytest.mark.parametrize("eager_second", [False, True])
def test_eager_member_duplicate_headers_precede_callbacks(class_authoring_without_engine, eager_second):
    calls = []

    class Duplicate(Expr):
        def __init__(self) -> None: ...

        @method(egg_fn="same exact member")
        def first(self) -> i64:
            calls.append("first")
            return i64(1)

        if eager_second:

            @method(egg_fn="same exact member")
            def second(self) -> i64:
                calls.append("second")
                return i64(2)

        else:

            @method(egg_fn="same exact member")
            def second(self) -> i64: ...

    runtime_duplicate, duplicate_context = _required_authored_class(Duplicate)
    with pytest.raises(ValueError, match="Duplicate catalog definition"):
        Duplicate.first
    assert not calls
    assert duplicate_context.catalog is not None
    assert {name for _, name in duplicate_context.catalog.members} == {"__init__"}
    assert not runtime_duplicate.__egg_attr_cache__


def test_eager_member_pending_view_eligibility_precedes_completion(class_authoring_without_engine):
    calls = []

    class Eligible(Expr):
        def __init__(self) -> None: ...

        def first(self) -> i64:
            calls.append("first")
            with pytest.raises(AttributeError, match="not an instance method"):
                runtime.RuntimeClass.__getattr__(Eligible, "number", False, True)  # type: ignore[arg-type]
            assert calls == ["first"]
            return i64(1)

        @property
        def number(self) -> i64:
            calls.append("number")
            return i64(2)

    assert isinstance(Eligible().first(), i64)
    assert calls == ["first", "number"]


def test_eager_member_conversion_is_once_without_ambient_ruleset(class_authoring_without_engine):
    class Source:
        pass

    calls = []

    def convert_source(value):
        calls.append(("conversion", get_current_ruleset()))
        return i64(2)

    converter(Source, i64, convert_source)

    class Converted(Expr):
        def __init__(self) -> None: ...

        def value(self) -> i64:
            calls.append(("body", get_current_ruleset()))
            return Source()  # type: ignore[return-value]

    ambient = ruleset()
    with set_current_ruleset(ambient):
        assert isinstance(Converted().value(), i64)
        assert isinstance(Converted().value(), i64)
        assert get_current_ruleset() is ambient
    assert calls == [("body", None), ("conversion", None)]
    assert not ambient._snapshot().read().kind.value.rules


@pytest.mark.parametrize("late_bad", [False, True])
def test_eager_member_default_reentry_retains_completed_refs_and_prefix(class_authoring_without_engine, late_bad):
    class Source:
        pass

    calls = []
    captured = []

    def convert_source(value):
        calls.append("conversion")
        captured.append(Defaults.first)
        return Defaults().first()

    converter(Source, i64, convert_source)

    class Defaults(Expr):
        def __init__(self) -> None: ...

        def first(self) -> i64:
            calls.append("body")
            return i64(2)

        def with_default(self, value: i64 = Source()) -> i64: ...  # type: ignore[assignment]

        if late_bad:

            @method(cost=1)  # type: ignore[type-var]
            def invalid(self) -> i64: ...

    runtime_defaults, defaults_context = _required_authored_class(Defaults)
    if late_bad:
        with pytest.raises(ValueError, match="cannot use cost"):
            Defaults()
        # Static validation precedes every pending conversion/body. The v2
        # converter-before-error expectation was newly authored, not legacy.
        assert calls == []
        assert captured == []
        assert defaults_context.catalog is not None
        assert {name for _, name in defaults_context.catalog.members} == {"__init__"}
        assert not runtime_defaults.__egg_attr_cache__
        for name in ("first", "with_default", "invalid"):
            with pytest.raises(AttributeError):
                getattr(Defaults, name)
        return
    assert isinstance(Defaults(), Defaults)
    assert calls == ["body", "conversion"]
    first = resolve_callable(Defaults.first)
    assert first is resolve_callable(captured[0])
    assert isinstance(first.read().kind.value, pb.Primitive)
    assert defaults_context.catalog is not None
    catalog_ref = defaults_context.catalog.definitions["callable", first.read().kind.value.name]
    # Owner.ref reconstructs Ref wrappers; canonical identity is owner/role/index,
    # not Python allocation identity and not a structural-equality substitute.
    assert catalog_ref == first
    assert catalog_ref.owner is first.owner
    assert {name for _, name in defaults_context.catalog.members} == {"__init__", "first", "with_default"}
    captured_function = _required_runtime_function(captured[0])
    assert captured_function.__egg_prepare__ is None
    assert captured_function.__egg_pending_declaration__ is None
    expression = Defaults().with_default()
    assert isinstance(expression, i64)
    packet = pack([to_runtime_expr(expression).__egg_ref__, first])
    assert (
        sum(
            record.kind.value.name == first.read().kind.value.name
            if record.kind is not None
            else pytest.fail("Missing declaration kind")
            for record in packet.program.declarations
        )
        == 1
    )
    default = resolve_callable(Defaults.with_default)
    view = default.read().bindings.python.views[0]
    assert (
        default.owner.ref("nodes", view.params[0].default_expr).read().kind.value.func == first.read().kind.value.name
    )
    assert calls == ["body", "conversion"]


@pytest.mark.parametrize("free_query", [False, True])
def test_eager_member_invalid_result_is_terminal_before_publication(class_authoring_without_engine, free_query):
    calls = []
    query = var("external", i64)

    class InvalidResult(Expr):
        def __init__(self) -> None: ...

        def value(self) -> i64:
            calls.append("body")
            return query if free_query else None  # type: ignore[return-value]

    runtime_invalid_result, invalid_result_context = _required_authored_class(InvalidResult)
    with pytest.raises(ValueError, match="Free variable" if free_query else "must return a value"):
        InvalidResult.value
    assert calls == ["body"]
    assert invalid_result_context.catalog is not None
    assert {name for _, name in invalid_result_context.catalog.members} == {"__init__"}
    assert not runtime_invalid_result.__egg_attr_cache__
    with pytest.raises(AttributeError):
        InvalidResult.value
    assert calls == ["body"]


def test_member_default_transitive_forward_headers_and_snapshot(class_authoring_without_engine):
    class Source:
        pass

    calls = []
    captured = []
    supplied = i64(3)
    supplied_ref = supplied.__egg_ref__

    def convert_source(value):
        calls.append("conversion")
        captured.append(Forward.first)
        return Forward().first()

    converter(Source, i64, convert_source)

    class Forward(Expr):
        def __init__(self) -> None: ...

        def first(self) -> i64:
            calls.append("first")
            supplied.__replace_expr__(i64(9))
            return self.later()

        def with_default(self, value: i64 = Source()) -> i64: ...  # type: ignore[assignment]
        def snapshot(self, value: i64 = supplied) -> i64: ...

        def later(self) -> i64:
            calls.append("later")
            return i64(2)

    expression = Forward().with_default()
    # Root completion is source ordered after the static barrier; dependencies
    # complete immediately. Converter-first ordering was never a public contract.
    assert calls == ["first", "later", "conversion"]
    first = resolve_callable(Forward.first)
    assert first is resolve_callable(captured[0])
    snapshot = resolve_callable(Forward.snapshot)
    default_index = snapshot.read().bindings.python.views[0].params[0].default_expr
    assert snapshot.owner.ref("nodes", default_index) == supplied_ref
    assert _required_callable_args(Forward().snapshot())[1].value == 3
    packet = pack([to_runtime_expr(expression).__egg_ref__, first])
    assert (
        sum(
            record.kind.value.name == first.read().kind.value.name
            if record.kind is not None
            else pytest.fail("Missing declaration kind")
            for record in packet.program.declarations
        )
        == 1
    )
    assert pb.Program.from_binary(packet.program.to_binary()) == packet.program
    assert calls == ["first", "later", "conversion"]


def test_member_default_failure_keeps_only_completed_prefix(class_authoring_without_engine):
    class Source:
        pass

    calls = []

    def convert_source(value):
        calls.append("conversion")
        _ = Partial.first
        msg = "parameter conversion marker"
        raise ValueError(msg)

    converter(Source, i64, convert_source)

    class Partial(Expr):
        def __init__(self) -> None: ...

        def first(self) -> i64:
            calls.append("first")
            return i64(1)

        def bad(self, value: i64 = Source()) -> i64: ...  # type: ignore[assignment]
        def unvisited(self) -> i64: ...

    _runtime_partial, partial_context = _required_authored_class(Partial)
    with pytest.raises(ValueError, match="parameter conversion marker"):
        Partial()
    assert calls == ["first", "conversion"]
    assert partial_context.catalog is not None
    assert {name for _, name in partial_context.catalog.members} == {"__init__", "first"}
    assert isinstance(Partial().first(), i64)
    for name in ("bad", "unvisited"):
        with pytest.raises(AttributeError):
            getattr(Partial, name)
    assert calls == ["first", "conversion"]


def test_member_default_cycle_is_bounded_and_terminal(class_authoring_without_engine):
    class Source:
        pass

    calls = []

    def convert_source(value):
        calls.append("conversion")
        return Cyclic().value(i64(1))

    converter(Source, i64, convert_source)

    class Cyclic(Expr):
        def __init__(self) -> None: ...
        def value(self, argument: i64 = Source()) -> i64: ...  # type: ignore[assignment]

    runtime_cyclic, cyclic_context = _required_authored_class(Cyclic)
    with pytest.raises(ValueError, match="Recursively resolving function declarations"):
        Cyclic()
    assert calls == ["conversion"]
    assert cyclic_context.catalog is not None
    assert {name for _, name in cyclic_context.catalog.members} == {"__init__"}
    # The converter actually used this completed prefix; only its cyclic member
    # must disappear, not the valid cached initializer.
    assert set(runtime_cyclic.__egg_attr_cache__) == {"__init__"}
    cached_initializer = runtime_cyclic.__egg_attr_cache__["__init__"]
    assert isinstance(cached_initializer, RuntimeFunction)
    assert cached_initializer.__egg_pending_declaration__ is None
    with pytest.raises(AttributeError):
        Cyclic.value
    assert calls == ["conversion"]


def test_member_default_pending_eligibility_does_not_convert(class_authoring_without_engine):
    class Source:
        def __init__(self, name: str) -> None:
            self.name = name

    calls = []

    def convert_source(value):
        calls.append(value.name)
        if value.name == "first":
            with pytest.raises(AttributeError, match="not an instance method"):
                runtime.RuntimeClass.__getattr__(Eligible, "named", False, True)  # type: ignore[arg-type]
            assert calls == ["first"]
        return i64(1)

    converter(Source, i64, convert_source)

    class Eligible(Expr):
        def __init__(self) -> None: ...
        def first(self, value: i64 = Source("first")) -> i64: ...  # type: ignore[assignment]

        @classmethod
        def named(cls, value: i64 = Source("named")) -> "Eligible": ...  # type: ignore[assignment]

    assert isinstance(Eligible().first(), i64)
    assert calls == ["first", "named"]


def test_initializer_default_completes_forward_classmethods(class_authoring_without_engine):
    class Source:
        pass

    calls = []

    def convert_source(value):
        calls.append("conversion")
        return Initialized.first()

    converter(Source, i64, convert_source)

    class Initialized(Expr):
        def __init__(self, value: i64 = Source()) -> None: ...  # type: ignore[assignment]

        @classmethod
        def first(cls) -> i64:
            calls.append("first")
            return cls.later()

        @classmethod
        def later(cls) -> i64:
            calls.append("later")
            return i64(2)

    expression = Initialized()
    assert calls == ["conversion", "first", "later"]
    assert str(_required_callable_args(expression)[0]) == "Initialized.first()"
    pack([to_runtime_expr(expression).__egg_ref__])
    assert calls == ["conversion", "first", "later"]


@pytest.mark.parametrize("kind", ["merge", "cost"])
@pytest.mark.parametrize("late_bad", [False, True])
def test_member_header_value_callbacks_wait_for_static_barrier(  # noqa: C901
    class_authoring_without_engine, monkeypatch, kind, late_bad
):
    from egglog.egraph import resolve_literal  # noqa: PLC0415

    calls = []
    marker = 123456789

    def literal(sort, value):
        if value is marker:
            calls.append("cost")
            Barrier().first()
        return resolve_literal(sort, value)

    monkeypatch.setattr("egglog.egraph.resolve_literal", literal)

    def merge(old, new):
        calls.append("merge")
        return Barrier().first() + old + new

    class Barrier(Expr):
        def __init__(self) -> None: ...

        def first(self) -> i64:
            calls.append("first")
            return self.later()

        if kind == "merge":

            @method(merge=merge)
            def configured(self) -> i64: ...

        else:

            @method(cost=marker)
            def configured(self) -> "Barrier": ...

        def later(self) -> i64:
            calls.append("later")
            return i64(2)

        if late_bad:

            @method(cost=1)  # type: ignore[type-var]
            def invalid(self) -> i64: ...

    runtime_barrier, barrier_context = _required_authored_class(Barrier)
    if late_bad:
        with pytest.raises(ValueError, match="cannot use cost"):
            Barrier()
        assert calls == []
        assert barrier_context.catalog is not None
        assert {name for _, name in barrier_context.catalog.members} == {"__init__"}
        assert not runtime_barrier.__egg_attr_cache__
    else:
        reference = resolve_callable(Barrier.configured)
        assert reference.read().kind.field == ("function" if kind == "merge" else "constructor")
        assert calls == ["first", "later", kind]
        pack([reference])
        assert calls == ["first", "later", kind]


@pytest.mark.parametrize("option", ["merge", "mutates_self", "unextractable", "subsume", "preserve"])
def test_initializer_cost_does_not_enable_other_rejected_options(class_authoring_without_engine, option):
    calls = []

    def merge(old, new):
        calls.append("merge")
        return old

    options = {option: merge if option == "merge" else True}

    class Invalid(Expr):
        @method(cost=3, **options)  # type: ignore[call-overload]
        def __init__(self, value: i64) -> None:
            calls.append("body")

    with pytest.raises(NotImplementedError, match="Method options"):
        Invalid(1)
    assert calls == []
    assert not Invalid.__egg_context__.catalog.members


@pytest.mark.parametrize("explicit_ruleset", [False, True])
def test_initializer_cost_waits_for_all_static_headers(class_authoring_without_engine, monkeypatch, explicit_ruleset):
    from egglog.egraph import resolve_literal  # noqa: PLC0415

    calls = []
    destination = ruleset()
    marker = 123456789

    def literal(sort, value):
        if value is marker:
            calls.append("cost")
        return resolve_literal(sort, value)

    monkeypatch.setattr("egglog.egraph.resolve_literal", literal)

    class Invalid(Expr, ruleset=destination if explicit_ruleset else None):
        @method(cost=marker)
        def __init__(self, value: i64) -> None: ...

        @method(cost=1)  # type: ignore[type-var]
        def invalid(self) -> i64: ...

    _runtime_invalid, invalid_context = _required_authored_class(Invalid)
    with pytest.raises(ValueError, match="Primitive-returning callables cannot use cost"):
        Invalid(1)  # type: ignore[arg-type]
    assert calls == []
    assert invalid_context.catalog is not None
    assert not invalid_context.catalog.members
    assert not destination._snapshot().read().kind.value.rules


@pytest.mark.parametrize("explicit", [False, True])
def test_member_default_barrier_preserves_explicit_rulesets(class_authoring_without_engine, explicit):
    class Result(Expr):
        def __init__(self) -> None: ...

    class Source:
        pass

    calls = []
    destination = ruleset()

    def convert_source(value):
        calls.append("conversion")
        return Defaults().first()

    converter(Source, Result, convert_source)

    class Defaults(Expr, ruleset=destination if explicit else None):
        def __init__(self) -> None: ...

        def first(self) -> Result:
            calls.append("first")
            return self.later()

        def defaulted(self, value: Result = Source()) -> Result: ...  # type: ignore[assignment]

        def later(self) -> Result:
            calls.append("later")
            return Result()

    assert isinstance(Defaults().defaulted(), Result)
    assert calls == (["conversion", "first", "later"] if explicit else ["first", "later", "conversion"])
    assert len(destination._snapshot().read().kind.value.rules) == (2 if explicit else 0)
    pack([to_runtime_expr(Defaults().defaulted()).__egg_ref__])


@pytest.mark.parametrize(
    ("annotation", "error", "message"),
    [("i64", NotImplementedError, "expected a ClassVar"), ("ClassVar[i64] = None", ConvertError, "Cannot convert")],
)
def test_invalid_class_annotations_are_explicit(class_authoring_without_engine, annotation, error, message):
    namespace = {"Expr": Expr, "ClassVar": ClassVar, "i64": i64, "__name__": __name__}
    exec(f"class Invalid(Expr):\n    VALUE: {annotation}\n", namespace)
    cls, context = _required_authored_class(namespace["Invalid"])
    with pytest.raises(error, match=message):
        cls.VALUE
    assert context.catalog is not None
    assert not context.catalog.members
    with pytest.raises(AttributeError):
        _ = cls.VALUE


def test_classvar_eager_defaults_are_closed_nullary_primitives(class_authoring_without_engine):
    from egglog import Maybe  # noqa: PLC0415

    class DefaultBox(Expr):
        def __init__(self, value: i64) -> None: ...

    scalar, box = i64(1), DefaultBox(2)  # type: ignore[arg-type]
    before = pack([scalar.__egg_ref__, to_runtime_expr(box).__egg_ref__]).program.to_binary()

    class Defaults(Expr):
        COUNT: ClassVar[i64] = scalar
        BOX: ClassVar[DefaultBox] = box
        MISSING: ClassVar[Maybe[i64] | None] = None

    runtime_defaults, _defaults_context = _required_authored_class(Defaults)
    defaults_sort = runtime_defaults.__egg_sort__
    assert defaults_sort is not None
    ambient = ruleset()
    with set_current_ruleset(ambient):
        values = [(Defaults.COUNT, scalar), (Defaults.BOX, box), (Defaults.MISSING, Maybe[i64].none())]
        assert get_current_ruleset() is ambient
    assert not ambient._snapshot().read().kind.value.rules
    for value, expected in values:
        assert value is not None
        declaration = resolve_callable(value)
        record = declaration.read()
        primitive = record.kind.value
        assert isinstance(primitive, pb.Primitive)
        assert not primitive.inputs
        assert expr_parts(RuntimeExpr(declaration.owner.ref("nodes", primitive.body))) == expr_parts(expected)
        view = record.bindings.python.views[0]
        assert view.kind == pb.PythonCallKind.CLASS_VARIABLE
        assert not view.params
        assert StructuralView(declaration.owner.ref("sorts", view.owner.kind.value)) == StructuralView(defaults_sort)
        assert str(value) == f"Defaults.{view.path[0]}"
        with pytest.raises(TypeError, match="table-backed Function"):
            set_(value).to(expected)
        packed = pack([to_runtime_expr(value).__egg_ref__])
        encoded = packed.program.to_binary()
        decoded = pb.Program.from_binary(encoded)
        restored = RuntimeExpr(Builder.from_program(decoded).publish().ref("nodes", packed.indices[0]))
        assert expr_parts(restored) == expr_parts(value)
        assert str(restored) == str(value)
        assert decoded.to_binary() == encoded
    assert pack([scalar.__egg_ref__, to_runtime_expr(box).__egg_ref__]).program.to_binary() == before


def test_classvar_eager_default_conversion_keeps_completed_prefix(class_authoring_without_engine):
    class Source:
        def __init__(self, valid: bool) -> None:
            self.valid = valid

    events = []

    def convert_source(value):
        events.append(get_current_ruleset())
        if value.valid:
            return i64(1)
        assert str(Partial.FIRST) == "Partial.FIRST"
        msg = "class default conversion marker"
        raise ValueError(msg)

    converter(Source, i64, convert_source)

    class Partial(Expr):
        FIRST: ClassVar[i64] = Source(True)  # type: ignore[assignment]
        BAD: ClassVar[i64] = Source(False)  # type: ignore[assignment]
        LATER: ClassVar[i64] = i64(3)

    ambient = ruleset()
    with set_current_ruleset(ambient):
        with pytest.raises(ValueError, match="class default conversion marker"):
            _ = Partial.FIRST
        assert get_current_ruleset() is ambient
    assert events == [None, None]
    assert isinstance(Partial.FIRST, i64)
    assert Partial.FIRST is Partial.FIRST
    for name in ("BAD", "LATER"):
        with pytest.raises(AttributeError):
            getattr(Partial, name)
    assert events == [None, None]
    assert not ambient._snapshot().read().kind.value.rules


@pytest.mark.parametrize("spelling", ["external", "_0"])
def test_classvar_eager_defaults_reject_unbound_query_variables(class_authoring_without_engine, spelling):
    class Invalid(Expr):
        VALUE: ClassVar[i64] = var(spelling, i64)

    _runtime_invalid, invalid_context = _required_authored_class(Invalid)
    with pytest.raises(ValueError, match=r"Free variable.*named eager body"):
        _ = Invalid.VALUE
    assert invalid_context.catalog is not None
    assert not invalid_context.catalog.members
    with pytest.raises(AttributeError):
        _ = Invalid.VALUE


def test_retained_member_context_lifetime_and_local_hooks():
    def make():
        class Local(Expr):
            ROOT: ClassVar["Local"]

            def child(self) -> "Local": ...

            @method(preserve=True)
            def tag(self):
                return "retained member"

        runtime_local, local_context = _required_authored_class(Local)
        expression = Local.ROOT.child()
        return expression, weakref(Local), weakref(local_context), weakref(runtime_local.__egg_definition__.owner)

    expression, cls, context, owner = make()
    gc.collect()
    assert cls() is not None
    cls().__egg_attr_cache__.clear()
    cls().__egg_context__.catalog.reindex()
    assert expression.tag() == "retained member"
    assert expression.child().tag() == "retained member"
    del expression
    gc.collect()
    assert cls() is context() is owner() is None


def test_member_default_union_freshness_and_snapshot():
    class Child(Expr):
        def __init__(self) -> None: ...

    builder = Builder()
    child = builder.import_ref(Child().__egg_ref__)
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Child.__egg_sort__),  # type: ignore[attr-defined]
            kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[child, child])),
        ),
    )
    default = RuntimeExpr(builder.publish().ref("nodes", index))

    class Parent(Expr):
        def __init__(self) -> None: ...
        def child(self, value: Child = default) -> Child: ...  # type: ignore[assignment]

    expression = Parent()
    first = _required_callable_args(expression.child())[1]
    default.__replace_expr__(to_runtime_expr(Child()))
    second = _required_callable_args(expression.child())[1]
    assert first.__egg_ref__ != second.__egg_ref__
    for result in (first, second):
        record = result.__egg_ref__.read()
        assert record.kind.field == "union"
        left, right = record.kind.value.members
        assert result.__egg_ref__.owner.ref("nodes", left) == result.__egg_ref__.owner.ref("nodes", right)


@pytest.mark.parametrize("options", [{"cost": 1}, {"unextractable": True}, {"subsume": True}])
def test_primitive_member_option_errors_are_not_constructor_restrictions(options):
    class Invalid(Expr):
        def __init__(self) -> None: ...

        @method(**options)
        def primitive(self) -> i64: ...

    _runtime_invalid, invalid_context = _required_authored_class(Invalid)
    with pytest.raises(ValueError, match=r"Primitive-returning callables cannot (use|be)"):
        Invalid()
    assert invalid_context.catalog is not None
    assert {name for _, name in invalid_context.catalog.members} == {"__init__"}
    assert isinstance(Invalid(), Invalid)
    with pytest.raises(AttributeError):
        _ = Invalid.primitive


def test_class_ruleset_bodyless_members_do_not_create_rewrites():
    destination = ruleset(name="class-rules")

    class Bodyless(Expr, ruleset=destination):
        def __init__(self) -> None: ...
        def value(self) -> i64: ...

    assert isinstance(Bodyless().value(), i64)
    assert destination._snapshot().read().kind.value.rules == []


@pytest.fixture
def class_authoring_without_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Class authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)


@pytest.mark.parametrize("style", ["unbound", "attribute", "symbolic"])
def test_symbolic_getattr_uses_existing_header(class_authoring_without_engine, style):
    class A(Expr):
        def __init__(self) -> None: ...
        def __getattr__(self, name: StringLike) -> "A": ...

    _runtime_a, a_context = _required_authored_class(A)
    receiver = A()
    original = receiver.__egg_ref__
    name = String("a") + "b" if style == "symbolic" else String("foo")
    result = (
        A.__getattr__(receiver, name)
        if style == "unbound"
        else receiver.foo
        if style == "attribute"
        else receiver.__getattr__(name)
    )
    declaration = resolve_callable(A.__getattr__)
    record = declaration.read()
    view = record.bindings.python.views[0]
    assert record.kind.field == "constructor"
    assert record.kind.value.output == record.kind.value.inputs[0].sort
    assert [argument.name for argument in record.kind.value.inputs] == ["self", "name"]
    assert view.kind == pb.PythonCallKind.METHOD
    assert view.receiver == 0
    assert [(parameter.name, parameter.core_input) for parameter in view.params] == [("name", 1)]
    assert not view.has_field("mutates")
    assert get_callable_fn(result) == A.__getattr__
    arguments = get_callable_args(result, A.__getattr__)
    assert arguments is not None
    assert expr_parts(arguments[0]) == expr_parts(receiver)
    assert expr_parts(arguments[1]) == expr_parts(name)
    if style == "symbolic":
        assert arguments[1].__egg_ref__ == name.__egg_ref__
    assert receiver.__egg_ref__ is original
    assert a_context.catalog is not None
    assert {member for _, member in a_context.catalog.members} == {"__init__", "__getattr__"}


def test_getattr_preserves_real_members_and_none(class_authoring_without_engine):
    calls = []

    class A(Expr):
        def __init__(self) -> None: ...
        def real(self) -> "A": ...

        @property
        def symbolic_property(self) -> "A": ...

        @method(preserve=True)
        def host(self):
            return "host method"

        @method(preserve=True)  # type: ignore[prop-decorator]
        @property
        def empty(self):
            return None

        __hash__ = None  # type: ignore[assignment]

        @method(preserve=True)
        def __getattr__(self, name) -> object:
            calls.append(name)
            return None if name == "nothing" else ("fallback", name)

    _runtime_a, a_context = _required_authored_class(A)
    receiver = A()
    assert get_callable_fn(receiver.real()) == A.real
    assert get_callable_fn(receiver.symbolic_property) == A.symbolic_property
    assert receiver.host() == "host method"
    assert receiver.empty is None
    assert receiver.__getattr__("__hash__") is None
    assert calls == []
    assert receiver.foo == ("fallback", "foo")
    assert receiver.nothing is None
    assert calls == ["foo", "nothing"]
    assert a_context.catalog is not None
    assert {member for _, member in a_context.catalog.members} == {
        "__init__",
        "real",
        "symbolic_property",
    }
    with pytest.raises(AttributeError):
        _ = A.foo  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = receiver.__egg_missing__
    assert calls == ["foo", "nothing"]


@pytest.mark.parametrize("descriptor", [property, classmethod])
def test_getattr_ineligible_preserved_descriptors_are_inert(class_authoring_without_engine, descriptor):
    class Invalid(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        @descriptor
        def __getattr__(self, *args) -> object:
            pytest.fail("Ineligible fallback descriptor was evaluated")

    receiver = Invalid()
    with pytest.raises(AttributeError, match="not an instance method"):
        _ = receiver.foo
    with pytest.raises(AttributeError, match="not an instance method"):
        receiver.__getattr__(String("foo"))


@pytest.mark.parametrize("kind", ["property", "classmethod", "class_variable"])
def test_getattr_ineligible_symbolic_views_are_inert(class_authoring_without_engine, monkeypatch, kind):
    class Invalid(Expr):
        def __init__(self) -> None: ...

        if kind == "property":

            @property
            def __getattr__(self) -> "Invalid": ...  # type: ignore[misc]  # noqa: PLE0302 -- deliberately ineligible descriptor

        elif kind == "classmethod":

            @classmethod
            def __getattr__(cls, name: StringLike) -> "Invalid": ...  # type: ignore[misc]

        else:
            __getattr__: ClassVar["Invalid"]  # type: ignore[no-redef]

    receiver = Invalid()
    if kind == "class_variable":
        runtime_invalid, _invalid_context = _required_authored_class(Invalid)
        cached = to_runtime_expr(Invalid.__getattr__)  # type: ignore[arg-type]
        assert isinstance(cached, RuntimeExpr)
        assert runtime_invalid.__egg_attr_cache__["__getattr__"] is cached

    def forbidden(*args, **kwargs):
        pytest.fail("Ineligible fallback view was invoked")

    monkeypatch.setattr(RuntimeFunction, "__call__", forbidden)
    with pytest.raises(AttributeError, match="not an instance method"):
        _ = receiver.foo
    with pytest.raises(AttributeError, match="not an instance method"):
        receiver.__getattr__(String("foo"))  # type: ignore[operator]


@pytest.mark.parametrize("source", ["property", "fallback", "prepare", "conversion"])
def test_getattr_propagates_original_errors(class_authoring_without_engine, source):
    marker = AttributeError(f"original {source} error")
    calls = []

    class A(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)  # type: ignore[prop-decorator]
        @property
        def broken(self):
            raise marker

        @method(preserve=True)
        def __getattr__(self, name) -> object:
            calls.append(name)
            raise marker

    _runtime_a, a_context = _required_authored_class(A)
    if source == "prepare":

        def failed_prepare(cls):
            raise marker

        a_context.prepare = failed_prepare
        receiver = var("unprepared", A)
        operation = partial(getattr, receiver, "foo")
    elif source == "conversion":

        class Symbolic(Expr):
            def __init__(self) -> None: ...
            def __getattr__(self, name: StringLike) -> "Symbolic": ...

        class BadName:
            pass

        def failed_conversion(value):
            raise marker

        converter(BadName, String, failed_conversion)
        receiver = Symbolic()
        operation = partial(receiver.__getattr__, BadName())  # type: ignore[arg-type]
    else:
        receiver = A()
        operation = partial(getattr, receiver, "broken" if source == "property" else "foo")
    with pytest.raises(AttributeError) as error:
        operation()
    assert error.value is marker
    assert calls == (["foo"] if source == "fallback" else [])


def test_getattr_does_not_synthesize_protocols_or_captures(class_authoring_without_engine):
    class A(Expr):
        def __init__(self) -> None: ...
        def __getattr__(self, name: StringLike) -> "A": ...

    receiver = A()

    class Preserved(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __getattr__(self, name) -> object:
            pytest.fail("Missing protocol or capture invoked dynamic fallback")

    for value in (receiver, Preserved()):
        for operation in (
            lambda value=value: -value,
            lambda value=value: value(),
            lambda value=value: value.__radd__(value),
            lambda value=value: UnstableFn(value.__neg__),
        ):
            with pytest.raises(AttributeError):
                operation()
    assert expr_parts(UnstableFn(receiver.__getattr__)) == expr_parts(UnstableFn(A.__getattr__, receiver))
    original = receiver.__egg_ref__
    with pytest.raises(TypeError):
        A.__getattr__(receiver)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        A.__getattr__(receiver, "foo", "bar")  # type: ignore[call-arg]
    with pytest.raises(ConvertError):
        receiver.__getattr__(i64(1))  # type: ignore[arg-type]
    assert receiver.__egg_ref__ is original
    with pytest.raises(AttributeError):
        _ = i64(1).absent  # type: ignore[attr-defined]


def test_getattr_does_not_swallow_metadata_errors(class_authoring_without_engine):
    calls = []

    class A(Expr):
        def __init__(self) -> None: ...
        def real(self) -> "A": ...

        @method(preserve=True)
        def __getattr__(self, name) -> object:
            calls.append(name)
            return name

    runtime_a, a_context = _required_authored_class(A)
    receiver = A()
    catalog = a_context.catalog
    assert catalog is not None
    key = next(key for key in catalog.members if key[1] == "real")
    reference, _ = catalog.members[key]
    catalog.members[key] = reference, 999
    # Exercise uncached resolution rather than a previously prepared callable.
    runtime_a.__egg_attr_cache__.pop("real", None)
    with pytest.raises(IndexError):
        _ = receiver.real
    assert calls == []


@pytest.mark.parametrize("strip_metadata", [False, True])
def test_getattr_does_not_attach_preserved_hooks_to_detached_records(class_authoring_without_engine, strip_metadata):
    class A(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __getattr__(self, name) -> object:
            pytest.fail("Detached records recovered a Python-only fallback hook")

    runtime_a, a_context = _required_authored_class(A)
    receiver = A()
    assert "__getattr__" in runtime_a.__egg_hooks__
    assert a_context.catalog is not None
    assert {member for _, member in a_context.catalog.members} == {"__init__"}
    packed = pack([receiver.__egg_ref__])
    program = pb.Program.from_binary(packed.program.to_binary())
    if strip_metadata:
        for declaration in program.declarations:
            declaration.bindings = None
            assert declaration.kind is not None
            if declaration.kind.field == "eq_sort":
                declaration.kind.value.bindings = None
    detached = RuntimeExpr(Builder.from_program(program).publish().ref("nodes", packed.indices[0]))
    with pytest.raises(AttributeError):
        _ = detached.foo
    with pytest.raises(AttributeError):
        detached.__getattr__(String("foo"))  # type: ignore[arg-type]


def test_getattr_inspection_keeps_symbolic_name_inert(class_authoring_without_engine, monkeypatch):
    class A(Expr):
        def __init__(self) -> None: ...
        def __getattr__(self, name: StringLike) -> "A": ...

    receiver = A()
    name = String("a") + "b"
    expression = receiver.__getattr__(name)

    def forbidden(*args, **kwargs):
        pytest.fail("Inspection evaluated or converted a symbolic attribute name")

    monkeypatch.setattr("egglog.conversion.resolve_literal", forbidden)
    assert get_callable_fn(expression) == A.__getattr__
    arguments = get_callable_args(expression, A.__getattr__)
    assert arguments is not None
    assert arguments[0].__egg_ref__ == receiver.__egg_ref__
    assert arguments[1].__egg_ref__ == name.__egg_ref__
    assert get_callable_fn(arguments[1]) == join
    assert [part.value for part in _required_callable_args(arguments[1], join)] == ["a", "b"]


def test_bodyless_mutating_members_use_canonical_calls(class_authoring_without_engine):
    class A(Expr):
        def __init__(self) -> None: ...
        def __delitem__(self, key: "A") -> None: ...
        def __setitem__(self, key: "A", value: "A") -> None: ...

    receiver, key, value = A(), A(), A()
    alias, snapshot = receiver, copy(receiver)
    original = to_runtime_expr(receiver).__egg_ref__
    key_root, value_root = to_runtime_expr(key).__egg_ref__, to_runtime_expr(value).__egg_ref__
    for name, parameters in [("__delitem__", ["key"]), ("__setitem__", ["key", "value"])]:
        declaration = resolve_callable(getattr(A, name))
        record = declaration.read()
        assert record.kind.field == "constructor"
        signature = record.kind.value
        view = record.bindings.python.views[0]
        assert signature.output == signature.inputs[0].sort
        assert [argument.name for argument in signature.inputs] == ["self", *parameters]
        assert view.kind == pb.PythonCallKind.METHOD
        assert view.receiver == view.mutates == 0
        assert view.has_field("receiver")
        assert view.has_field("mutates")
        assert [(parameter.name, parameter.core_input) for parameter in view.params] == [
            (name, index) for index, name in enumerate(parameters, start=1)
        ]
    assert receiver.__delitem__(key) is None  # type: ignore[func-returns-value]
    assert alias is receiver
    assert to_runtime_expr(receiver).__egg_ref__ is not original
    assert to_runtime_expr(snapshot).__egg_ref__ == original
    assert to_runtime_expr(key).__egg_ref__ is key_root
    assert to_runtime_expr(value).__egg_ref__ is value_root
    assert [expr_parts(argument) for argument in _required_callable_args(receiver, A.__delitem__)] == [
        expr_parts(snapshot),
        expr_parts(key),
    ]
    deleted = copy(receiver)
    assert A.__setitem__(receiver, key, value) is None  # type: ignore[func-returns-value]
    assert [expr_parts(argument) for argument in _required_callable_args(receiver, A.__setitem__)] == [
        expr_parts(deleted),
        expr_parts(key),
        expr_parts(value),
    ]
    assert to_runtime_expr(snapshot).__egg_ref__ == original
    assert expr_parts(alias) == expr_parts(receiver)


def test_bodyless_free_mutation_owns_calls_and_aliases(class_authoring_without_engine):
    class Mutable(Expr):
        def __init__(self, value: i64) -> None: ...

    @function(mutates_first_arg=True)
    def update(value: Mutable, amount: i64 = 4) -> None: ...  # type: ignore[assignment]

    reference = resolve_callable(update)
    declaration = reference.read()
    assert declaration.kind.field == "constructor"
    signature = declaration.kind.value
    view = declaration.bindings.python.views[0]
    assert signature.output == signature.inputs[0].sort
    assert view.kind == pb.PythonCallKind.FUNCTION
    assert view.has_field("mutates")
    assert view.mutates == 0
    assert not view.has_field("receiver")
    value = Mutable(1)  # type: ignore[arg-type]
    alias, original = value, copy(value)
    assert update(value) is None
    assert alias is value
    assert get_callable_fn(value) == update
    assert [expr_parts(argument) for argument in _required_callable_args(value)] == [
        expr_parts(original),
        expr_parts(i64(4)),
    ]
    assert expr_parts(original) == expr_parts(Mutable(1))  # type: ignore[arg-type]
    encoded = pack([to_runtime_expr(value).__egg_ref__]).program.to_binary()
    assert pb.Program.from_binary(encoded).to_binary() == encoded
    root = to_runtime_expr(value).__egg_ref__
    for args, kwargs, error in [
        ((), {}, TypeError),
        ((value, 2), {"amount": 3}, TypeError),
        ((value, String("wrong")), {}, ConvertError),
        ((i64(1),), {}, ConvertError),
    ]:
        with pytest.raises(error):
            update(*args, **kwargs)
        assert to_runtime_expr(value).__egg_ref__ is root


def test_canonical_mutation_display_keeps_sharing_and_reverse_slots(class_authoring_without_engine):
    class Mutable(Expr):
        def __init__(self, value: i64) -> None: ...
        def __add__(self, other: "Mutable") -> "Mutable": ...
        def __setitem__(self, key: i64, value: i64) -> None: ...
        def __delitem__(self, key: i64) -> None: ...

        @method(mutates_self=True, reverse_args=True)
        def update(self, label: String, amount: i64) -> None: ...

    value = Mutable(1)  # type: ignore[arg-type]
    value.update(String("tag"), i64(2))
    expected = "_Mutable_1 = Mutable(1)\n_Mutable_1.update('tag', 2)\n_Mutable_1"
    assert str(value) == expected
    assert str(value + value) == expected + " + _Mutable_1"
    saved = copy(value)
    value[3] = 4  # type: ignore[assignment, index]
    assert str(saved + value) == (
        "_Mutable_1 = Mutable(1)\n_Mutable_1.update('tag', 2)\n"
        "_Mutable_2 = copy(_Mutable_1)\n_Mutable_2[3] = 4\n_Mutable_1 + _Mutable_2"
    )
    del value[3]  # type: ignore[arg-type]
    assert (
        str(value)
        == "_Mutable_1 = Mutable(1)\n_Mutable_1.update('tag', 2)\n_Mutable_1[3] = 4\ndel _Mutable_1[3]\n_Mutable_1"
    )
    packed = pack([to_runtime_expr(value).__egg_ref__])
    owner = Builder.from_program(pb.Program.from_binary(packed.program.to_binary())).publish()
    assert str(RuntimeExpr(owner.ref("nodes", packed.indices[0]))) == str(value)


def test_eager_free_mutation_uses_original_formals_once(class_authoring_without_engine):
    class Mutable(Expr):
        def __init__(self, value: i64) -> None: ...

    calls = []

    @function(mutates_first_arg=True)
    def replace(value: Mutable, other: Mutable) -> None:
        calls.append(get_current_ruleset())
        value.__replace_expr__(other)
        other.__replace_expr__(Mutable(9))  # type: ignore[arg-type]

    reference = resolve_callable(replace)
    primitive = reference.read().kind.value
    assert isinstance(primitive, pb.Primitive)
    body = reference.owner.ref("nodes", primitive.body)
    assert body.read().kind.field == "var"
    assert body.read().kind.value == "_1"
    value, other = Mutable(1), Mutable(2)  # type: ignore[arg-type]
    original, other_root = copy(value), to_runtime_expr(other).__egg_ref__
    assert replace(value, other) is None
    assert replace(value, other) is None
    assert calls == [None]
    assert expr_parts(original) == expr_parts(Mutable(1))  # type: ignore[arg-type]
    assert to_runtime_expr(other).__egg_ref__ is other_root
    assert reference.read().bindings.python.views[0].has_field("mutates")
    encoded = pack([to_runtime_expr(value).__egg_ref__]).program.to_binary()
    assert pb.Program.from_binary(encoded).to_binary() == encoded


@pytest.mark.parametrize("outcome", ["value", "wrong-sort", "raise"])
def test_eager_free_mutation_failure_is_terminal_before_rebinding(class_authoring_without_engine, outcome):
    class Mutable(Expr):
        def __init__(self) -> None: ...

    marker = RuntimeError("mutation body marker")
    calls = []

    @function(mutates_first_arg=True)
    def invalid(value: Mutable) -> None:
        calls.append(1)
        if outcome == "value":
            return value  # type: ignore[return-value]
        if outcome == "wrong-sort":
            value.__replace_expr__(i64(1))  # type: ignore[arg-type]
        else:
            raise marker
        return None

    value = Mutable()
    root = to_runtime_expr(value).__egg_ref__
    error = {"value": ValueError, "wrong-sort": ConvertError, "raise": RuntimeError}[outcome]
    for _ in range(2):
        with pytest.raises(error) as caught:
            invalid(value)
        if outcome == "raise":
            assert caught.value is marker
        assert to_runtime_expr(value).__egg_ref__ is root
    assert calls == [1]
    runtime_invalid = _required_runtime_function(invalid)
    assert runtime_invalid.__egg_declaration__ is None


@pytest.mark.parametrize("invalid", ["return", "missing-input"])
def test_free_mutation_signature_rejects_before_body(class_authoring_without_engine, invalid):
    if invalid == "return":

        def body(value: i64) -> i64:
            pytest.fail("Invalid mutation return annotation executed its body")

    else:

        def body() -> None:  # type: ignore[misc]
            pytest.fail("Mutation without an input executed its body")

    wrapped = function(mutates_first_arg=True)(body)
    with pytest.raises(ValueError, match="Mutating callables"):
        resolve_callable(wrapped)
    runtime_wrapped = _required_runtime_function(wrapped)
    assert runtime_wrapped.__egg_declaration__ is None


@pytest.mark.parametrize("corruption", ["index", "output", "initializer", "unmapped"])
def test_mutation_metadata_rejects_before_argument_callbacks(class_authoring_without_engine, corruption):
    @function(mutates_first_arg=True)
    def update(value: i64) -> None: ...

    packed = pack([resolve_callable(update)])
    program = packed.program
    declaration = program.declarations[packed.indices[0]]
    view = _required_python_bindings(declaration).views[0]
    if corruption == "index":
        view.mutates = 1
    elif corruption == "output":
        function_record = _required_oneof(declaration.kind, "function", pb.Function)
        function_record.output = len(program.sorts)
        program.sorts.append(pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "wrong-output")))
    elif corruption == "initializer":
        view.kind = pb.PythonCallKind.INITIALIZER
    else:
        view.params.clear()
    owner = Builder.from_program(program).publish()

    def forbidden(*args):
        pytest.fail("Malformed mutation metadata reached an argument callback")

    with pytest.raises(ValueError, match="Invalid mutated input"):
        _author_call(
            owner.ref("declarations", packed.indices[0]), 0, (object(),), {}, node_of=forbidden, convert=forbidden
        )


@pytest.mark.parametrize("target", [None, 1, i64(1)])
def test_mutation_target_validation_precedes_conversions(class_authoring_without_engine, target):
    calls = []

    class Input:
        pass

    def convert_input(value):
        calls.append(value)
        return i64(2)

    converter(Input, i64, convert_input)

    class Mutable(Expr):
        def __init__(self) -> None: ...

        @method(mutates_self=True, reverse_args=True)
        def update(self, amount: i64) -> None: ...

    wrong = to_runtime_expr(target).__egg_ref__ if isinstance(target, RuntimeExpr) else None
    with pytest.raises(ConvertError if wrong is not None else TypeError):
        Mutable.update(target, Input())  # type: ignore[arg-type]
    assert calls == []
    if wrong is not None:
        assert to_runtime_expr(target).__egg_ref__ is wrong


def test_bodyless_named_mutation_defaults_and_errors(class_authoring_without_engine):
    default = i64(4)
    destination = ruleset()

    class Accumulator(Expr, ruleset=destination):
        def __init__(self) -> None: ...

        @method(mutates_self=True, egg_fn="exact mutable member")
        def update(self, left: i64 = default, right: i64 = i64(5)) -> None:
            """A bodyless mutating declaration."""

        @method(mutates_self=True)
        def reset(self) -> None:
            return None

    runtime_accumulator, accumulator_context = _required_authored_class(Accumulator)
    receiver = Accumulator()
    declaration = resolve_callable(Accumulator.update)
    assert declaration.read().kind.value.name == "exact mutable member"
    assert Accumulator.update.__doc__ == "A bodyless mutating declaration."
    assert destination._snapshot().read().kind.value.rules == []
    default.__replace_expr__(i64(9))
    original = copy(receiver)
    assert receiver.update() is None
    arguments = _required_callable_args(receiver, Accumulator.update)
    assert expr_parts(arguments[0]) == expr_parts(original)
    assert [argument.value for argument in arguments[1:]] == [4, 5]
    prior = copy(receiver)
    assert Accumulator.update(receiver, right=7, left=6) is None  # type: ignore[arg-type]
    arguments = _required_callable_args(receiver, Accumulator.update)
    assert expr_parts(arguments[0]) == expr_parts(prior)
    assert [argument.value for argument in arguments[1:]] == [6, 7]
    runtime_accumulator.__egg_attr_cache__.clear()
    assert accumulator_context.catalog is not None
    accumulator_context.catalog.members.clear()
    accumulator_context.catalog.reindex()
    assert resolve_callable(Accumulator.update) == declaration
    root = to_runtime_expr(receiver).__egg_ref__
    for args, kwargs, error in [
        ((1, 2, 3), {}, TypeError),
        ((1,), {"left": 2}, TypeError),
        ((), {"unknown": 1}, TypeError),
        ((String("wrong"),), {}, ConvertError),
    ]:
        with pytest.raises(error):
            receiver.update(*args, **kwargs)  # type: ignore[arg-type]
        assert to_runtime_expr(receiver).__egg_ref__ is root

    class BadInput:
        pass

    def fail_conversion(value):
        msg = "mutating input conversion failed"
        raise ValueError(msg)

    converter(BadInput, i64, fail_conversion)
    with pytest.raises(ValueError, match="mutating input conversion failed"):
        receiver.update(BadInput())  # type: ignore[arg-type]
    assert to_runtime_expr(receiver).__egg_ref__ is root
    wrong_receiver = i64(1)
    wrong_root = to_runtime_expr(wrong_receiver).__egg_ref__
    with pytest.raises(ConvertError):
        Accumulator.update(wrong_receiver)  # type: ignore[arg-type]
    assert to_runtime_expr(wrong_receiver).__egg_ref__ is wrong_root
    assert receiver.reset() is None
    assert _required_callable_args(receiver, Accumulator.reset)[0].__egg_ref__ == root
    assert destination._snapshot().read().kind.value.rules == []


@pytest.mark.parametrize("name", ["__delitem__", "__setitem__", "update"])
def test_bodyless_mutation_reconstructs_without_result_normalization(class_authoring_without_engine, monkeypatch, name):
    class Mutable(Expr):
        def __init__(self) -> None: ...
        def __delitem__(self, key: "Mutable") -> None: ...
        def __setitem__(self, key: "Mutable", value: "Mutable") -> None: ...

        @method(mutates_self=True)
        def update(self, key: "Mutable") -> None: ...

    receiver = Mutable()
    arguments = [Mutable(), Mutable()] if name == "__setitem__" else [Mutable()]
    assert getattr(receiver, name)(*arguments) is None
    packed = pack([to_runtime_expr(receiver).__egg_ref__])
    encoded = packed.program.to_binary()
    decoded = pb.Program.from_binary(encoded)
    root = Builder.from_program(decoded).publish().ref("nodes", packed.indices[0])
    rebuilt = RuntimeExpr(root)

    def forbidden(*args, **kwargs):
        pytest.fail("Canonical mutation reconstruction converted an existing expression")

    monkeypatch.setattr("egglog.conversion.resolve_literal", forbidden)
    function = get_callable_fn(rebuilt)
    assert function is not None
    arguments = _required_callable_args(rebuilt, function)
    assert StructuralView(resolve_callable(function)) == StructuralView(resolve_callable(getattr(Mutable, name)))
    assert len(arguments) == (3 if name == "__setitem__" else 2)
    old_receiver = arguments[0].__egg_ref__
    others = tuple(argument.__egg_ref__ for argument in arguments[1:])
    assert function(*arguments) is None
    assert arguments[0].__egg_ref__ is not old_receiver
    assert expr_parts(arguments[0]) == expr_parts(rebuilt)
    assert tuple(argument.__egg_ref__ for argument in arguments[1:]) == others
    assert rebuilt.__egg_ref__ == root
    assert decoded.to_binary() == encoded


@pytest.mark.parametrize("descriptor", [classmethod, property])
def test_mutation_noninstance_views_remain_rejected(class_authoring_without_engine, descriptor):
    class Invalid(Expr):
        def __init__(self) -> None: ...

        @method(mutates_self=True)
        @descriptor
        def mutate(self) -> "Invalid": ...

    _runtime_invalid, invalid_context = _required_authored_class(Invalid)
    with pytest.raises(NotImplementedError, match="not migrated"):
        Invalid()
    assert invalid_context.catalog is not None
    assert {name for _, name in invalid_context.catalog.members} == {"__init__"}


def test_eager_mutating_member_bodies_remain_rejected_before_callback(class_authoring_without_engine):
    class Invalid(Expr):
        def __init__(self) -> None: ...

        @method(mutates_self=True)
        def mutate(self) -> None:
            pytest.fail("Unsupported mutating member body was executed")

    _runtime_invalid, invalid_context = _required_authored_class(Invalid)
    with pytest.raises(NotImplementedError, match="not migrated"):
        Invalid()
    assert invalid_context.catalog is not None
    assert {name for _, name in invalid_context.catalog.members} == {"__init__"}


@pytest.mark.parametrize("reverse", [False, True])
def test_mutating_member_rewrite_keeps_original_binders(class_authoring_without_engine, reverse):
    destination = ruleset()
    calls = []

    class Mutable(Expr, ruleset=destination):
        def __init__(self, value: i64) -> None: ...

        @method(mutates_self=True, reverse_args=reverse)
        def replace(self, other: "Mutable") -> None:
            calls.append(get_current_ruleset())
            self.__replace_expr__(other)
            other.__replace_expr__(Mutable(9))  # type: ignore[arg-type]

    value, other = Mutable(1), Mutable(2)  # type: ignore[arg-type]
    original = copy(value)
    other_root = to_runtime_expr(other).__egg_ref__
    assert value.replace(other) is None
    assert calls == [destination]
    assert to_runtime_expr(other).__egg_ref__ is other_root
    assert expr_parts(original) == expr_parts(Mutable(1))  # type: ignore[arg-type]
    declaration = resolve_callable(Mutable.replace).read()
    view = declaration.bindings.python.views[0]
    assert view.receiver == view.mutates == int(reverse)
    snapshot = destination._snapshot()
    assert len(snapshot.read().kind.value.rules) == 1
    record = snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])
    rewrite_record = record.read().kind.value
    assert isinstance(rewrite_record, pb.Rewrite)
    lhs = record.owner.ref("nodes", rewrite_record.lhs)
    assert [lhs.owner.ref("nodes", index).read().kind.value for index in lhs.read().kind.value.args] == (
        ["other", "self"] if reverse else ["self", "other"]
    )
    rhs = record.owner.ref("nodes", rewrite_record.rhs)
    assert rhs.read().kind.field == "var"
    assert rhs.read().kind.value == "other"
    encoded = pack([to_runtime_expr(value).__egg_ref__], commands=[snapshot]).program.to_binary()
    assert pb.Program.from_binary(encoded).to_binary() == encoded


def test_class_ruleset_bodyless_callable_headers(class_authoring_without_engine):
    destination = ruleset()

    class Boolean(Expr, ruleset=destination):
        NEVER: ClassVar["Boolean"]

        def __init__(self, value: BoolLike) -> None: ...
        def __invert__(self) -> "Boolean": ...

        @method(preserve=True)
        def local(self):
            return "host hook"

        @classmethod
        def if_(cls, condition: BoolLike, yes: Callable[[], "Boolean"], no: Callable[[], "Boolean"]) -> "Boolean":
            """Docstring-only declarations do not create rewrites."""

    expression = Boolean(True)
    assert _required_callable_args(expression, Boolean)[0].value is True
    assert isinstance(Boolean.NEVER, Boolean)
    assert isinstance(~expression, Boolean)
    assert expression.local() == "host hook"
    runtime_boolean, _boolean_context = _required_authored_class(Boolean)
    boolean_sort = runtime_boolean.__egg_sort__
    assert boolean_sort is not None
    reference = resolve_callable(Boolean.if_)
    inputs = reference.read().kind.value.inputs
    for argument in inputs[1:]:
        sort = reference.owner.ref("sorts", argument.sort)
        assert sort.read().kind.field == "func"
        function_sort = sort.read().kind.value
        assert function_sort.params == []
        assert StructuralView(sort.owner.ref("sorts", function_sort.result)) == StructuralView(boolean_sort)
    assert destination._snapshot().read().kind.value.rules == []

    @function(ruleset=destination)
    def bodyless() -> Boolean: ...

    with pytest.raises(ValueError, match="Explicit rulesets require a body"):
        resolve_callable(bodyless)


@pytest.mark.parametrize("formed_default", [False, True])
def test_class_late_header_failure_keeps_prefix(class_authoring_without_engine, formed_default):
    class HeaderResult(Expr, egg_sort=f"HeaderResult_{formed_default}"):
        def __init__(self, value: i64) -> None: ...

    destination = ruleset()
    events = []

    class Source:
        pass

    def convert_source(value):
        events.append("convert")
        return HeaderResult(1)  # type: ignore[arg-type]

    converter(Source, HeaderResult, convert_source)

    class Partial(Expr, ruleset=destination):
        FIRST: ClassVar[HeaderResult] = HeaderResult(1) if formed_default else Source()  # type: ignore[arg-type, assignment]

        def __init__(self) -> None: ...

        def early(self) -> HeaderResult:
            events.append("body")
            return HeaderResult(2)  # type: ignore[arg-type]

        @method(cost=1)  # type: ignore[type-var]
        def bad(self) -> i64: ...

        def later(self) -> HeaderResult: ...

    _runtime_partial, partial_context = _required_authored_class(Partial)
    with pytest.raises(ValueError, match="Primitive-returning callables cannot use cost"):
        Partial()
    expected = [] if formed_default else ["convert"]
    assert events == expected
    assert isinstance(Partial.FIRST, HeaderResult)
    assert isinstance(Partial().early(), HeaderResult)
    assert len(destination._snapshot().read().kind.value.rules) == 1
    for name in ("bad", "later"):
        with pytest.raises(AttributeError):
            getattr(Partial, name)
    assert events == expected
    assert partial_context.prepare is None


def test_class_first_header_failure_is_terminal_without_poison(class_authoring_without_engine):
    class Partial(Expr):
        @method(cost=1)  # type: ignore[type-var]
        def bad(self) -> i64: ...

        def later(self) -> i64: ...

    runtime_partial, partial_context = _required_authored_class(Partial)
    with pytest.raises(ValueError, match="Primitive-returning callables cannot use cost"):
        _ = Partial.bad
    assert partial_context.catalog is not None
    assert partial_context.catalog.roots == (runtime_partial.__egg_definition__,)
    for name in ("bad", "later"):
        with pytest.raises(AttributeError):
            getattr(Partial, name)


def test_classvar_header_precedes_failing_default_conversion(class_authoring_without_engine):
    class ConversionResult(Expr):
        def __init__(self, value: i64) -> None: ...

    class Source:
        pass

    events = []
    destination = ruleset()

    def convert_source(value):
        events.append(Partial.FIRST)
        msg = "conversion marker"
        raise ValueError(msg)

    converter(Source, ConversionResult, convert_source)

    class Partial(Expr, ruleset=destination):
        FIRST: ClassVar[ConversionResult] = Source()  # type: ignore[assignment]

        def __init__(self) -> None: ...

    with pytest.raises(ValueError, match="conversion marker"):
        _ = Partial.FIRST
    assert events == [Partial.FIRST]
    assert isinstance(Partial.FIRST, ConversionResult)
    assert destination._snapshot().read().kind.value.rules == []
    with pytest.raises(AttributeError):
        Partial()
    assert len(events) == 1


def test_class_default_bodies_see_all_headers_and_keep_failure_prefix(class_authoring_without_engine):
    class BodyResult(Expr):
        def __init__(self, value: i64) -> None: ...

    destination = ruleset()
    events = []

    class Partial(Expr, ruleset=destination):
        FIRST: ClassVar[BodyResult] = BodyResult(1)  # type: ignore[arg-type]

        def __init__(self) -> None: ...

        def early(self, value: i64) -> BodyResult:
            events.append(("early", len(destination._snapshot().read().kind.value.rules)))
            return self.later(value)

        def later(self, value: i64) -> BodyResult:
            events.append(("later", len(destination._snapshot().read().kind.value.rules)))
            msg = "body marker"
            raise ValueError(msg)

        def unvisited(self) -> BodyResult:
            pytest.fail("Body after failure ran")

    with pytest.raises(ValueError, match="body marker"):
        Partial()
    assert events == [("early", 1), ("later", 2)]
    value = Partial()
    assert isinstance(value.early(2), BodyResult)  # type: ignore[arg-type]
    assert isinstance(value.later(3), BodyResult)  # type: ignore[arg-type]
    assert isinstance(value.unvisited(), BodyResult)
    assert events == [("early", 1), ("later", 2)]
    snapshot = destination._snapshot()
    assert len(snapshot.read().kind.value.rules) == 2
    encoded = pack(commands=[snapshot]).program.to_binary()
    assert pb.Program.from_binary(encoded).to_binary() == encoded


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_class_cached_member_lookup_waits_for_authoring_attempt(class_authoring_without_engine, fail, eager):
    entered, release, requested, returned = Event(), Event(), Event(), Event()
    destination = ruleset()
    cached = []

    class Partial(Expr, ruleset=None if eager else destination):
        def __init__(self) -> None: ...

        def earlier(self) -> "Partial": ...

        def body(self) -> "Partial":
            cached.append(Partial.earlier)
            entered.set()
            assert release.wait(5)
            if fail:
                msg = "thread body marker"
                raise ValueError(msg)
            return self

    def read_cached():
        requested.set()
        result = Partial.earlier
        returned.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        compilation = executor.submit(Partial)
        try:
            assert entered.wait(5)
            lookup = executor.submit(read_cached)
            assert requested.wait(5)
            assert not returned.wait(0.05)
        finally:
            release.set()
        if fail:
            with pytest.raises(ValueError, match="thread body marker"):
                compilation.result()
        else:
            assert isinstance(compilation.result(), Partial)
        assert lookup.result() is cached[0]


def test_class_default_views_share_canonical_parameter_binding(class_authoring_without_engine):
    from egglog.egraph import get_current_ruleset  # noqa: PLC0415

    destination = ruleset()
    callbacks: list[tuple[str, type[Expr], Ruleset | None] | tuple[str, Ruleset | None]] = []

    class Defaults(Expr, ruleset=destination):
        def __init__(self, value: i64) -> None: ...

        @classmethod
        def named(cls, value: i64 = 3) -> "Defaults":  # type: ignore[assignment]
            callbacks.append(("named", cls, get_current_ruleset()))
            return cls(value)

        @property
        def own(self) -> "Defaults":
            callbacks.append(("own", get_current_ruleset()))
            return self

        @method(subsume=True)
        def wrapped(self, value: i64 = 4) -> "Defaults":  # type: ignore[assignment]
            callbacks.append(("wrapped", get_current_ruleset()))
            return Defaults(value)

    expression = Defaults(2)  # type: ignore[arg-type]
    assert callbacks == [("named", Defaults, destination), ("own", destination), ("wrapped", destination)]
    assert get_current_ruleset() is None
    assert _required_callable_args(Defaults.named())[0].value == 3
    assert _required_callable_args(expression.wrapped())[1].value == 4
    assert isinstance(expression.own, Defaults)
    snapshot = destination._snapshot()
    records = [snapshot.owner.ref("rules", index) for index in snapshot.read().kind.value.rules]
    assert [record.read().kind.value.subsume for record in records] == [False, False, True]
    for record in records:
        rewrite_record = record.read().kind.value
        lhs_ref = record.owner.ref("nodes", rewrite_record.lhs)
        rhs_ref = record.owner.ref("nodes", rewrite_record.rhs)
        lhs = lhs_ref.read().kind.value
        rhs = rhs_ref.read()
        if lhs.func.endswith("named"):
            assert lhs_ref.owner.ref("nodes", lhs.args[0]).read().kind.value == "value"
            assert [StructuralView(rhs_ref.owner.ref("nodes", arg)) for arg in rhs.kind.value.args] == [
                StructuralView(lhs_ref.owner.ref("nodes", arg)) for arg in lhs.args
            ]
        elif lhs.func.endswith("own"):
            assert rhs.kind.field == "var"
            assert rhs.kind.value == "self"
        else:
            assert lhs_ref.owner.ref("nodes", lhs.args[0]).read().kind.value == "self"
            assert lhs_ref.owner.ref("nodes", lhs.args[1]).read().kind.value == "value"
            assert [StructuralView(rhs_ref.owner.ref("nodes", arg)) for arg in rhs.kind.value.args] == [
                StructuralView(lhs_ref.owner.ref("nodes", lhs.args[1]))
            ]
    assert len(pack(commands=[snapshot]).program.rules) == 3


@pytest.mark.parametrize("replace_receiver", [False, True])
def test_class_default_lhs_keeps_original_binders_after_wrapper_replacement(
    class_authoring_without_engine, replace_receiver
):
    destination = ruleset()

    class Binders(Expr, ruleset=destination, egg_sort=f"DefaultBinders_{replace_receiver}"):
        def __init__(self, value: i64) -> None: ...

        def body(self, value: i64) -> "Binders":
            if replace_receiver:
                self.__replace_expr__(Binders(9))  # type: ignore[arg-type]
                return self
            value.__replace_expr__(i64(7))
            return Binders(value)

    expression = Binders(1).body(2)  # type: ignore[arg-type]
    assert _required_callable_args(expression)[1].value == 2
    snapshot = destination._snapshot()
    assert len(snapshot.read().kind.value.rules) == 1
    reference = snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])
    rewrite_record = reference.read().kind.value
    lhs = reference.owner.ref("nodes", rewrite_record.lhs)
    lhs_args = [lhs.owner.ref("nodes", index) for index in lhs.read().kind.value.args]
    assert [argument.read().kind.field for argument in lhs_args] == ["var", "var"]
    assert [argument.read().kind.value for argument in lhs_args] == ["self", "value"]
    rhs = RuntimeExpr(reference.owner.ref("nodes", rewrite_record.rhs))
    assert _required_callable_args(rhs, Binders)[0].value == (9 if replace_receiver else 7)
    assert len(pack(commands=[snapshot]).program.rules) == 1


@pytest.mark.parametrize("result", [None, i64(2)])
def test_class_invalid_default_body_result_does_not_poison_headers(class_authoring_without_engine, result):
    destination = ruleset()
    calls = []

    class InvalidResult(Expr, ruleset=destination):
        def __init__(self) -> None: ...

        def body(self) -> "InvalidResult":
            calls.append(1)
            return result

    error = ValueError if result is None else ConvertError
    with pytest.raises(error):
        InvalidResult()
    assert isinstance(InvalidResult().body(), InvalidResult)
    assert calls == [1]
    assert destination._snapshot().read().kind.value.rules == []


@pytest.mark.parametrize("default", [None, i64(1)])
def test_class_primitive_default_with_ruleset_remains_invalid(class_authoring_without_engine, default):
    destination = ruleset()

    class InvalidDefault(Expr, ruleset=destination):
        FIRST: ClassVar[i64] = default

    _runtime_invalid_default, invalid_default_context = _required_authored_class(InvalidDefault)
    with pytest.raises(ValueError, match="Primitive-returning defaults cannot use an explicit ruleset"):
        _ = InvalidDefault.FIRST
    assert invalid_default_context.catalog is not None
    assert not invalid_default_context.catalog.members
    assert destination._snapshot().read().kind.value.rules == []
    with pytest.raises(AttributeError):
        _ = InvalidDefault.FIRST


@pytest.mark.parametrize("annotation", [Callable, Callable[..., i64], Callable[[TypeVar("Argument")], i64]])
def test_unresolved_callable_annotations_remain_explicit(class_authoring_without_engine, annotation):
    with pytest.raises(NotImplementedError, match="Callable annotations require"):
        runtime.resolve_type_annotation(annotation)


def test_user_sort_cannot_borrow_builtin_codec_or_members_by_name():
    class Shadow(Expr, egg_sort="i64"):
        def __init__(self, value: i64) -> None: ...

    expression = Shadow(3)  # type: ignore[arg-type]
    assert expression.__egg_ref__.read().kind.field == "call"
    assert _required_callable_args(expression, Shadow)[0].value == 3
    with pytest.raises(AttributeError, match="member"):
        _ = Shadow.__add__  # type: ignore[operator]


def test_unresolved_constructor_annotation_does_not_publish_members():
    class Missing(Expr):
        def __init__(self, value: "NotDefinedHere") -> None: ...  # type: ignore[name-defined]  # noqa: F821

    runtime_missing, missing_context = _required_authored_class(Missing)
    with pytest.raises(NameError, match="NotDefinedHere"):
        Missing(1)
    assert missing_context.catalog is not None
    assert not missing_context.catalog.members
    assert not runtime_missing.__egg_attr_cache__
    with pytest.raises(AttributeError):
        Missing(1)


def test_user_default_expansion_has_fresh_union_identity_and_retains_sharing():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    child = Num(1)  # type: ignore[arg-type]
    builder = Builder()
    member = builder.import_ref(child.__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Num.__egg_sort__),  # type: ignore[attr-defined]
            kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[member, member])),
        ),
    )
    default = RuntimeExpr(builder.publish().ref("nodes", root))

    class Box(Expr):
        def __init__(self, value: Num = default) -> None: ...  # type: ignore[assignment]

    first = _required_callable_args(Box(), Box)[0]
    second = _required_callable_args(Box(), Box)[0]
    assert first.__egg_ref__ != second.__egg_ref__ != default.__egg_ref__
    assert expr_parts(first) != expr_parts(second)
    for expression in (first, second):
        members = expression.__egg_ref__.read().kind.value.members
        assert len(members) == 2
        assert members[0] == members[1]

    @function
    def pair(left: Num = default, right: Num = default) -> Box: ...  # type: ignore[assignment]

    first_pair = _required_callable_args(pair())
    second_pair = _required_callable_args(pair())
    assert first_pair[0].__egg_ref__ == first_pair[1].__egg_ref__
    assert second_pair[0].__egg_ref__ == second_pair[1].__egg_ref__
    assert first_pair[0].__egg_ref__ != second_pair[0].__egg_ref__
    assert first_pair[0].__egg_ref__ != default.__egg_ref__


def test_reconstructed_constructor_uses_changed_protobuf_parameter_not_python_signature():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    program = pack([Num(1).__egg_ref__]).program  # type: ignore[arg-type]
    index, constructor = next(
        (index, declaration)
        for index, declaration in enumerate(program.declarations)
        if (
            declaration.kind.field == "constructor"
            if declaration.kind is not None
            else pytest.fail("Missing declaration kind")
        )
    )
    _required_python_bindings(constructor).views[0].params[0].name = "renamed"
    owner = Builder.from_program(program).publish()
    runtime_num, _num_context = _required_authored_class(Num)
    function = RuntimeFunction(owner.ref("declarations", index), 0, runtime_num)
    result = function(renamed=8)
    assert _required_callable_args(result)[0].value == 8
    with pytest.raises(TypeError):
        function(value=8)


def test_generic_inference_uses_binder_positions_and_is_transactional():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)))
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[0, 1])))
    )
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="String"))))
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[3, 4])))
    )
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[4, 3])))
    )
    owner = builder.publish()
    bindings: dict[int, Ref] = {}
    # GenericSignature labels may repeat: indices, not labels, identify binders.
    signature = pb.GenericSignature(type_params=["T", "T"], inputs=[pb.Arg(sort=2)], output=0)
    infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 5), len(signature.type_params), bindings)
    assert bindings == {0: owner.ref("sorts", 3), 1: owner.ref("sorts", 4)}
    assert StructuralView(substitute_sort(owner.ref("sorts", 2), bindings)) == StructuralView(owner.ref("sorts", 5))
    before = bindings.copy()
    with pytest.raises(TypeConstraintError):
        infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 6), len(signature.type_params), bindings)
    assert bindings == before
    independent: dict[int, Ref] = {}
    infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 6), len(signature.type_params), independent)
    assert independent == {0: owner.ref("sorts", 4), 1: owner.ref("sorts", 3)}


def test_sort_inference_rejects_missing_binders_arity_and_open_actuals():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Box")))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[1], result=1))))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(result=1))))
    owner = builder.publish()
    with pytest.raises(TypeConstraintError, match="binder"):
        infer_sort(owner.ref("sorts", 0), owner.ref("sorts", 1), 0, {})
    with pytest.raises(TypeConstraintError, match="closed"):
        infer_sort(owner.ref("sorts", 0), owner.ref("sorts", 0), 1, {})
    with pytest.raises(TypeConstraintError, match="arity"):
        infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 3), 0, {})
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        substitute_sort(owner.ref("sorts", 0), {})


def test_sort_inference_and_substitution_are_iterative():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Leaf")))
    pattern, concrete = 0, 1
    for _ in range(10_000):
        pattern = builder.add(
            "sorts",
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[pattern]))),
        )
        concrete = builder.add(
            "sorts",
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[concrete]))),
        )
    owner = builder.publish()
    bindings: dict[int, Ref] = {}
    infer_sort(owner.ref("sorts", pattern), owner.ref("sorts", concrete), 1, bindings)
    result = substitute_sort(owner.ref("sorts", pattern), bindings)
    assert StructuralView(result) == StructuralView(owner.ref("sorts", concrete))


def test_call_authoring_binds_keywords_defaults_and_core_input_order():
    builder = Builder()
    sort = builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    default = builder.add(
        "nodes",
        pb.Node(
            sort_id=sort,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 4))
            ),
        ),
    )
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name="sample.subtract",
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature", pb.GenericSignature(inputs=[pb.Arg(sort=sort), pb.Arg(sort=sort)], output=sort)
                    ),
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.FUNCTION,
                            path=["sample", "subtract_reversed"],
                            params=[
                                pb.PythonParameter(core_input=1, name="right"),
                                pb.PythonParameter(core_input=0, name="left", default_expr=default),
                            ],
                        )
                    ]
                )
            ),
        ),
    )
    owner = builder.publish()

    def convert(expected: Ref, value: object) -> Ref:
        assert isinstance(value, int)
        literal = Builder()
        literal.add(
            "nodes",
            pb.Node(
                sort_id=literal.import_ref(expected),
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            ),
        )
        return literal.publish().ref("nodes", 0)

    root, mutated = _author_call(
        owner.ref("declarations", declaration), 0, (), {"right": 9}, node_of=lambda _: None, convert=convert
    )
    assert mutated is None
    assert isinstance(root, Ref)
    node = root.read()
    assert node.kind.value.func == "sample.subtract"
    args = [root.owner.ref("nodes", index) for index in node.kind.value.args]
    assert [arg.read().kind.value.value.value for arg in args] == [4, 9]
    assert args[0] != owner.ref("nodes", default)
    with pytest.raises(TypeError, match="unexpected keyword"):
        _author_call(
            owner.ref("declarations", declaration),
            0,
            (),
            {"right": 9, "bogus": 9},
            node_of=lambda _: None,
            convert=convert,
        )


@pytest.mark.parametrize(("order", "captures"), [((1, 0), ()), ((0, 2, 1), (i64(10),))])
@pytest.mark.parametrize("decoded", [False, True])
def test_function_value_rejects_reordered_remaining_arguments(order, captures, decoded, class_authoring_without_engine):
    builder = Builder()
    integer = runtime.resolve_type_annotation(i64)
    assert isinstance(integer, Ref)
    sort = builder.import_ref(integer)
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function](
                "function",
                pb.Function(name="sample.subtract_reversed", inputs=[pb.Arg(sort=sort) for _ in order], output=sort),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.FUNCTION,
                            path=["sample", "subtract_reversed"],
                            params=[
                                pb.PythonParameter(core_input=core, name=f"arg{position}")
                                for position, core in enumerate(order)
                            ],
                        )
                    ]
                )
            ),
        ),
    )
    if decoded:
        function_sort = builder.add(
            "sorts",
            pb.Sort(
                kind=Oneof[Literal["func"], pb.FuncSort](
                    "func", pb.FuncSort(params=[sort] * (len(order) - len(captures)), result=sort)
                )
            ),
        )
        node = builder.add(
            "nodes",
            pb.Node(
                sort_id=function_sort,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(
                        value=Oneof[Literal["partial_call"], pb.Call](
                            "partial_call",
                            pb.Call(
                                func="sample.subtract_reversed",
                                args=[builder.import_ref(to_runtime_expr(capture).__egg_ref__) for capture in captures],
                            ),
                        )
                    ),
                ),
            ),
        )
        expression = RuntimeExpr(builder.publish().ref("nodes", node))
        with pytest.raises(NotImplementedError, match="Reordered Python function-value presentations"):
            get_literal_value(expression)
    else:
        function = RuntimeFunction(builder.publish().ref("declarations", declaration), 0)
        with pytest.raises(NotImplementedError, match="Reordered Python function-value presentations"):
            runtime._partial_function(function, captures, None)


def test_function_value_rejects_mutation_before_replacing_receiver(class_authoring_without_engine):
    original = to_runtime_expr(i64(10))
    reference = original.__egg_ref__
    builder = Builder()
    sort = builder.import_ref(reference.owner.ref("sorts", reference.read().sort_id))
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function](
                "function", pb.Function(name="sample.mutable", inputs=[pb.Arg(sort=sort)], output=sort)
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.METHOD,
                            path=["mutable"],
                            owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", sort)),
                            receiver=0,
                            mutates=0,
                        )
                    ]
                )
            ),
        ),
    )
    function = RuntimeFunction(builder.publish().ref("declarations", declaration), 0, original)
    with pytest.raises(AssertionError, match="Mutable partial functions not supported"):
        runtime._partial_function(function, (), None)
    assert original.__egg_ref__ is reference
    assert original.value == 10


@pytest.mark.parametrize(
    ("target", "captures", "wrapper"),
    [
        (i64.__add__, (i64(10),), UnstableFn[i64, i64]),
        (Vec[i64].empty, (), UnstableFn[Vec[i64]]),
        (Vec(i64(1)).__getitem__, (), UnstableFn[i64, i64]),
        (join, (), UnstableFn[String, String, String]),
        (Bool.__and__, (Bool(True),), UnstableFn[Bool, Bool]),
    ],
)
def test_named_function_values_roundtrip_actual_canonical_records(
    target, captures, wrapper, class_authoring_without_engine
):
    authored = to_runtime_expr(wrapper(target, *captures))
    packed = pack([authored.__egg_ref__])
    decoded = pb.Program.from_binary(packed.program.to_binary())
    restored = RuntimeExpr(Builder.from_program(decoded).publish().ref("nodes", packed.indices[0]))
    assert StructuralView(restored.__egg_ref__) == StructuralView(authored.__egg_ref__)
    assert str(restored) == str(authored)
    assert get_literal_value(restored) is not None
    assert str(restored.value) == str(authored.value)
    kind = restored.__egg_ref__.read().kind
    assert kind is not None
    assert kind.field == "primitive_value"
    assert kind.value.value is not None
    assert kind.value.value.field == "partial_call"
    sort = restored.__egg_ref__.owner.ref("sorts", restored.__egg_ref__.read().sort_id).read()
    assert sort.kind is not None
    assert sort.kind.field == "func"


@pytest.mark.parametrize("corruption", [None, "capture-sort", "arity", "result-sort", "sort-kind", "callee"])
def test_function_value_zero_indices_and_malformed_records(corruption, class_authoring_without_engine):
    builder = Builder()
    literal = to_runtime_expr(i64(10)).__egg_ref__
    captured = builder.import_ref(literal)
    assert captured == 0
    integer = builder.import_ref(literal.owner.ref("sorts", literal.read().sort_id))
    assert integer == 0
    builder.import_ref(resolve_callable(i64.__add__))
    string = runtime.resolve_type_annotation(String)
    assert isinstance(string, Ref)
    output = builder.import_ref(string) if corruption == "result-sort" else integer
    remaining = [integer, integer] if corruption == "arity" else [integer]
    function_sort = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=remaining, result=output)))
    )
    if corruption == "capture-sort":
        captured = builder.import_ref(to_runtime_expr(String("wrong")).__egg_ref__)
    node = builder.add(
        "nodes",
        pb.Node(
            sort_id=integer if corruption == "sort-kind" else function_sort,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["partial_call"], pb.Call](
                        "partial_call",
                        pb.Call(func="unknown" if corruption == "callee" else "egglog.core.i64.add", args=[captured]),
                    )
                ),
            ),
        ),
    )
    expression = RuntimeExpr(builder.publish().ref("nodes", node))
    if corruption is not None:
        with pytest.raises((ConvertError, TypeConstraintError, TypeError, LookupError)):
            get_literal_value(expression)
    else:
        packed = pack([expression.__egg_ref__])
        restored = RuntimeExpr(
            Builder
            .from_program(pb.Program.from_binary(packed.program.to_binary()))
            .publish()
            .ref("nodes", packed.indices[0])
        )
        value = restored.value
        assert isinstance(value, partial)
        assert StructuralView(resolve_callable(value.func)) == StructuralView(resolve_callable(i64.__add__))
        assert value.func == i64.__add__
        assert hash(value.func) == hash(i64.__add__)
        assert value.args == (i64(10),)
        assert callable(restored)
        assert isinstance(restored(i64(2)), RuntimeExpr)


def test_decoded_builtin_callable_identity_uses_definition_and_selected_view():
    original = i64.__add__
    packed = pack([resolve_callable(original)])
    encoded = packed.program.to_binary()
    left_owner = Builder.from_program(pb.Program.from_binary(encoded)).publish()
    right_owner = Builder.from_program(pb.Program.from_binary(encoded)).publish()
    left_ref = left_owner.ref("declarations", packed.indices[0])
    right_ref = right_owner.ref("declarations", packed.indices[0])
    assert left_ref != right_ref
    left = RuntimeFunction(left_ref, 0, cast("runtime.RuntimeClass", i64))
    right = RuntimeFunction(right_ref, 0, cast("runtime.RuntimeClass", i64))
    assert StructuralView(left_ref) == StructuralView(right_ref)
    assert left == right == original
    assert hash(left) == hash(right) == hash(original)
    assert {left: "same callable"}[right] == "same callable"

    for change in ("name", "signature", "view"):
        program = pb.Program.from_binary(encoded)
        record = program.declarations[packed.indices[0]]
        assert record.kind is not None
        assert record.kind.field == "host_primitive"
        assert record.bindings is not None
        assert record.bindings.python is not None
        if change == "name":
            record.kind.value.name = "test.distinct.provider.add"
        elif change == "signature":
            assert record.kind.value.typing is not None
            assert record.kind.value.typing.field == "signature"
            record.kind.value.typing.value.inputs.append(pb.Arg(sort=record.kind.value.typing.value.output))
            record.bindings.python.views[0].params.append(pb.PythonParameter(core_input=2, name="extra"))
        else:
            view = pb.PythonCallable.from_binary(record.bindings.python.views[0].to_binary())
            view.path = ["other_add"]
            record.bindings.python.views.append(view)
        ref = Builder.from_program(program).publish().ref("declarations", packed.indices[0])
        candidate = RuntimeFunction(ref, 0, cast("runtime.RuntimeClass", i64))
        assert candidate != original
        if change == "view":
            assert candidate != RuntimeFunction(ref, 1, cast("runtime.RuntimeClass", i64))


def test_function_value_decoding_never_converts_canonical_captures(monkeypatch, class_authoring_without_engine):
    from egglog import conversion  # noqa: PLC0415

    monkeypatch.setattr(conversion, "CONVERSIONS", conversion.CONVERSIONS.copy())
    calls = []

    def convert_string(value):
        calls.append(value)
        return i64(10)

    converter(String, i64, convert_string, cost=0)
    authored = runtime._partial_function(cast("RuntimeFunction", i64.__add__), (String("authoring conversion"),), None)
    assert authored[1] == (to_runtime_expr(i64(10)),)
    assert len(calls) == 1
    calls.clear()

    builder = Builder()
    captured = builder.import_ref(to_runtime_expr(String("malformed decoded capture")).__egg_ref__)
    builder.import_ref(resolve_callable(i64.__add__))
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(authored[0]),
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["partial_call"], pb.Call](
                        "partial_call", pb.Call(func="egglog.core.i64.add", args=[captured])
                    )
                ),
            ),
        ),
    )
    malformed = RuntimeExpr(builder.publish().ref("nodes", index))
    with pytest.raises(TypeConstraintError, match="Expected i64, got String"):
        get_literal_value(malformed)
    with pytest.raises(TypeConstraintError, match="Expected i64, got String"):
        _ = malformed.value
    with pytest.raises(TypeConstraintError, match="Expected i64, got String"):
        str(malformed)
    assert calls == []


def test_named_function_expected_sorts_specialize_generic_builtins(class_authoring_without_engine):
    getter = UnstableFn[i64, Vec[i64], i64](Vec.__getitem__)
    assert getter.value == Vec[i64].__getitem__
    assert isinstance(getter(Vec(i64(1)), i64(0)), i64)
    constructor = UnstableFn[Vec[i64], i64, i64](Vec[i64])
    assert constructor.value == Vec[i64]
    assert isinstance(constructor(i64(1), i64(2)), Vec)
    empty = UnstableFn[Vec[i64]](Vec.empty)
    assert empty.value == Vec[i64].empty
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        UnstableFn(Vec.empty)
    wrong_result = runtime.resolve_type_annotation(UnstableFn[String, i64])
    assert isinstance(wrong_result, Ref)
    with pytest.raises(TypeConstraintError):
        runtime._partial_function(cast("RuntimeFunction", i64.__invert__), (), wrong_result)

    def stored(): ...

    stored.__annotations__["return"] = UnstableFn[i64, i64]
    output = to_runtime_expr(function(stored)())
    expected = runtime.resolve_type_annotation(UnstableFn[i64, i64])
    assert isinstance(expected, Ref)
    assert StructuralView(output.__egg_ref__.owner.ref("sorts", output.__egg_ref__.read().sort_id)) == StructuralView(
        expected
    )
    assert get_literal_value(output) is None


def test_named_function_capture_keeps_local_context_without_engine(class_authoring_without_engine):
    def make():
        class Local(Expr):
            def __init__(self) -> None: ...

            def keep(self): ...

            keep.__annotations__["return"] = "Local"

            @method(preserve=True)
            def tag(self):
                return "retained capture"

        expression = UnstableFn(Local.keep, Local())
        runtime_class = cast("runtime.RuntimeClass", Local)
        return (
            expression,
            weakref(Local),
            weakref(runtime_class.__egg_context__),
            weakref(runtime_class.__egg_definition__.owner),
        )

    expression, cls, context, owner = make()
    value = expression.value
    assert isinstance(value, partial)
    del expression
    _CLASS_CACHE.clear()
    gc.collect()
    assert cls() is not None
    assert context() is not None
    assert owner() is not None
    assert value.args[0].tag() == "retained capture"
    assert value().tag() == "retained capture"
    del value
    gc.collect()
    assert cls() is context() is owner() is None


def test_generic_empty_call_requires_result_owner_and_keeps_signature_unchanged():
    builder = Builder()
    variable = builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    pattern = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[variable])))
    )
    scalar = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
    )
    concrete = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[scalar])))
    )
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name="sample.vec.of",
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature",
                        pb.GenericSignature(type_params=["T"], output=pattern, varargs=[pb.Arg(sort=variable)]),
                    ),
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.INITIALIZER,
                            owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", pattern)),
                            params=[pb.PythonParameter(core_input=0, name="args")],
                        )
                    ]
                )
            ),
        ),
    )
    owner = builder.publish()
    ref = owner.ref("declarations", declaration)
    before = ref.read().to_binary()

    def forbidden(*args):
        pytest.fail("An empty constructor must not convert an argument")

    root, mutated = _author_call(
        ref, 0, (), {}, node_of=forbidden, convert=forbidden, owner=owner.ref("sorts", concrete)
    )
    assert mutated is None
    assert isinstance(root, Ref)
    assert list(root.read().kind.value.args) == []
    assert StructuralView(root.owner.ref("sorts", root.read().sort_id)) == StructuralView(owner.ref("sorts", concrete))
    assert ref.read().to_binary() == before
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        _author_call(ref, 0, (), {}, node_of=forbidden, convert=forbidden)


def test_grouped_tail_checks_every_element_and_complete_substitution():
    builder = Builder()
    key = builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    value = builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)))
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name="test.grouped.tail",
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature",
                        pb.GenericSignature(
                            type_params=["K", "V"],
                            inputs=[pb.Arg(sort=key, name="seed")],
                            varargs=[pb.Arg(sort=key), pb.Arg(sort=value)],
                            output=value,
                        ),
                    ),
                ),
            )
        ),
    )
    reference = builder.publish().ref("declarations", declaration)
    before = reference.read().to_binary()

    def forbidden(sort, value):
        msg = "No conversion in this structural typing witness"
        raise TypeConstraintError(msg)

    def call(*values):
        return _author_call(
            reference,
            None,
            values,
            {},
            node_of=lambda x: x.__egg_ref__ if isinstance(x, RuntimeExpr) else None,
            convert=forbidden,
        )[0]

    arguments = (i64(0), i64(1), f64(2.0), i64(3), f64(4.0))
    root = call(*arguments)
    node = root.read()
    assert StructuralView(root.owner.ref("sorts", node.sort_id)) == StructuralView(f64.__egg_sort__)  # type: ignore[attr-defined]
    assert [root.owner.ref("nodes", i) for i in node.kind.value.args] == [x.__egg_ref__ for x in arguments]
    with pytest.raises(TypeError, match="complete groups of 2"):
        call(i64(0), i64(1))
    with pytest.raises(TypeConstraintError, match="No conversion"):
        call(i64(0), i64(1), f64(2.0), f64(3.0), f64(4.0))
    with pytest.raises(TypeConstraintError, match="No conversion"):
        call(i64(0), i64(1), f64(2.0), i64(3), i64(4))
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        call(i64(0))
    assert reference.read().to_binary() == before
    source = pack([root]).program
    primitive = next(
        _required_oneof(d.kind, "host_primitive", pb.HostPrimitive)
        for d in source.declarations
        if (d.kind.field == "host_primitive" if d.kind is not None else pytest.fail("Missing declaration kind"))
    )
    assert isinstance(primitive, pb.HostPrimitive)
    signature = _required_oneof(primitive.typing, "signature", pb.GenericSignature)
    assert len(signature.varargs) == 2
    signature.type_params.append("Unused")
    adopted = Builder.from_program(source).publish()
    incomplete = next(
        adopted.ref("declarations", index)
        for index, declaration in enumerate(source.declarations)
        if (
            declaration.kind.field == "host_primitive"
            if declaration.kind is not None
            else pytest.fail("Missing declaration kind")
        )
    )
    with pytest.raises(TypeConstraintError, match="Not every generic parameter"):
        _author_call(
            incomplete,
            None,
            arguments,
            {},
            node_of=lambda x: x.__egg_ref__ if isinstance(x, RuntimeExpr) else None,
            convert=forbidden,
        )


def test_multiset_local_protocols_preserve_symbolic_occurrences_without_engine(monkeypatch):
    from egglog import MultiSet, MultiSetLike  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("MultiSet authoring or inspection accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    query = var("element", i64)
    expression = MultiSet(query, i64(1), query)
    assert expression.value == list(expression) == [query, i64(1), query]
    assert len(expression) == 3
    assert query in expression
    assert expression.value[0].__egg_ref__ == expression.value[2].__egg_ref__
    assert expression.value[0].__egg_ref__.read().kind.field == "var"
    assert get_literal_value(expression) is None
    with pytest.warns(DeprecationWarning, match="use .value"):
        assert expression.eval() == expression.value
    match expression:
        case MultiSet(children):
            assert children == expression.value
        case _:
            pytest.fail("MultiSet value matching failed")
    packed = pack([expression.__egg_ref__])
    args = _required_oneof(packed.program.nodes[packed.indices[0]].kind, "call", pb.Call).args
    assert args[0] == args[2] != args[1]
    assert packed.program.nodes[args[0]].kind is not None
    _required_oneof(packed.program.nodes[args[0]].kind, "var", str)
    assert convert((), MultiSet[i64]).__egg_ref__.read().kind.value.func == "egglog.core.multiset.of"

    @function
    def size(value: MultiSetLike[i64, int]) -> i64: ...

    assert _required_callable_args(size((1, 2, 1)))[0].value == [i64(1), i64(2), i64(1)]
    assert _required_callable_args(size(()))[0].value == []
    for source in ([1], {1}, ("wrong",)):
        with pytest.raises(ConvertError):
            convert(source, MultiSet[i64])
    with pytest.raises(ExprValueError):
        _ = expression.insert(i64(3)).value
    with pytest.raises(TypeConstraintError):
        MultiSet()


@pytest.mark.parametrize("empty", [False, True])
def test_multiset_formed_values_keep_refs_and_inert_pretty(empty, monkeypatch):
    from egglog import MultiSet  # noqa: PLC0415

    builder = Builder()
    first = builder.import_ref(i64(2).__egg_ref__)
    second = builder.import_ref(i64(1).__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(MultiSet[i64].__egg_sort__),  # type: ignore[attr-defined]
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["multiset"], pb.ValueList](
                        "multiset", pb.ValueList(items=[] if empty else [first, second, first])
                    )
                ),
            ),
        ),
    )
    owner = builder.publish()
    expression = RuntimeExpr(owner.ref("nodes", root))
    children = [] if empty else [owner.ref("nodes", index) for index in (first, second, first)]
    before = expression.__egg_ref__.read().to_binary()

    def forbidden(*args, **kwargs):
        pytest.fail("MultiSet formed-value inspection authored or executed a Call")

    monkeypatch.setattr(runtime, "_author_call", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    del builder, owner
    gc.collect()
    values = expression.value
    assert isinstance(values, list)
    references = []
    for child in values:
        assert isinstance(child, RuntimeExpr)
        references.append(child.__egg_ref__)
    assert references == children
    assert list(expression) == expression.value  # type: ignore[call-overload]
    assert str(expression) == ("MultiSet[i64]()" if empty else "MultiSet[i64](i64(2), i64(1), i64(2))")
    assert get_callable_args(expression) is None
    assert get_literal_value(expression) is None
    assert hash(copy(expression)) == hash(expression)
    packed = pack([expression.__egg_ref__])
    adopted = Builder.from_program(pb.Program.from_binary(packed.program.to_binary())).publish()
    relocated = RuntimeExpr(adopted.ref("nodes", packed.indices[0]))
    assert relocated.value == expression.value
    assert str(relocated) == str(expression)
    assert expression.__egg_ref__.read().to_binary() == before


@pytest.mark.parametrize("same", [False, True])
def test_multiset_occurrences_preserve_union_identity(same):
    from egglog import MultiSet  # noqa: PLC0415

    first = RuntimeExpr(eq(i64(1)).to(i64(2)).__egg_ref__)
    second = first if same else RuntimeExpr(eq(i64(1)).to(i64(2)).__egg_ref__)
    expression = MultiSet(first, second)
    assert len(expression) == 2
    assert [child.__egg_ref__ for child in expression.value] == [first.__egg_ref__, second.__egg_ref__]
    packed = pack([expression.__egg_ref__])
    args = _required_oneof(packed.program.nodes[packed.indices[0]].kind, "call", pb.Call).args
    assert (args[0] == args[1]) == same


@pytest.mark.parametrize("malformed", ["wrong_sort", "invalid_ref", "cycle", "arity", "wrong_arm", "user_call"])
def test_multiset_formed_values_reject_malformed_children(malformed):
    from egglog import MultiSet  # noqa: PLC0415

    builder = Builder()
    child = builder.import_ref((f64(1.0) if malformed == "wrong_sort" else i64(1)).__egg_ref__)
    sort_ref = MultiSet[i64].__egg_sort__  # type: ignore[attr-defined]
    if malformed == "arity":
        record = sort_ref.read()
        record.kind.value.args.clear()
        sort = builder.add("sorts", record)
    else:
        sort = builder.import_ref(sort_ref)
    if malformed == "invalid_ref":
        child = 999
    elif malformed == "cycle":
        child = 1
    arm = "vec" if malformed == "wrong_arm" else "multiset"
    kind = Oneof[Literal["primitive_value"], pb.PrimitiveValue](
        "primitive_value",
        pb.PrimitiveValue(value=Oneof(arm, pb.ValueList(items=[child]))),  # type: ignore[arg-type]
    )
    if malformed == "user_call":
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function", pb.Function(name="egglog.core.multiset.of", output=sort)
                )
            ),
        )
        kind = Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.multiset.of", args=[child]))
    root = builder.add("nodes", pb.Node(sort_id=sort, kind=kind))
    with pytest.raises((ValueError, IndexError, AttributeError)):
        _ = RuntimeExpr(builder.publish().ref("nodes", root)).value


def test_multiset_formed_symbolic_children_outlive_parent_without_authoring(monkeypatch):
    from egglog import MultiSet  # noqa: PLC0415

    query = var("retained_multiset_element", i64)
    symbolic = query + i64(1)
    before = symbolic.__egg_ref__.read().to_binary()
    expected_text, expected_hash = str(symbolic), hash(symbolic)
    builder = Builder()
    child = builder.import_ref(symbolic.__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(MultiSet[i64].__egg_sort__),  # type: ignore[attr-defined]
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(
                    value=Oneof[Literal["multiset"], pb.ValueList]("multiset", pb.ValueList(items=[child, child]))
                ),
            ),
        ),
    )
    owner = builder.publish()
    expression = RuntimeExpr(owner.ref("nodes", root))
    parent_owner, child_owner = weakref(owner), weakref(symbolic.__egg_ref__.owner)

    def forbidden(*args, **kwargs):
        pytest.fail("Retained MultiSet child inspection authored or executed a Call")

    monkeypatch.setattr(runtime, "_author_call", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    retained = expression.value
    del expression, builder, owner, symbolic, query
    gc.collect()
    assert parent_owner() is None
    assert child_owner() is not None
    assert isinstance(retained, list)
    assert len(retained) == 2
    first_child, second_child = retained
    assert isinstance(first_child, RuntimeExpr)
    assert isinstance(second_child, RuntimeExpr)
    assert first_child.__egg_ref__ == second_child.__egg_ref__
    assert first_child.__egg_ref__.read().kind.field == "call"
    assert first_child.__egg_ref__.read().to_binary() == before
    assert str(first_child) == str(second_child) == expected_text
    assert hash(first_child) == hash(second_child) == expected_hash
    assert _required_callable_args(first_child)[0].__egg_ref__.read().kind.field == "var"
