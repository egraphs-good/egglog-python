# mypy: allow-empty-bodies
# Ellipsis bodies are DSL declarations or annotation-introspection fixtures.
"""Ordinary Maybe authoring/inspection and real-byte execution."""

import copy
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import (
    EGraph,
    Expr,
    ExprValueError,
    Map,
    Maybe,
    String,
    Vec,
    bindings,
    conversion,
    eq,
    f64,
    function,
    i64,
    union,
)
from egglog._catalog import builtin_catalog
from egglog._program import Builder, StructuralView, _copy_record, pack
from egglog.deconstruct import get_callable_args, get_callable_fn
from egglog.egraph_state import EggSmolError
from egglog.runtime import RuntimeExpr, RuntimeFunction
from egglog.type_constraint_solver import TypeConstraintError


def test_maybe_local_value_inspection_is_inert(monkeypatch):
    none = Maybe[i64].none()
    symbolic = i64(1) / i64(0)
    some = Maybe[i64].some(symbolic)
    nested = Maybe[Maybe[i64]].some(none)
    indirect = Vec[Maybe[i64]](some)[i64(0)]

    def unexpected(*args, **kwargs):
        msg = "inspection evaluated or manufactured a call"
        raise AssertionError(msg)

    monkeypatch.setattr(EGraph, "__init__", unexpected)
    monkeypatch.setattr(RuntimeFunction, "__call__", unexpected)
    assert none.value is None
    some_child = some.value
    assert some_child is not None
    assert StructuralView(some_child.__egg_ref__) == StructuralView(symbolic.__egg_ref__)
    assert nested.value is not None
    assert nested.value.value is None
    with pytest.raises(ExprValueError):
        indirect.value
    match nested:
        case Maybe(child):
            assert child is not None
            assert child.value is None
        case _:
            pytest.fail("Maybe value pattern did not match")


def test_maybe_ordinary_surface_executes_as_bytes():
    graph = EGraph()
    assert graph.extract(Maybe[f64].none()).value is None
    some = graph.extract(Maybe[f64].some(1.0))  # type: ignore[arg-type]
    assert some.value is not None
    assert some.value.value == 1.0
    assert graph.extract(Maybe[f64].some(1.0).unwrap()).value == 1.0  # type: ignore[arg-type]
    assert graph.extract(Maybe[f64].none().unwrap_or(2.5)).value == 2.5  # type: ignore[arg-type]
    assert graph.extract(Maybe[i64].some(i64(7)).unwrap_or(i64(2))).value == 7
    with pytest.raises(EggSmolError, match=r"evaluation|Evaluation|failed"):
        graph.extract(Maybe[i64].none().unwrap())
    with pytest.raises(EggSmolError, match=r"evaluation|Evaluation|failed"):
        graph.extract(Maybe[i64].some(i64(7)).unwrap_or(i64(1) / i64(0)))
    assert str(some) == "Maybe[f64].some(f64(1.0))"
    roundtripped_child = EGraph().extract(some).value
    assert roundtripped_child is not None
    assert roundtripped_child.value == 1.0


def test_higher_order_map_fold_and_maybe_callbacks_execute_as_bytes(monkeypatch):
    from egglog import Pair, catch, map_fold_kv  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Higher-order authoring accessed an engine")

    mapping = Map[i64, f64].empty().insert(i64(1), f64(2.5)).insert(i64(3), f64(4.5))
    offset = i64(10)
    initial = Pair(i64(0), f64(0.0))
    with monkeypatch.context() as guard:
        guard.setattr(bindings, "_ProtoEngine", forbidden)
        expression = map_fold_kv(
            lambda total, key, item: Pair(
                total.left + key + offset,
                total.right + catch(lambda: mapping[key]).match(lambda found: found + item, f64(0.0)),
            ),
            initial,
            mapping,
        )
        assert get_callable_fn(expression) == map_fold_kv
        args = get_callable_args(expression)
        assert args is not None
        # Deconstruction returns runtime presentations of the authored container classes.
        assert args[1:] == (initial, mapping)  # type: ignore[comparison-overlap]
        packed = pack([expression.__egg_ref__])
        detached = RuntimeExpr(
            Builder
            .from_program(pb.Program.from_binary(packed.program.to_binary()))
            .publish()
            .ref("nodes", packed.indices[0])
        )
        assert StructuralView(detached.__egg_ref__) == StructuralView(expression.__egg_ref__)

    graph = EGraph()
    assert graph.extract(expression).value == (i64(24), f64(14.0))
    assert graph.extract(catch(lambda: mapping[i64(99)])).value is None
    assert graph.extract(catch(lambda: mapping[i64(1)]).match(lambda item: item + 1.0, f64(7.0))).value == 3.5
    assert graph.extract(Maybe[i64].none().match(lambda item: item / i64(0), i64(7))).value == 7
    assert graph.extract(Maybe[i64].some(i64(3)).match(lambda item: item.to_string(), String("none"))).value == "3"


def test_maybe_none_conversion_and_constructor_deconstruction():
    @function
    def identity(value: Maybe[i64]) -> Maybe[i64]: ...

    assert StructuralView(identity(None).__egg_ref__) == StructuralView(identity(Maybe[i64].none()).__egg_ref__)  # type: ignore[arg-type]
    converted = conversion.convert(None, Maybe[i64])
    assert isinstance(converted, RuntimeExpr)
    assert converted.value is None
    for call in [Maybe[i64].none(), Maybe[i64].some(i64(4)), Maybe[Maybe[i64]].some(Maybe[i64].none())]:
        assert isinstance(call, RuntimeExpr)
        args, fn = get_callable_args(call), get_callable_fn(call)
        assert args is not None
        assert fn is not None
        assert isinstance(fn, RuntimeFunction)
        rebuilt_call = fn(*args)
        assert rebuilt_call is not None
        assert StructuralView(rebuilt_call.__egg_ref__) == StructuralView(call.__egg_ref__)
    declaration = builtin_catalog().definitions["callable", "egglog.experimental.maybe.none"]
    builder = Builder()
    record = declaration.read()
    record.bindings = None
    # Retain the signature's owned sort edges while dropping only presentation.
    builder.add(
        "declarations",
        _copy_record(record, relocate=lambda role, index: builder.import_ref(declaration.owner.ref(role, index))),
    )
    none = Maybe[i64].none().__egg_ref__
    index = builder.add(
        "nodes", _copy_record(none.read(), relocate=lambda role, index: builder.import_ref(none.owner.ref(role, index)))
    )
    rebuilt = RuntimeExpr(builder.publish().ref("nodes", index))
    assert rebuilt.value is None
    args, fn = get_callable_args(rebuilt), get_callable_fn(rebuilt)
    assert args == ()
    assert fn is not None
    assert isinstance(fn, RuntimeFunction)
    rebuilt_call = fn()
    assert rebuilt_call is not None
    assert StructuralView(rebuilt_call.__egg_ref__) == StructuralView(rebuilt.__egg_ref__)


def test_generic_classmethod_reconstruction_preserves_bound_callable_identity(monkeypatch):
    cases: list[tuple[Callable[..., object], tuple[object, ...]]] = [
        (Vec[i64].empty, ()),
        (Map[i64, String].empty, ()),
        (Maybe[i64].none, ()),
        (Maybe[String].none, ()),
        (Maybe[i64].some, (i64(4),)),
        (Maybe[Maybe[i64]].some, (Maybe[i64].none(),)),
    ]
    calls = []
    for fn, args in cases:
        call = fn(*args)
        assert isinstance(call, RuntimeExpr)
        calls.append(call)
    for (fn, _), call in zip(cases, calls, strict=True):
        packet = pack([call.__egg_ref__])
        imported = RuntimeExpr(
            Builder
            .from_program(pb.Program.from_binary(packet.program.to_binary()))
            .publish()
            .ref("nodes", packet.indices[0])
        )
        for original in [call, imported]:
            recovered = get_callable_fn(original)
            assert isinstance(recovered, RuntimeFunction)
            assert recovered == fn
            assert hash(recovered) == hash(fn)
            assert recovered.__egg_type_bindings__ == ()
            args = get_callable_args(original)
            assert args is not None
            rebuilt = recovered(*args)
            assert rebuilt is not None
            assert StructuralView(rebuilt.__egg_ref__) == StructuralView(call.__egg_ref__)
    assert get_callable_fn(calls[2]) != get_callable_fn(calls[3])

    def forbidden(*args, **kwargs):
        pytest.fail("Local reconstruction evaluated, converted, or manufactured a call")

    monkeypatch.setattr(conversion, "resolve_literal", forbidden)
    monkeypatch.setattr(RuntimeFunction, "__call__", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    for (fn, _), call in zip(cases, calls, strict=True):
        assert get_callable_fn(call) == fn


def test_maybe_formed_child_zero_nested_presence_import_and_hash():
    builder = Builder()
    i64_sort = i64(0).__egg_ref__
    child = builder.import_ref(i64_sort)
    maybe = Maybe[i64].none().__egg_ref__
    sort = builder.import_ref(maybe.owner.ref("sorts", maybe.read().sort_id))
    builder.import_ref(builtin_catalog().definitions["host_sort_family", "Maybe"])
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=sort,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value",
                pb.PrimitiveValue(value=Oneof[Literal["maybe"], pb.MaybeValue]("maybe", pb.MaybeValue(value=child))),
            ),
        ),
    )
    assert child == 0
    formed = RuntimeExpr(builder.publish().ref("nodes", index))
    formed_value = formed.value
    assert formed_value is not None
    assert isinstance(formed_value, RuntimeExpr)
    assert formed_value.value == 0
    packed = pack([formed.__egg_ref__])
    imported = RuntimeExpr(Builder.from_program(packed.program).publish().ref("nodes", packed.indices[0]))
    assert StructuralView(imported.__egg_ref__) == StructuralView(formed.__egg_ref__)
    assert hash(StructuralView(imported.__egg_ref__)) == hash(StructuralView(formed.__egg_ref__))
    nested = EGraph().extract(Maybe[Maybe[i64]].some(Maybe[i64].none()))
    assert nested.value is not None
    assert nested.value.value is None
    assert StructuralView(nested.__egg_ref__) != StructuralView(EGraph().extract(Maybe[Maybe[i64]].none()).__egg_ref__)


def test_maybe_nested_equality_rebuilds():
    class Atom(Expr):
        def __init__(self, value: i64) -> None: ...

    a, b = Atom(i64(1)), Atom(i64(2))
    left = Maybe[Maybe[Atom]].some(Maybe[Atom].some(a))
    right = Maybe[Maybe[Atom]].some(Maybe[Atom].some(b))
    graph = EGraph()
    graph.register(left, right, union(a).with_(b))
    graph.run(1)
    graph.check(eq(left).to(right))
    formed = graph.extract(left)
    assert formed.value is not None
    assert formed.value.value is not None
    EGraph().extract(formed)


def test_maybe_stub_is_derived_from_actual_member_metadata():
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    program = pb.Program.from_binary((Path(__file__).resolve().parents[1] / "egglog/_builtin_catalog.pb").read_bytes())
    before = program.to_binary()
    generated = render(program)
    assert program.to_binary() == before
    assert "def value(self) -> _Maybe_T_0 | None:" in generated
    assert "def none(cls) -> Maybe[_Maybe_T_0]:" in generated
    assert "def some(cls, value: _Maybe_T_0) -> Maybe[_Maybe_T_0]:" in generated
    assert "def unwrap(self) -> _Maybe_T_0:" in generated
    assert "def unwrap_or(self, default: _Maybe_T_0) -> _Maybe_T_0:" in generated
    declarations: dict[str, pb.Declaration] = {}
    for declaration in program.declarations:
        declaration_kind = declaration.kind
        assert declaration_kind is not None
        declarations[declaration_kind.value.name] = declaration
    family_kind = declarations["Maybe"].kind
    assert family_kind is not None
    assert family_kind.field == "host_sort_family"
    family = family_kind.value
    assert family.bindings is not None
    assert family.bindings.python is not None
    family.bindings.python.path[-1] = "OptionalValue"
    family.bindings.python.type_params = ["Element"]
    some_bindings = declarations["egglog.experimental.maybe.some"].bindings
    assert some_bindings is not None
    assert some_bindings.python is not None
    some_bindings.python.views[0].path = ["present"]
    some_bindings.python.views[0].params[0].name = "child"
    renamed = render(program)
    assert "def value(self) -> _OptionalValue_Element_0 | None:" in renamed
    assert "def present(cls, child: _OptionalValue_Element_0) -> OptionalValue[_OptionalValue_Element_0]:" in renamed
    owner = Builder.from_program(program).publish()
    relocated = pack([owner.ref("declarations", i) for i in reversed(range(len(program.declarations)))]).program
    reordered = render(relocated)
    for line in renamed.splitlines():
        if "_OptionalValue_" in line or "class OptionalValue(" in line:
            assert line in reordered


def test_maybe_formed_pretty_uses_owned_renamed_constructor_metadata():
    formed = EGraph().extract(Maybe[i64].some(i64(0)))
    packet = pack([
        formed.__egg_ref__,
        builtin_catalog().definitions["callable", "egglog.experimental.maybe.some"],
    ])
    program = packet.program
    for declaration in program.declarations:
        kind = declaration.kind
        assert kind is not None
        if kind.field == "host_sort_family" and kind.value.name == "Maybe":
            assert kind.value.bindings is not None
            assert kind.value.bindings.python is not None
            kind.value.bindings.python.path[-1] = "OptionalValue"
        elif kind.field == "host_primitive" and kind.value.name == "egglog.experimental.maybe.some":
            assert declaration.bindings is not None
            assert declaration.bindings.python is not None
            declaration.bindings.python.views[0].path = ["present"]
    value = RuntimeExpr(Builder.from_program(program).publish().ref("nodes", packet.indices[0]))
    child_value = value.value
    assert child_value is not None
    assert isinstance(child_value, RuntimeExpr)
    assert child_value.value == 0
    assert "OptionalValue[i64].present(" in str(value)


def test_maybe_formed_and_call_defaults_execute_without_changing_representation():
    graph = EGraph()
    for default in [Maybe[i64].none(), Maybe[i64].some(i64(0))]:
        for value in [default, graph.extract(default)]:

            @function
            def identity(x: Maybe[i64] = value) -> Maybe[i64]:
                return x

            called = identity()
            args = get_callable_args(called)
            assert args is not None
            assert len(args) == 1
            assert StructuralView(args[0].__egg_ref__) == StructuralView(value.__egg_ref__)
            result = EGraph().extract(called)
            assert (result.value is None) == (value.value is None)
            if result.value is not None:
                assert result.value.value == 0


@pytest.mark.parametrize("mutation", ["wrong_child", "missing_child", "wrong_family", "arity", "scheme", "provenance"])
def test_maybe_inspection_rejects_malformed_local_values(mutation):
    original = Maybe[i64].some(i64(0)).__egg_ref__
    packet = pack([original])
    program = copy.deepcopy(packet.program)
    node = program.nodes[packet.indices[0]]
    node_kind = node.kind
    assert node_kind is not None
    assert node_kind.field == "call"
    sort_kind = program.sorts[node.sort_id].kind
    assert sort_kind is not None
    assert sort_kind.field == "family"
    if mutation == "wrong_child":
        program.nodes[node_kind.value.args[0]].sort_id = node.sort_id
    elif mutation == "missing_child":
        node_kind.value.args.clear()
    elif mutation == "wrong_family":
        sort_kind.value.name = "Vec"
    elif mutation == "arity":
        sort_kind.value.args.clear()
    else:
        for declaration in program.declarations:
            declaration_kind = declaration.kind
            assert declaration_kind is not None
            if (
                declaration_kind.field == "host_primitive"
                and declaration_kind.value.name == "egglog.experimental.maybe.some"
            ):
                break
        else:
            raise StopIteration
        if mutation == "scheme":
            declaration_kind = declaration.kind
            assert declaration_kind is not None
            assert declaration_kind.field == "host_primitive"
            signature_kind = declaration_kind.value.typing
            assert signature_kind is not None
            assert signature_kind.field == "signature"
            signature = signature_kind.value
            signature.output = signature.inputs[0].sort
        else:
            declaration.kind = Oneof[Literal["constructor"], pb.Constructor](
                "constructor", pb.Constructor(name="egglog.experimental.maybe.some", inputs=[], output=node.sort_id)
            )
    with pytest.raises((ValueError, TypeError, TypeConstraintError, ExprValueError)):
        _ = RuntimeExpr(Builder.from_program(program).publish().ref("nodes", packet.indices[0])).value
