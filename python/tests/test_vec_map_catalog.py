"""Ordinary container methods over the complete native-generated catalog."""

import copy
import runpy
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import (
    ConvertError,
    EGraph,
    Expr,
    ExprValueError,
    Map,
    Pair,
    String,
    UnstableFn,
    Vec,
    bindings,
    conversion,
    eq,
    f64,
    i64,
)
from egglog._catalog import builtin_catalog
from egglog._presentation import CallKind, CallPresentation, Parameter, Presentation
from egglog._program import Builder, Ref, StructuralView, pack
from egglog.deconstruct import get_callable_args, get_callable_fn, get_literal_value
from egglog.egraph_state import EggSmolError
from egglog.runtime import RuntimeExpr, RuntimeFunction, resolve_type_annotation
from egglog.type_constraint_solver import TypeConstraintError


def test_append_consuming_receiver_and_free_call_reconstruction():
    xs, ys = Vec[i64](i64(1)), Vec[i64](i64(2))
    for call, arity in [(xs.append(), 1), (xs.append(ys, xs), 3), (Vec[i64].append(xs, ys, xs), 3)]:
        assert len(call.__egg_ref__.read().kind.value.args) == arity
        args = get_callable_args(call)
        fn = get_callable_fn(call)
        assert args is not None
        assert fn is not None
        assert len(args) == arity
        assert StructuralView(fn(*args).__egg_ref__) == StructuralView(call.__egg_ref__)
        assert ".append(" in str(call)
    declaration = builtin_catalog().definitions[("callable", "egglog.core.vec.append")]
    free = RuntimeFunction(declaration)
    rebuilt = free(xs, ys)
    assert rebuilt is not None
    assert StructuralView(rebuilt.__egg_ref__) == StructuralView(xs.append(ys).__egg_ref__)
    builder = Builder()
    builder.import_ref(declaration)
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(xs.__egg_ref__.owner.ref("sorts", xs.__egg_ref__.read().sort_id)),
            kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.vec.append")),
        ),
    )
    zero = RuntimeExpr(builder.publish().ref("nodes", index))
    assert str(zero) == "egglog.core.vec.append()"
    assert get_callable_args(zero) == ()
    fn = get_callable_fn(zero)
    assert fn is not None
    assert StructuralView(fn().__egg_ref__) == StructuralView(zero.__egg_ref__)


def test_vec_union_exact_equality_substitution():
    class Atom(Expr):
        def __init__(self, value: i64) -> None: ...

    left, right = Vec[Atom](Atom(i64(1))), Vec[Atom](Atom(i64(2)))
    assert (left | right).__egg_ref__.read().kind.value.func == "egglog.core.vec.union"
    for value in [Vec[i64](i64(1)), Vec[Vec[Atom]](left)]:
        with pytest.raises(TypeConstraintError, match="requires an equality sort"):
            value | value  # type: ignore[operator]  # Deliberately invalid equality-sort operands.


def test_generator_rejects_invalid_equality_binders():
    render = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/generate_builtin_catalog.py"))[
        "render_stubs"
    ]
    original = pb.BuiltinCatalog.from_binary(
        (Path(__file__).resolve().parents[1] / "egglog/_builtin_catalog.pb").read_bytes()
    )
    for binders, restrictions in [(1, [1]), (1, [0, 0]), (2, [1, 0])]:
        invalid = copy.deepcopy(original)
        for declaration in invalid.primitives:
            if declaration.name == "egglog.core.vec.union":
                break
        else:
            raise StopIteration
        signature_kind = declaration.typing
        assert signature_kind is not None
        assert signature_kind.field == "signature"
        signature = signature_kind.value
        signature.type_params = ["T"] * binders
        signature.equality_type_params = restrictions
        with pytest.raises(ValueError, match="Invalid equality type parameter indices"):
            render(invalid)


def test_ordinary_vec_map_methods_execute_and_reconstruct():
    graph = EGraph()
    xs, ys = Vec[i64](i64(7), i64(11), i64(7)), Vec[i64](i64(3))
    for call, expected in [
        (xs.append(), (7, 11, 7)),
        (xs.append(ys, [i64(9)]), (7, 11, 7, 3, 9)),
        (xs.push(i64(3)), (7, 11, 7, 3)),
        (xs.pop(), (7, 11)),
        (Vec[i64].empty().pop(), ()),
        (xs.set(1, i64(9)), (7, 9, 7)),
        (xs.remove(1), (7, 7)),
    ]:
        fn, args = get_callable_fn(call), get_callable_args(call)
        assert fn is not None
        assert args is not None
        assert StructuralView(fn(*args).__egg_ref__) == StructuralView(call.__egg_ref__)
        with pytest.raises(ExprValueError):
            _ = call.value
        assert tuple(value.value for value in graph.extract(call).value) == expected
    assert graph.extract(xs.length()).value == 3
    assert graph.check_bool(xs.contains(i64(11)))
    assert not graph.check_bool(xs.contains(i64(42)))
    assert graph.check_bool(xs.not_contains(i64(42)))
    assert not graph.check_bool(xs.not_contains(i64(11)))
    for index in [-1, 3]:
        for call in [xs.set(index, i64(9)), xs.remove(index)]:
            with pytest.raises(EggSmolError, match=r"evaluation|Evaluation|failed"):
                graph.extract(call)
    declaration = builtin_catalog().definitions["callable", "egglog.core.vec.append"]
    zero = _stored_call(declaration, [], Vec[i64])
    assert graph.extract(zero).value == ()
    nested = Vec[Vec[i64]](xs)
    assert len(graph.extract(nested.append(nested)).value) == 2
    mapping = Map[i64, i64].empty().insert(i64(1), i64(7)).insert(i64(2), i64(9))
    assert graph.extract(mapping.length()).value == 2
    removed = mapping.remove(i64(1))
    fn, args = get_callable_fn(removed), get_callable_args(removed)
    assert fn is not None
    assert args is not None
    assert StructuralView(fn(*args).__egg_ref__) == StructuralView(removed.__egg_ref__)
    assert {k.value: v.value for k, v in graph.extract(removed).value.items()} == {2: 9}
    graph.check(eq(mapping.remove(i64(42))).to(mapping))
    assert graph.check_bool(mapping.contains(i64(1)))
    assert not graph.check_bool(mapping.contains(i64(42)))
    assert graph.check_bool(mapping.not_contains(i64(42)))
    assert not graph.check_bool(mapping.not_contains(i64(1)))


def test_vec_union_merges_elements_and_keeps_failure_atomic():
    class Atom(Expr):
        def __init__(self, value: i64) -> None: ...

    graph = EGraph()
    a, b, c = Atom(i64(1)), Atom(i64(2)), Atom(i64(3))
    left, right = Vec[Atom](a), Vec[Atom](b)
    graph.register(left | right)
    graph.check(eq(a).to(b))
    graph.check(eq(graph.extract(left | right)).to(left))
    with pytest.raises(EggSmolError, match=r"evaluation|Evaluation|failed"):
        graph.register(left | Vec[Atom](c, c))
    assert not graph.check_bool(eq(a).to(c))
    with pytest.raises(EggSmolError):
        graph.check(eq(left | right).to(left))


def _stored_call(declaration, args, result_type):
    """Import actual canonical references into an unevaluated call fixture."""
    builder = Builder()
    builder.import_ref(declaration)
    record = declaration.read()
    assert isinstance(record, pb.Primitive)
    result = resolve_type_annotation(result_type)
    assert isinstance(result, Ref)
    index = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(result),
            kind=Oneof[Literal["call"], pb.Call](
                "call",
                pb.Call(func=record.name, args=[builder.import_ref(arg.__egg_ref__) for arg in args]),
            ),
        ),
    )
    return RuntimeExpr(builder.publish().ref("nodes", index))


def test_zero_append_specialization_is_stable_and_structural(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local reconstruction accessed an evaluator or converter")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    monkeypatch.setattr(conversion, "resolve_literal", forbidden)
    declaration = builtin_catalog().definitions["callable", "egglog.core.vec.append"]
    recovered = []
    for typ in [i64, String]:
        # This fixture specializes real runtime sort presentations, not a static type alias.
        original = _stored_call(declaration, [], Vec[typ])  # type: ignore[valid-type]
        fn = get_callable_fn(original)
        assert fn is not None
        runtime_fn = fn
        assert isinstance(runtime_fn, RuntimeFunction)
        assert fn != RuntimeFunction(declaration)
        assert [index for index, _ in runtime_fn.__egg_type_bindings__] == [0]
        assert StructuralView(fn().__egg_ref__) == StructuralView(original.__egg_ref__)
        packet = pack([original.__egg_ref__])
        imported = RuntimeExpr(
            Builder
            .from_program(
                pb.Program.from_binary(packet.program.to_binary()), environment=original.__egg_ref__.owner._environment
            )
            .publish()
            .ref("nodes", packet.indices[0])
        )
        other = get_callable_fn(imported)
        assert other is not None
        assert other == fn
        assert hash(other) == hash(fn)
        assert {fn: True}[other]
        assert get_callable_args(original, other) == ()
        assert get_callable_args(original, RuntimeFunction(declaration)) == ()
        recovered.append(fn)
    assert recovered[0] != recovered[1]


def test_zero_append_conflicts_and_named_partial_roundtrip():
    declaration = builtin_catalog().definitions["callable", "egglog.core.vec.append"]
    fn = get_callable_fn(_stored_call(declaration, [], Vec[i64]))
    assert fn is not None
    runtime_fn = fn
    assert isinstance(runtime_fn, RuntimeFunction)
    before = runtime_fn.__egg_type_bindings__
    with pytest.raises((TypeConstraintError, TypeError, ConvertError)):
        fn(Vec[f64](f64(1.0)))
    assert runtime_fn.__egg_type_bindings__ == before
    # RuntimeClass supports parameterized checks; MyPy rejects their Python syntax intrinsically.
    assert isinstance(fn(), Vec[i64])  # type: ignore[misc]
    with pytest.raises(TypeConstraintError):
        UnstableFn[Vec[String]](fn)
    for named in [UnstableFn(fn), UnstableFn[Vec[i64]](fn)]:
        packet = pack([named.__egg_ref__])
        decoded = RuntimeExpr(
            Builder
            .from_program(
                pb.Program.from_binary(packet.program.to_binary()), environment=named.__egg_ref__.owner._environment
            )
            .publish()
            .ref("nodes", packet.indices[0])
        )
        restored = get_literal_value(decoded)
        assert isinstance(restored, RuntimeFunction)
        assert restored == fn
        assert hash(restored) == hash(fn)
        restored_call, original_call = restored(), fn()
        assert restored_call is not None
        assert original_call is not None
        assert StructuralView(restored_call.__egg_ref__) == StructuralView(original_call.__egg_ref__)
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        RuntimeFunction(declaration)()


@pytest.mark.parametrize("selected_view", [False, True])
def test_minimal_result_binding_preserves_input_polymorphism(selected_view, native_catalog_program):
    # Inference-only fixture: no provider/body/engine is installed or invoked.
    program = native_catalog_program
    for declaration in program.primitives:
        if declaration.name == "egglog.core.pair.make":
            break
    else:
        raise StopIteration
    declaration.name = "test.result_pair"
    signature_kind = declaration.typing
    assert signature_kind is not None
    assert signature_kind.field == "signature"
    signature_kind.value.inputs.pop()
    builder = Builder.from_builtins(program)
    if selected_view:
        builder.presentations["callable", declaration.name] = Presentation(
            views=(
                CallPresentation(
                    kind=CallKind.FUNCTION,
                    path=("result_pair",),
                    params=(Parameter(core_input=0, name="value"),),
                ),
            )
        )
    owner = builder.publish()
    reference = owner.ref("primitives", program.primitives.index(declaration))
    first = get_callable_fn(_stored_call(reference, [i64(1)], Pair[i64, String]))
    second = get_callable_fn(_stored_call(reference, [f64(1.0)], Pair[f64, String]))
    assert first is not None
    runtime_first = first
    assert isinstance(runtime_first, RuntimeFunction)
    assert [index for index, _ in runtime_first.__egg_type_bindings__] == [1]
    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(first(f64(2.0)), Pair[f64, String])  # type: ignore[misc]  # RuntimeClass parameterized check.
    different = get_callable_fn(_stored_call(reference, [i64(1)], Pair[i64, i64]))
    assert different != first
    with pytest.raises(TypeConstraintError):
        get_callable_fn(_stored_call(reference, [i64(1)], Pair[f64, String]))
    # A concrete named Func intentionally also fixes its remaining input type.
    concrete = UnstableFn[Pair[i64, String], i64](first)
    decoded = get_literal_value(concrete)
    assert isinstance(decoded, RuntimeFunction)
    assert [index for index, _ in decoded.__egg_type_bindings__] == [0, 1]
    assert isinstance(decoded(i64(3)), Pair[i64, String])  # type: ignore[misc]  # RuntimeClass parameterized check.
    with pytest.raises((TypeConstraintError, TypeError, ConvertError)):
        decoded(f64(3.0))


def test_method_result_binding_preserves_independent_input_polymorphism(monkeypatch, native_catalog_program):
    # Inference-only Vec<T> receiver, V input, Pair<V,U> result. No provider.
    program = native_catalog_program
    for declaration in program.primitives:
        if declaration.name == "egglog.core.pair.make":
            break
    else:
        raise StopIteration
    declaration.name = "test.method_result_pair"
    signature_kind = declaration.typing
    assert signature_kind is not None
    assert signature_kind.field == "signature"
    signature = signature_kind.value
    first_var, second_var = [arg.sort for arg in signature.inputs]
    third_var = len(program.sorts)
    program.sorts.append(pb.Sort(kind=Oneof[Literal["var"], int]("var", 2)))
    receiver_sort = len(program.sorts)
    program.sorts.append(
        pb.Sort(
            kind=Oneof[Literal["application"], pb.SortApplication](
                "application", pb.SortApplication(name="Vec", args=[first_var])
            )
        )
    )
    signature.type_params.append("V")
    signature.inputs = [pb.Arg(sort=receiver_sort), pb.Arg(sort=third_var)]
    signature.output = len(program.sorts)
    program.sorts.append(
        pb.Sort(
            kind=Oneof[Literal["application"], pb.SortApplication](
                "application", pb.SortApplication(name="Pair", args=[third_var, second_var])
            )
        )
    )
    builder = Builder.from_builtins(program)
    view = CallPresentation(
        kind=CallKind.METHOD,
        path=("result_pair",),
        receiver=0,
        params=(Parameter(core_input=1, name="value"),),
    )
    owner = builder.publish(
        presentation_factory=lambda owned: {
            ("callable", declaration.name): Presentation(
                views=(replace(view, owner=owned.ref("sorts", receiver_sort)),)
            )
        }
    )
    reference = owner.ref("primitives", program.primitives.index(declaration))
    receiver = Vec[i64](i64(0))
    original = _stored_call(reference, [receiver, i64(1)], Pair[i64, String])
    first = get_callable_fn(original)
    second = get_callable_fn(_stored_call(reference, [receiver, f64(1.0)], Pair[f64, String]))
    assert first is not None
    runtime_first = first
    assert isinstance(runtime_first, RuntimeFunction)
    assert [index for index, _ in runtime_first.__egg_type_bindings__] == [1]
    assert first == second
    assert hash(first) == hash(second)
    args = get_callable_args(original)
    assert args is not None
    assert StructuralView(first(*args).__egg_ref__) == StructuralView(original.__egg_ref__)
    assert isinstance(first(receiver, f64(2.0)), Pair[f64, String])  # type: ignore[misc]  # RuntimeClass parameterized check.
    assert get_callable_fn(_stored_call(reference, [receiver, i64(1)], Pair[i64, i64])) != first
    with pytest.raises((TypeConstraintError, TypeError, ConvertError)):
        first(Vec[f64](f64(0.0)), i64(1))
    malformed = [
        _stored_call(reference, [receiver, i64(1)], Pair[f64, String]),
        _stored_call(reference, [receiver], Pair[i64, String]),
    ]
    # The owner must not reinterpret U as the receiver's already-inferred T.
    bad_owner = len(program.sorts)
    program.sorts.append(
        pb.Sort(
            kind=Oneof[Literal["application"], pb.SortApplication](
                "application", pb.SortApplication(name="Vec", args=[second_var])
            )
        )
    )
    builder = Builder.from_builtins(program)
    owner = builder.publish(
        presentation_factory=lambda owned: {
            ("callable", declaration.name): Presentation(views=(replace(view, owner=owned.ref("sorts", bad_owner)),))
        }
    )
    bad_reference = owner.ref("primitives", program.primitives.index(declaration))
    malformed.append(_stored_call(bad_reference, [receiver, i64(1)], Pair[i64, String]))

    def forbidden(*args, **kwargs):
        pytest.fail("Local reconstruction evaluated, converted, or manufactured a call")

    monkeypatch.setattr(conversion, "resolve_literal", forbidden)
    monkeypatch.setattr(RuntimeFunction, "__call__", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    assert get_callable_fn(original) == first
    for call in malformed:
        with pytest.raises(TypeConstraintError):
            get_callable_fn(call)


def test_malformed_stored_calls_and_type_bindings_fail_without_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Inspection attempted conversion")

    declaration = builtin_catalog().definitions["callable", "egglog.core.vec.append"]
    wrong_input = _stored_call(declaration, [i64(1)], Vec[i64])
    wrong_result = _stored_call(declaration, [Vec[i64](i64(1))], Vec[String])
    monkeypatch.setattr(conversion, "resolve_literal", forbidden)
    for call in [wrong_input, wrong_result]:
        with pytest.raises(TypeConstraintError):
            get_callable_fn(call)
    integer = resolve_type_annotation(i64)
    assert isinstance(integer, Ref)
    open_builder = Builder()
    open_index = open_builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    open_sort = open_builder.publish().ref("sorts", open_index)
    for malformed in [
        ((0, integer), (0, integer)),
        ((1, integer),),
        ((-1, integer),),
        ((0, open_sort),),
        ((0, i64(1).__egg_ref__),),
    ]:
        with pytest.raises(TypeConstraintError):
            RuntimeFunction(declaration, type_bindings=malformed)
