"""Record-native authoring gates for the single public runtime migration."""

import gc
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from time import sleep
from typing import ClassVar, Literal, get_type_hints
from weakref import ref as weakref

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import (
    Bool,
    BoolLike,
    ConvertError,
    Expr,
    Map,
    MapLike,
    Pair,
    String,
    StringLike,
    Unit,
    Vec,
    VecLike,
    bindings,
    constant,
    convert,
    converter,
    eq,
    expr_action,
    expr_parts,
    f64,
    function,
    get_type_args,
    i64,
    join,
    method,
    runtime,
    set_,
)
from egglog._program import Builder, Ref, StructuralView, pack
from egglog.builtins import ExprValueError, _map_entries, _pair_value, _vec_value
from egglog.deconstruct import get_callable_args, get_callable_fn, get_constant_name
from egglog.runtime import _CLASS_CACHE, RuntimeExpr, RuntimeFunction, _author_call, resolve_callable
from egglog.type_constraint_solver import TypeConstraintError, infer_sort, substitute_sort


def test_scalar_wave_local_protocols_and_matching_never_evaluate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Scalar authoring accessed an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
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
        assert tuple(child.value for child in get_callable_args(expression)) == values
    assert str(String("a") + "b") == "join('a', 'b')"
    assert str(~Bool(False)) == "~Bool(False)"
    assert str(~(Bool(True) | False)) == "~(Bool(True) | False)"
    assert str((~Bool(True)) & False) == "(~Bool(True)) & False"


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_scalar_wave_canonical_bool_pretty_preserves_every_argument(count):
    canonical = RuntimeFunction(Bool.__or__.__egg_ref__, None)
    values = [index % 2 == 0 for index in range(count)]
    expression = canonical(*values)
    assert [child.value for child in get_callable_args(expression)] == values
    expected = (
        "Bool(True) | False"
        if count == 2
        else "egglog.core.bool.or(" + ", ".join(f"Bool({value})" for value in values) + ")"
    )
    assert str(expression) == expected


@pytest.mark.parametrize("target", [String("a").__add__, Bool(False).__or__])
def test_scalar_wave_binary_view_rejects_wrong_arity_and_sort(target):
    for args in ((), ("a", "b")):
        with pytest.raises(TypeError):
            target(*args)
    with pytest.raises(ConvertError):
        target(i64(1))
    with pytest.raises(TypeError, match="unexpected keyword"):
        target(other=True, name="implies")


def test_scalar_wave_specialized_view_permutation_and_zero_count():
    packet = pack([(String("left") + "right").__egg_ref__])
    declaration = next(d for d in packet.program.declarations if d.kind.value.name == "egglog.core.string.concat")
    view = declaration.bindings.python.views[1]
    view.receiver, view.params[0].core_input = 1, 0
    owner = Builder.from_program(packet.program).publish()
    target = RuntimeFunction(
        owner.ref("declarations", packet.program.declarations.index(declaration)), 1, String("right")
    )
    result = target("left")
    assert [child.value for child in get_callable_args(result)] == ["left", "right"]
    assert str(result) == "join('left', 'right')"
    view = declaration.bindings.python.views[0]
    view.tail_repetitions = 0
    view.params.clear()
    owner = Builder.from_program(packet.program).publish()
    empty = RuntimeFunction(owner.ref("declarations", packet.program.declarations.index(declaration)), 0)
    assert get_callable_args(empty()) == ()
    with pytest.raises(TypeError):
        empty("extra")


def test_scalar_wave_specialized_bool_deconstruction_uses_effective_slots():
    packet = pack([(Bool(True) | False).__egg_ref__])
    declaration = next(d for d in packet.program.declarations if d.kind.value.name == "egglog.core.bool.or")
    view = declaration.bindings.python.views[0]
    view.receiver, view.params[0].core_input = 1, 0
    owner = Builder.from_program(packet.program).publish()
    method = RuntimeFunction(owner.ref("declarations", packet.program.declarations.index(declaration)), 0, Bool(False))
    result = method(True)
    call = result.__egg_ref__.read().kind.value
    assert [RuntimeExpr(result.__egg_ref__.owner.ref("nodes", i)).value for i in call.args] == [True, False]
    assert [argument.value for argument in get_callable_args(result)] == [False, True]
    assert str(result) == "Bool(False) | True"
    assert expr_parts(get_callable_fn(result)(*get_callable_args(result))) == expr_parts(result)


@pytest.mark.parametrize(
    "mutation", ["fixed", "missing", "missing_input", "duplicate", "default", "mutation", "huge", "owner", "property"]
)
def test_scalar_wave_authoring_rejects_invalid_specialization(mutation):
    packet = pack([(Bool(True) | False).__egg_ref__])
    declaration = next(d for d in packet.program.declarations if d.kind.value.name == "egglog.core.bool.or")
    signature, view = declaration.kind.value.typing.value, declaration.bindings.python.views[0]
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
        view.owner.kind = Oneof[Literal["sort"], int]("sort", len(packet.program.sorts) - 1)
    elif mutation == "property":
        view.kind = pb.PythonCallKind.PROPERTY
    else:
        view.tail_repetitions = 2**32 - 1
    owner = Builder.from_program(packet.program).publish()
    method = RuntimeFunction(owner.ref("declarations", packet.program.declarations.index(declaration)), 0, Bool(True))
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
    first_args, second_args = get_callable_args(first), get_callable_args(second)
    assert first_args[0].__egg_ref__ != second_args[0].__egg_ref__
    assert expr_parts(first_args[0]) == expr_parts(default)
    assert first_args[1].value is False
    assert get_callable_args(flag())[0].value is False
    assert [part.value for part in get_callable_args(message("x", True))] == ["x", True]
    pair = Pair[String, Bool]("key", False)
    mapping = Map[String, Bool].empty().insert("key", False)
    assert tuple(part.value for part in pair.value) == ("key", False)
    assert mapping.value[String("key")].value is False
    assert isinstance(mapping[String("key")], Bool)
    with pytest.raises(ConvertError):
        Pair[String, Bool](False, "key")
    with pytest.raises(ConvertError):
        message(i64(1))


def test_map_authoring_symbolic_children_and_local_protocols(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Map authoring/inspection accessed an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
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
    assert isinstance(expression, Map[i64, f64])
    assert get_callable_args(expression)[0].__egg_ref__ == entries.__egg_ref__
    assert expression.value[i64(1)].__egg_ref__ == newest.__egg_ref__
    assert Map[i64, f64].from_pairs(Vec[Pair[i64, f64]].empty()).value == {}
    with pytest.raises(TypeError):
        Map[i64, f64].from_pairs()
    with pytest.raises(TypeError):
        Map[i64, f64].from_pairs(items=entries)
    with pytest.raises(ConvertError):
        Map[i64, f64].from_pairs(Pair(i64(1), f64(2.0)))
    with pytest.raises(ConvertError):
        Map[i64, f64].from_pairs(Vec(Pair(f64(1.0), i64(2))))

    @function
    def computed() -> Vec[Pair[i64, f64]]: ...

    symbolic = computed()
    mapping = Map.from_pairs(symbolic)
    assert get_callable_args(mapping)[0].__egg_ref__ == symbolic.__egg_ref__
    assert len(mapping.__egg_ref__.read().kind.value.args) == 1
    assert "computed" in str(mapping)
    with pytest.raises(ExprValueError):
        _ = mapping.value
    assert isinstance(mapping[i64(1)], f64)
    assert "Map" in {d.kind.value.name for d in pack([mapping.__egg_ref__]).program.declarations}


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
            sort_id=builder.import_ref(Map[i64, f64].__egg_sort__),
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
        assert next(iter(expression.value)).__egg_ref__ == owner.ref("nodes", key)
        assert expression.value[i64(2)].__egg_ref__ == owner.ref("nodes", later)
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

    converted = get_callable_args(size({1: 2.0}))[0]
    assert isinstance(converted, Map[i64, f64])
    assert converted.value[i64(1)].value == 2.0
    with pytest.raises(TypeError):
        _ = MapLike[i64, f64, int]
    with pytest.raises(ConvertError):
        size({1: i64(2)})
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
            _map_entries(NominalMap())
        with pytest.raises(ConvertError):
            convert({}, NominalMap)
        return
    builder = Builder()
    key = builder.import_ref((f64(1.0) if invalid == "key_sort" else i64(1)).__egg_ref__)
    value = builder.import_ref((i64(2) if invalid == "value_sort" else f64(2.0)).__egg_ref__)
    sort = builder.import_ref((Vec[i64] if invalid == "wrong_family" else Map[i64, f64]).__egg_sort__)
    if invalid == "arity":
        record = Map[i64, f64].__egg_sort__.read()
        record.kind.value.args = [builder.import_ref(i64.__egg_sort__)]
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
                sort_id=builder.import_ref(f64.__egg_sort__),
                kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="egglog.core.f64.add", args=[value, value])),
            ),
        )
    elif invalid in {"union_oob", "union_cycle"}:
        value = builder.reserve("nodes")
        builder.fill(
            "nodes",
            value,
            pb.Node(
                sort_id=builder.import_ref(f64.__egg_sort__),
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
    pair_sort = builder.import_ref(Pair[i64, f64].__egg_sort__)
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
                            pb.Arg(sort=builder.import_ref(i64.__egg_sort__)),
                            pb.Arg(sort=builder.import_ref(f64.__egg_sort__)),
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
            sort_id=builder.import_ref(Vec[Pair[i64, f64]].__egg_sort__),
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
            sort_id=builder.import_ref(Map[i64, f64].__egg_sort__),
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
    if source == "dict":
        expression = convert({1: 2 if same_sort else 2.0}, target)
        assert expression.value == {i64(1): value}
        assert str(expression) == f"Map[i64, {value_type}].empty().put({value.value!r}, 1)"
        assert [argument.value for argument in get_callable_args(expression)[1:]] == [value.value, 1]
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
    expression = convert({1: 2.0}, target)
    assert str(expression) == "Mapping[i64, f64].vacant().put(1, 2.0)"
    assert expression.value[i64(1)].value == 2.0
    from_entries = target.from_entries(Vec(Pair(i64(1), f64(2.0))))
    assert from_entries.value[i64(1)].value == 2.0
    assert ".from_entries(" in str(from_entries.lookup(i64(1)))
    assert str(from_entries.lookup(i64(1))).endswith(".lookup(1)")
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


def test_vec_generic_authoring_local_protocols_and_pretty(vec_runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Vec authoring/inspection accessed an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
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
    argument = get_callable_args(expression)[0]
    assert isinstance(argument, vec_runtime[i64])
    assert tuple(child.value for child in argument.value) == (1, 2)
    assert get_callable_args(size(()))[0].value == ()
    assert get_type_args() == ()


def test_vec_annotation_protocol_preserves_lookup_and_pending_classes(vec_runtime):
    class Token(Expr):
        def __init__(self, value: i64) -> None: ...

    pending = Token.__egg_context__.prepare
    assert Token.__args__ == ()
    assert Token.__egg_context__.prepare is pending
    assert i64.__args__ == ()
    assert vec_runtime[i64].__args__ == (i64,)
    assert Pair[i64, f64].__args__ == (i64, f64)
    assert len(vec_runtime.__args__) == 1

    def annotated(value: Pair[i64, f64]) -> VecLike[i64, int]: ...

    hints = get_type_hints(annotated)
    assert hints["value"] == Pair[i64, f64]
    with pytest.raises(TypeError):
        _ = VecLike[i64, int, float]

    def wrong(value: VecLike[int, int]) -> i64: ...

    with pytest.raises(TypeError, match="Type arguments must resolve"):
        get_type_hints(wrong)
    with pytest.raises(AttributeError, match="no generated Python member"):
        _ = vec_runtime[i64].missing
    with pytest.raises(AttributeError):
        _ = Token.missing
    assert Token.__egg_context__.prepare is None  # Ordinary lookup still prepares.


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
            _vec_value(NominalVec())
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
    assert isinstance(pair, Pair[i64, f64])
    assert resolve_callable(computed_pair).read().kind.field == "function"
    assert {d.kind.value.name for d in pack([pair.__egg_ref__]).program.declarations} >= {"Pair", "i64", "f64"}
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
    assert [child.__egg_ref__ for child in value.value] == (
        [] if empty else [owner.ref("nodes", i) for i in [first, second, first]]
    )
    assert str(value) == ("Vec[i64].empty()" if empty else "Vec[i64](i64(1), i64(2), i64(1))")
    assert get_callable_args(value) is None


def test_vec_getitem_pretty_uses_public_syntax(vec_runtime):
    assert str(vec_runtime[i64](i64(1))[0]) == "Vec[i64](1)[0]"


def test_pair_proposed_authoring_and_symbolic_inspection(pair_runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pair authoring accessed the engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
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
    assert contexts == [(i64, f64)]
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
    assert calls == [(value, (i64, f64))]
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
    scalar = builder.import_ref(i64.__egg_sort__)
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
    left, right = expression.value
    assert left.__egg_ref__ == owner.ref("nodes", first)
    assert right.__egg_ref__ == owner.ref("nodes", second)
    assert (left.__egg_ref__ == right.__egg_ref__) == same
    assert (StructuralView(left.__egg_ref__) == StructuralView(right.__egg_ref__)) == same
    authored = pair_runtime(left, right)
    assert authored.value[0].__egg_ref__ == left.__egg_ref__
    assert authored.value[1].__egg_ref__ == right.__egg_ref__
    encoded = pack([expression.__egg_ref__]).program
    assert any(n.kind.field == "primitive_value" and n.kind.value.value.field == "pair" for n in encoded.nodes)


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
    assert [child.value for child in expression.value] == [0, 2.0]
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
        _pair_value(NominalPair(1, 2.0))
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
        pb.Node(sort_id=builder.import_ref(target.__egg_sort__), kind=Oneof[Literal["var"], str]("var", "pair")),
    )
    sentinel = RuntimeExpr(builder.publish().ref("nodes", node))
    converter(tuple, Pair, lambda _: sentinel, cost=0)
    result = convert((i64(1), f64(2.0)), Pair[i64, f64])
    assert result is sentinel


@pytest.mark.parametrize("doc", [None, "", "function documentation"])
def test_function_source_records_defaults_and_metadata(doc, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Function authoring accessed the engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
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
    expression = score(1)
    assert str(expression) == "score(1)"
    assert str(score(offset=3, x=1)) == "score(1, 3)"
    assert str(set_(score(1, 2)).to(7)) == "set_(score(1)).to(i64(7))"
    assert str(set_(score(1, 3)).to(7)) == "set_(score(1, 3)).to(i64(7))"
    default.__egg_ref__ = i64(9).__egg_ref__
    assert [a.value for a in get_callable_args(score(1))] == [1, 2]
    recovered = get_callable_fn(expression)
    assert recovered.__module__ == __name__
    assert recovered.__doc__ == doc
    assert recovered == score
    assert get_callable_args(expression, score) is not None
    assert get_callable_args(score(offset=2, x=1))[1].value == 2
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
    definition = Documented.__egg_definition__.read()
    assert definition.has_field("doc") == (doc is not None)
    initializer = resolve_callable(Documented)
    assert initializer.read().has_field("doc") == (doc is not None)
    assert RuntimeFunction(initializer, 0, Documented).__doc__ == doc


def test_free_function_identity_uses_python_name_not_signature_or_owner():
    class Box(Expr): ...

    def first(x: i64 = i64(1)) -> i64: ...

    def second(x: f64 = f64(2.0)) -> Box: ...

    second.__name__ = first.__name__
    left, right = function(first), function(egg_fn="different native name")(second)
    assert left == right
    assert hash(left) == hash(right)
    assert {left: "same callable"}[right] == "same callable"
    assert get_callable_args(left(), right)[0].value == 1
    second.__module__ = "different_module"
    other_module = function(second)
    assert left != other_module
    first.__name__ = "different_name"
    assert left != function(first)


def test_member_deconstruction_filter_ignores_receiver_binding():
    value = i64(1)
    expression = value + 2
    # Numeric special methods are Python forwarders, not callable records.
    assert get_callable_args(expression, value.__add__) is None
    bound_member = RuntimeExpr.__getattr__(value, "__add__")
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
        def __eq__(self, other) -> bool:
            calls.append(1)
            if raises:
                msg = "bound equality marker"
                raise ValueError(msg)
            return False

    token = constant("token", Token)
    member = i64.__add__
    # Comparison only: deliberately do not invoke an i64 member with Token.
    left = RuntimeFunction(member.__egg_ref__, member.__egg_view__, token)
    right = RuntimeFunction(member.__egg_ref__, member.__egg_view__, token)
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
    assert all(get_callable_fn(value).__egg_ref__ == resolve_callable(leaf) for value in values)
    assert leaf.__egg_prepare__ is None


def test_function_annotation_locals_shadow_globals():
    i64 = f64  # noqa: F841 -- get_type_hints resolves the quoted local name

    @function
    def measure(value: "i64") -> "i64": ...

    expression = measure(1.5)
    assert isinstance(expression, f64)
    assert get_callable_args(expression)[0].value == 1.5


def test_bodyless_none_and_docstring_do_not_execute_python():
    @function
    def plain() -> i64:
        return None

    @function
    def documented() -> f64:
        """Only documentation."""

    assert resolve_callable(plain).read().kind.field == "function"
    assert resolve_callable(documented).read().kind.field == "function"
    assert str(plain()) == "plain()"
    assert documented.__doc__ == "Only documentation."


def test_function_source_validation_is_lazy_and_never_executes_bodies():
    calls = 0

    @function
    def body() -> i64:
        nonlocal calls
        calls += 1
        return i64(1)

    for _ in range(2):
        with pytest.raises(NotImplementedError, match="bodies"):
            body()
    assert calls == 0
    assert body.__egg_prepare__ is None
    with pytest.raises(NotImplementedError, match="bodies"):
        resolve_callable(body)

    @function(cost=1)
    def invalid() -> i64: ...

    with pytest.raises(ValueError, match="Primitive-returning callables cannot use cost"):
        invalid()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"unextractable": True}, ValueError, "Primitive-returning callables cannot be unextractable"),
        ({"ruleset": object()}, ValueError, "Explicit rulesets require a body"),
        ({"subsume": True}, ValueError, "Primitive-returning callables cannot use subsume"),
        ({"builtin": True, "merge": lambda old, new: old}, ValueError, "Builtin callables cannot use merge"),
        ({"builtin": True, "ruleset": object()}, ValueError, "Builtin callables cannot use an explicit ruleset"),
        ({"builtin": True}, NotImplementedError, "not migrated"),
        ({"merge": lambda old, new: pytest.fail("merge called")}, NotImplementedError, "not migrated"),
        ({"mutates_first_arg": True}, NotImplementedError, "not migrated"),
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
        variadic(1)
    with pytest.raises(TypeError, match="unexpected keyword"):
        function(unknown_option=True)
    with pytest.raises(TypeError, match="either a function or keyword options"):
        function(positional, unknown_option=True)


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


def test_set_and_term_authoring_own_commands_without_evaluating(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Action authoring cannot create an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    builder = Builder()
    output = builder.import_ref(i64.__egg_sort__)
    index = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["function"], pb.Function]("function", pb.Function(name="slot", output=output))
        ),
    )
    target = RuntimeFunction(builder.publish().ref("declarations", index))()
    rhs = i64(4)
    action = set_(target).to(rhs)
    with pytest.raises(TypeError, match="unhashable type: 'Action'"):
        hash(action)
    term = expr_action(target)
    rhs.__egg_ref__ = i64(9).__egg_ref__
    target.__egg_ref__ = i64(5).__egg_ref__
    packed = pack(commands=[action.__egg_ref__, term.__egg_ref__]).program
    assignment = packed.commands[0].kind.value.kind.value
    assert assignment.target.func == "slot"
    assert packed.nodes[assignment.value].kind.value.value.value == 4
    assert packed.commands[1].kind.value.kind.field == "term"
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

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    left = constant("score", i64, egg_name="score")
    right = constant("score", i64, egg_name="score")
    first, second = set_(left).to(1), set_(right).to(1)
    assert first == second
    assert (first != second) is False
    assert first != set_(right).to(2)
    assert first != expr_action(right)
    assert first != set_(constant("other", i64, egg_name="other")).to(1)
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
        return expr_action(token), Token.__egg_hooks__["tag"]

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


def test_action_equality_preserves_union_identity_and_order():
    union = eq(i64(1)).to(i64(2))
    assert expr_action(RuntimeExpr(union.__egg_ref__)) == expr_action(RuntimeExpr(union.__egg_ref__))
    independent = eq(i64(1)).to(i64(2))
    assert expr_action(RuntimeExpr(union.__egg_ref__)) != expr_action(RuntimeExpr(independent.__egg_ref__))
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


def test_valid_but_unmigrated_constant_forms_fail_without_callbacks():
    def forbidden(old, new):
        pytest.fail("Unmigrated merge callback executed")

    with pytest.raises(NotImplementedError, match="defaults, merges and rulesets"):
        constant("default", i64, i64(3))
    with pytest.raises(NotImplementedError, match="defaults, merges and rulesets"):
        constant("merge", i64, merge=forbidden)


@pytest.mark.parametrize("egg_name", [None, ""])
def test_constant_generated_name_preserves_percent_prefix(egg_name):
    value = constant("current", i64, egg_name=egg_name)
    call = value.__egg_ref__.read().kind.value
    assert call.func == f"%{__name__.replace('.', '_')}_current"


def test_constants_own_distinct_canonical_presentation_and_declaration_kinds(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Constant authoring cannot create an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
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
    assert get_constant_name(table).name == "current"
    assert get_constant_name(table).module == __name__
    assert get_constant_name(i64(1)) is None
    assert get_constant_name(i64(1) + 2) is None
    with pytest.raises(TypeError, match="Expected Expression"):
        get_constant_name(1)
    with pytest.raises(TypeError, match="table-backed"):
        set_(token).to(token)
    packed = pack([table.__egg_ref__])
    program = pb.Program.from_binary(packed.program.to_binary())
    declaration = next(d for d in program.declarations if d.kind.field == "function")
    declaration.bindings.python.views[0].kind = pb.PythonCallKind.FUNCTION
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

        return constant("token", Token), weakref(Token), weakref(Token.__egg_definition__.owner)

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

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    default = i64(5)

    class Num(Expr, egg_sort="exact sort"):
        @method(egg_fn="exact constructor")
        def __init__(self, value: i64 = default) -> None: ...

    first = Num()
    declaration = resolve_callable(Num)
    record = declaration.read()
    assert record.kind.value.name == "exact constructor"
    assert Num.__egg_definition__.read().kind.value.name == "exact sort"
    assert record.bindings.python.views[0].params[0].name == "value"
    assert get_callable_args(first, Num)[0].value == 5
    default.__replace_expr__(i64(9))
    assert get_callable_args(Num(), Num)[0].value == 5
    assert expr_parts(Num(value=3)) == expr_parts(Num(3))
    assert Num.__egg_context__.prepare is None
    Num.__egg_attr_cache__.clear()
    Num.__egg_context__.catalog.members.clear()
    Num.__egg_context__.catalog.reindex()
    assert resolve_callable(Num) == declaration
    assert get_callable_args(Num(), Num)[0].value == 5
    with pytest.raises(TypeError):
        Num(1, value=2)
    with pytest.raises(TypeError):
        Num(unknown=2)
    with pytest.raises(TypeError):
        Num(1, 2)
    with pytest.raises(ConvertError, match="Cannot convert"):
        Num("wrong sort")


def test_user_preserved_protocols_and_match_args():
    class Local(Expr):
        def __init__(self) -> None: ...

        __match_args__ = ("value",)

        @method(preserve=True)
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
        def __eq__(self, other) -> bool:
            return other == "sentinel"

        @method(preserve=True)
        def __ne__(self, other) -> bool:
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

        expression = Local()
        return expression, weakref(Local), weakref(Local.__egg_definition__.owner), weakref(Local.__egg_context__)

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
    assert StructuralView(left.owner.ref("sorts", left.read().kind.value.inputs[0].sort)) == StructuralView(
        Right.__egg_sort__
    )
    assert StructuralView(right.owner.ref("sorts", right.read().kind.value.inputs[0].sort)) == StructuralView(
        Left.__egg_sort__
    )


def test_constructor_scope_publishes_once_across_authoring_threads():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    calls = []
    prepare = Num.__egg_context__.prepare

    def delayed_prepare(cls):
        calls.append(cls)
        sleep(0.02)  # Hold first compilation open while other authors enter.
        return prepare(cls)

    Num.__egg_context__.prepare = delayed_prepare
    with ThreadPoolExecutor(max_workers=4) as executor:
        expressions = list(executor.map(Num, range(8)))
    assert calls == [Num]
    assert [get_callable_args(expression, Num)[0].value for expression in expressions] == list(range(8))


@pytest.mark.parametrize(
    "options",
    [
        {"cost": 1},
        {"merge": lambda old, new: old},
        {"mutates_self": True},
        {"unextractable": True},
        {"subsume": True},
        {"reverse_args": True},
        {"preserve": True},
    ],
)
def test_unmigrated_constructor_options_never_publish_members(options):
    class Invalid(Expr):
        @method(**options)
        def __init__(self) -> None: ...

    with pytest.raises(NotImplementedError, match="options"):
        Invalid()
    assert Invalid.__egg_context__.catalog is None
    assert not Invalid.__egg_attr_cache__


@pytest.mark.parametrize("options", [{"cost": 1}, {}])
def test_wrapped_ignored_name_rejects_unsupported_members_before_publication(options):
    class Num(Expr):
        def __init__(self) -> None: ...

        @method(**options)
        def __radd__(self, other: i64) -> i64: ...

    with pytest.raises(NotImplementedError, match="__radd__"):
        Num()
    assert Num.__egg_context__.catalog is None
    assert not Num.__egg_attr_cache__


def test_preserved_wrapped_ignored_name_remains_a_local_hook():
    class Num(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __radd__(self, other: int) -> int:
            return other + 1

    expression = Num()
    assert Num.__radd__(expression, 2) == 3
    assert {name for _, name in Num.__egg_context__.catalog.members} == {"__init__"}


def test_unsupported_class_members_and_parameter_kinds_are_not_ignored():
    class HasBody(Expr):
        def __init__(self) -> None: ...

        def other(self) -> i64:
            return i64(1)

    class KeywordOnly(Expr):
        def __init__(self, *, value: i64) -> None: ...

    for cls in (HasBody, KeywordOnly):
        with pytest.raises(NotImplementedError):
            cls()
        assert cls.__egg_context__.catalog is None
        assert not cls.__egg_attr_cache__


def test_concrete_member_records_defaults_and_unbound_views(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Class authoring accessed an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
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

    expression = Node()
    assert expr_parts(Node.score(expression, 2)) == expr_parts(expression.score(x=2))
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
        Node.__egg_sort__
    )
    assert resolve_callable(Node.make).read().kind.field == "constructor"
    assert resolve_callable(Node.ZERO).read().kind.field == "constructor"
    assert resolve_callable(Node.COUNT).read().kind.field == "function"
    default.__replace_expr__(i64(9))
    assert get_callable_args(expression.score())[1].value == 4
    assert get_callable_args(Node.make())[0].value == 4
    Node.__egg_attr_cache__.clear()
    Node.__egg_context__.catalog.members.clear()
    Node.__egg_context__.catalog.reindex()
    assert resolve_callable(Node.score) == score
    assert get_callable_args(expression.score())[1].value == 4
    with pytest.raises(TypeError):
        expression.score(1, x=2)
    with pytest.raises(TypeError):
        Node.make(unknown=1)
    with pytest.raises(ConvertError):
        expression.score(f64(1))
    with pytest.raises(ConvertError):
        Node.score(i64(1), i64(2))


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
    assert isinstance(first.VALUE, later)
    assert isinstance(later.ROOT, first)
    assert isinstance(later.ROOT.choose(first.VALUE), later)
    assert later.__egg_context__.catalog is not None


@pytest.mark.parametrize("descriptor", [property, classmethod])
def test_member_decorator_order_preserves_value_error(descriptor):
    class Invalid(Expr):
        def __init__(self) -> None: ...

        @descriptor
        @method(egg_fn="wrong order")
        def member(self) -> i64: ...

    with pytest.raises(ValueError, match="above @classmethod or @property"):
        Invalid()
    assert Invalid.__egg_context__.catalog is None
    assert not Invalid.__egg_attr_cache__


@pytest.mark.parametrize("doc", [None, "", "member documentation"])
def test_concrete_member_documentation_presence(doc):
    class Documented(Expr):
        def __init__(self) -> None: ...
        def value(self) -> i64: ...

        value.__doc__ = doc

        @classmethod
        def named(cls) -> i64: ...

        named.__func__.__doc__ = doc

        @property
        def weight(self) -> i64: ...

        weight.fget.__doc__ = doc

    for fn in (Documented.value, Documented.named, Documented.weight):
        assert fn.__doc__ == doc
        assert fn.__module__ == __name__
        assert resolve_callable(fn).read().has_field("doc") == (doc is not None)


def test_member_scope_failure_does_not_publish_or_execute_body():
    calls = []

    class Invalid(Expr):
        VALUE: ClassVar[i64]

        def __init__(self) -> None: ...
        def valid(self) -> i64: ...

        def body(self) -> i64:
            calls.append(1)
            return i64(2)

    with pytest.raises(NotImplementedError, match="bodies"):
        Invalid.valid
    assert calls == []
    assert Invalid.__egg_context__.catalog is None
    assert not Invalid.__egg_attr_cache__


@pytest.mark.parametrize("annotation", ["i64", "ClassVar[i64] = None", "ClassVar[i64] = i64(1)"])
def test_invalid_or_unmigrated_class_annotations_are_explicit(annotation):
    namespace = {"Expr": Expr, "ClassVar": ClassVar, "i64": i64, "__name__": __name__}
    exec(f"class Invalid(Expr):\n    VALUE: {annotation}\n", namespace)
    cls = namespace["Invalid"]
    with pytest.raises(NotImplementedError, match=r"expected a ClassVar|defaults"):
        cls.VALUE
    assert cls.__egg_context__.catalog is None


def test_retained_member_context_lifetime_and_local_hooks():
    def make():
        class Local(Expr):
            ROOT: ClassVar["Local"]

            def child(self) -> "Local": ...

            @method(preserve=True)
            def tag(self):
                return "retained member"

        expression = Local.ROOT.child()
        return expression, weakref(Local), weakref(Local.__egg_context__), weakref(Local.__egg_definition__.owner)

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
            sort_id=builder.import_ref(Child.__egg_sort__),
            kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[child, child])),
        ),
    )
    default = RuntimeExpr(builder.publish().ref("nodes", index))

    class Parent(Expr):
        def __init__(self) -> None: ...
        def child(self, value: Child = default) -> Child: ...

    expression = Parent()
    first = get_callable_args(expression.child())[1]
    default.__replace_expr__(Child())
    second = get_callable_args(expression.child())[1]
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

    with pytest.raises(ValueError, match=r"Primitive-returning callables cannot (use|be)"):
        Invalid()
    assert Invalid.__egg_context__.catalog is None


def test_class_ruleset_binding_remains_unsupported_not_invalid():
    from egglog import ruleset  # noqa: PLC0415

    class ValidButUnmigrated(Expr, ruleset=ruleset(name="class-rules")):
        def __init__(self) -> None: ...
        def value(self) -> i64: ...

    with pytest.raises(NotImplementedError, match="Class ruleset binding"):
        ValidButUnmigrated()


def test_user_sort_cannot_borrow_builtin_codec_or_members_by_name():
    class Shadow(Expr, egg_sort="i64"):
        def __init__(self, value: i64) -> None: ...

    expression = Shadow(3)
    assert expression.__egg_ref__.read().kind.field == "call"
    assert get_callable_args(expression, Shadow)[0].value == 3
    with pytest.raises(AttributeError, match="member"):
        _ = Shadow.__add__


def test_unresolved_constructor_annotation_does_not_publish_members():
    class Missing(Expr):
        def __init__(self, value: "NotDefinedHere") -> None: ...  # noqa: F821 -- intentional unresolved annotation

    with pytest.raises(NameError, match="NotDefinedHere"):
        Missing(1)
    assert Missing.__egg_context__.catalog is None
    assert not Missing.__egg_attr_cache__


def test_user_default_expansion_has_fresh_union_identity_and_retains_sharing():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    child = Num(1)
    builder = Builder()
    member = builder.import_ref(child.__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Num.__egg_sort__),
            kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[member, member])),
        ),
    )
    default = RuntimeExpr(builder.publish().ref("nodes", root))

    class Box(Expr):
        def __init__(self, value: Num = default) -> None: ...

    first = get_callable_args(Box(), Box)[0]
    second = get_callable_args(Box(), Box)[0]
    assert first.__egg_ref__ != second.__egg_ref__ != default.__egg_ref__
    assert expr_parts(first) != expr_parts(second)
    for expression in (first, second):
        members = expression.__egg_ref__.read().kind.value.members
        assert len(members) == 2
        assert members[0] == members[1]

    @function
    def pair(left: Num = default, right: Num = default) -> Box: ...

    first_pair = get_callable_args(pair())
    second_pair = get_callable_args(pair())
    assert first_pair[0].__egg_ref__ == first_pair[1].__egg_ref__
    assert second_pair[0].__egg_ref__ == second_pair[1].__egg_ref__
    assert first_pair[0].__egg_ref__ != second_pair[0].__egg_ref__
    assert first_pair[0].__egg_ref__ != default.__egg_ref__


def test_reconstructed_constructor_uses_changed_protobuf_parameter_not_python_signature():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    program = pack([Num(1).__egg_ref__]).program
    index, constructor = next(
        (index, declaration)
        for index, declaration in enumerate(program.declarations)
        if declaration.kind.field == "constructor"
    )
    constructor.bindings.python.views[0].params[0].name = "renamed"
    owner = Builder.from_program(program).publish()
    function = RuntimeFunction(owner.ref("declarations", index), 0, Num)
    result = function(renamed=8)
    assert get_callable_args(result)[0].value == 8
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
        return _author_call(reference, None, values, {}, node_of=lambda x: x.__egg_ref__, convert=forbidden)[0]

    arguments = (i64(0), i64(1), f64(2.0), i64(3), f64(4.0))
    root = call(*arguments)
    node = root.read()
    assert StructuralView(root.owner.ref("sorts", node.sort_id)) == StructuralView(f64.__egg_sort__)
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
    signature = next(d.kind.value.typing.value for d in source.declarations if d.kind.field == "host_primitive")
    assert len(signature.varargs) == 2
    signature.type_params.append("Unused")
    adopted = Builder.from_program(source).publish()
    incomplete = next(
        adopted.ref("declarations", index)
        for index, declaration in enumerate(source.declarations)
        if declaration.kind.field == "host_primitive"
    )
    with pytest.raises(TypeConstraintError, match="Not every generic parameter"):
        _author_call(incomplete, None, arguments, {}, node_of=lambda x: x.__egg_ref__, convert=forbidden)
