"""Normal Python syntax over the generated-record runtime, without an engine."""

import gc
from collections.abc import Callable
from copy import copy
from fractions import Fraction
from functools import partial
from typing import ClassVar, cast

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb

from egglog import (
    Bool,
    BoolLike,
    ConvertError,
    EGraph,
    Expr,
    ExprCallable,
    ExprValueError,
    Fact,
    Map,
    Pair,
    String,
    StringLike,
    Unit,
    Vec,
    _program,
    bindings,
    constant,
    eq,
    expr_parts,
    f64,
    function,
    i64,
    join,
    method,
    ne,
    set_,
    union,
    var,
)
from egglog.deconstruct import get_callable_args, get_callable_fn, get_constant_name, get_literal_value
from egglog.runtime import RuntimeExpr, RuntimeFunction


def test_named_function_values_author_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Named function authoring accessed an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    from egglog import UnstableFn  # noqa: PLC0415 -- exercise the public import under the no-engine guard

    @function
    def combine(left: i64, right: i64) -> i64: ...  # type: ignore[empty-body]

    whole = UnstableFn(combine)
    assert whole.value == combine
    assert str(whole) == "combine"
    prefix = UnstableFn(combine, i64(10))
    assert isinstance(prefix.value, partial)
    assert prefix.value.func == combine
    assert prefix.value.args == (i64(10),)
    assert expr_parts(prefix.value(i64(2))) == expr_parts(combine(i64(10), i64(2)))
    assert expr_parts(copy(prefix)) == expr_parts(prefix)
    assert str(prefix) == "partial(combine, i64(10))"
    assert isinstance(prefix(i64(2)), i64)
    assert isinstance(UnstableFn(combine, i64(10), i64(2))(), i64)
    assert expr_parts(UnstableFn[i64, i64](combine, i64(10))) == expr_parts(prefix)
    assert hash(copy(prefix)) == hash(prefix)
    arguments = get_callable_args(whole(i64(10), i64(2)))
    assert arguments is not None
    assert arguments == tuple(cast("RuntimeExpr", value) for value in (whole, i64(10), i64(2)))
    nullary = UnstableFn(combine, i64(10), i64(2))
    assert isinstance(nullary.value, partial)
    assert nullary.value.func == combine
    assert nullary.value.args == (i64(10), i64(2))
    arguments = get_callable_args(nullary())
    assert arguments is not None
    assert arguments == (cast("RuntimeExpr", nullary),)

    class Accumulator(Expr):
        def __init__(self) -> None: ...
        def add(self, value: i64) -> i64: ...  # type: ignore[empty-body]

        @classmethod
        def make(cls, value: i64) -> "Accumulator": ...  # type: ignore[empty-body]

    receiver = Accumulator()
    assert expr_parts(UnstableFn(receiver.add)) == expr_parts(UnstableFn(Accumulator.add, receiver))
    assert expr_parts(UnstableFn(receiver.add, i64(2))) == expr_parts(UnstableFn(Accumulator.add, receiver, i64(2)))
    method_value = UnstableFn(receiver.add).value
    assert isinstance(method_value, partial)
    assert method_value.func == Accumulator.add
    assert method_value.args == (receiver,)
    class_value = UnstableFn(Accumulator.make, i64(2)).value
    assert isinstance(class_value, partial)
    assert class_value.func == Accumulator.make
    assert class_value.args == (i64(2),)
    assert expr_parts(UnstableFn(i64(10).__add__)) == expr_parts(UnstableFn(i64.__add__, i64(10)))

    @function
    def consume(callback: Callable[[i64], i64]) -> i64: ...  # type: ignore[empty-body]

    assert expr_parts(consume(partial(combine, i64(10)))) == expr_parts(consume(prefix))
    assert expr_parts(consume(UnstableFn(i64.__invert__))) == expr_parts(consume(i64.__invert__))
    nested = UnstableFn(UnstableFn[i64, i64].__call__, prefix)
    assert isinstance(nested(i64(2)), i64)
    assert isinstance(nested.value, partial)
    assert nested.value.args == (prefix,)

    symbolic = var("callback", UnstableFn[i64, i64])
    assert get_literal_value(symbolic) is None
    with pytest.raises(ExprValueError, match=r"UnstableFn\(f, \*args\)"):
        _ = symbolic.value
    apply = cast("RuntimeExpr", prefix).__getattr__("__call__")
    assert isinstance(apply, RuntimeFunction)
    with pytest.raises(TypeError, match="expects 1 arguments"):
        apply()
    with pytest.raises(TypeError, match="expects 1 arguments"):
        apply(i64(1), i64(2))
    with pytest.raises(ConvertError):
        apply(String("wrong sort"))


def test_named_eager_function_values_author_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from egglog import UnstableFn  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Named eager authoring accessed an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)

    def make_combine():
        @function
        def combine(left: i64, right: i64 = i64(2)) -> i64:
            """Named eager addition."""
            return left + right

        return combine

    combine = make_combine()
    whole = UnstableFn(combine)
    prefix = UnstableFn(combine, i64(10))
    nullary = UnstableFn(combine, i64(10), i64(2))
    assert whole.value == combine == make_combine()
    assert hash(whole.value) == hash(combine) == hash(make_combine())
    assert str(whole) == "combine"
    assert str(prefix) == "partial(combine, i64(10))"
    assert str(nullary) == "partial(combine, i64(10), i64(2))"
    assert isinstance(prefix.value, partial)
    assert prefix.value.func == combine
    assert prefix.value.args == (i64(10),)
    assert expr_parts(prefix.value(i64(2))) == expr_parts(combine(i64(10)))
    assert isinstance(whole(i64(10), i64(2)), i64)
    assert isinstance(prefix(i64(2)), i64)
    assert isinstance(nullary(), i64)
    assert get_callable_fn(combine(i64(10))) == combine
    assert combine.__doc__ == "Named eager addition."

    @function
    def different(left: i64, right: i64) -> i64:
        return left + right

    assert whole.value != different

    class Box(Expr):
        def __init__(self, value: i64) -> None: ...

    @function
    def boxed(value: i64) -> Box:
        return Box(value + 1)

    assert isinstance(boxed(i64(3)), Box)
    assert isinstance(UnstableFn(boxed, i64(3))(), Box)
    assert UnstableFn(boxed).value == boxed


def test_function_sizes_public_forms_bytes() -> None:
    @function
    def f() -> i64: ...  # type: ignore[empty-body]

    class Counted(Expr):
        X: ClassVar[i64]

        def __init__(self) -> None: ...
        def m(self) -> i64: ...  # type: ignore[empty-body]

        @property
        def p(self) -> i64: ...  # type: ignore[empty-body]

        @classmethod
        def cm(cls) -> i64: ...  # type: ignore[empty-body]

        @method(preserve=True)
        def tag(self) -> str:
            return "retained Counted hook"

    current = constant("counted-current", i64)
    graph = EGraph(
        set_(f()).to(i64(0)),
        set_(Counted().m()).to(i64(0)),
        set_(Counted.X).to(i64(0)),
        set_(Counted().p).to(i64(0)),
        set_(current).to(i64(0)),
        set_(Counted.cm()).to(i64(0)),
    )
    targets: tuple[ExprCallable, ...] = (
        Counted,
        f,
        Counted.m,
        Counted.X,
        cast("ExprCallable", Counted.p),
        current,
        Counted.cm,
    )
    for target in targets:
        assert graph.function_size(target) == 1
    rows = graph.all_function_sizes()
    assert set(rows) == {(target, 1) for target in targets}
    del graph
    gc.collect()
    assert set(rows) == {(target, 1) for target in targets}
    restored = next(target for target, _ in rows if target == Counted)
    assert hash(restored) == hash(Counted)
    assert cast("type[Counted]", restored)().tag() == "retained Counted hook"


def test_function_sizes_empty_order_and_primitive_rejection_bytes() -> None:
    @function(egg_fn='z-count "last"')
    def last(value: i64) -> i64: ...  # type: ignore[empty-body]

    @function(egg_fn="a-count first")
    def first(value: i64) -> i64: ...  # type: ignore[empty-body]

    graph = EGraph()
    assert graph.all_function_sizes() == []
    assert graph.function_size(last) == 0
    assert graph.all_function_sizes() == [(last, 0)]
    graph.register(set_(first(i64(1))).to(i64(7)), set_(last(i64(1))).to(i64(8)))
    assert graph.all_function_sizes() == [(first, 1), (last, 1)]
    with pytest.raises(ValueError, match="table-backed"):
        graph.function_size(i64.__add__)
    assert graph.all_function_sizes() == [(first, 1), (last, 1)]


def test_function_values_public_bytes() -> None:
    @function(egg_fn="table with spaces")
    def score(value: i64) -> i64: ...  # type: ignore[empty-body]

    graph = EGraph()
    assert graph.function_values(score, 0) == {}
    assert graph.function_values(score) == {}
    graph.register(set_(score(i64(1))).to(i64(2)), set_(score(i64(3))).to(i64(4)))
    rows = graph.function_values(score)
    assert rows == {score(i64(1)): i64(2), score(i64(3)): i64(4)}
    for limit in (0, False, 1, True, 2, 2**64 - 1):
        assert list(graph.function_values(score, limit).items()) == list(rows.items())[:limit]
    with pytest.raises(ValueError, match="table-backed"):
        graph.function_values(i64.__add__)


def test_function_values_nullary_and_container_bytes() -> None:
    current = constant("current", i64)

    @function
    def entry(values: Vec[i64]) -> Pair[i64, i64]: ...  # type: ignore[empty-body]

    graph = EGraph(set_(current).to(i64(7)), set_(entry(Vec(i64(1), i64(2)))).to(Pair(i64(3), i64(3))))
    assert graph.function_values(current) == {current: i64(7)}
    ((key, result),) = graph.function_values(entry).items()
    assert get_callable_fn(key) == entry
    args = get_callable_args(key, entry)
    assert args is not None
    assert len(args) == 1
    argument = cast("Vec[i64]", args[0])
    assert tuple(child.value for child in argument.value) == (1, 2)
    assert tuple(child.value for child in result.value) == (3, 3)
    del graph
    gc.collect()
    assert tuple(child.value for child in argument.value) == (1, 2)
    assert tuple(child.value for child in result.value) == (3, 3)
    assert EGraph().extract(result).value == (i64(3), i64(3))


def test_function_values_constructor_key_keeps_selected_table_bytes() -> None:
    class Number(Expr):
        def __init__(self, value: i64) -> None: ...

        @method(preserve=True)
        def tag(self) -> str:
            return "local Number"

    @function
    def wrap(value: Number) -> Number: ...  # type: ignore[empty-body]

    value = Number(i64(1))
    graph = EGraph(union(wrap(value)).with_(value))
    ((key, result),) = graph.function_values(wrap).items()
    assert get_callable_fn(key) == wrap
    assert get_callable_args(key, wrap) == get_callable_args(wrap(value), wrap)
    assert get_callable_fn(result) == Number
    assert result == value
    assert result.tag() == "local Number"
    del graph
    gc.collect()
    assert str(key) == "wrap(Number(1))"
    assert result.tag() == "local Number"


def test_set_public_native_bytes() -> None:
    # Requires the genuine Set wheel; source-only tests cannot accept this gate.
    from egglog import Set, convert  # noqa: PLC0415

    graph = EGraph()
    one, two, three = i64(1), i64(2), i64(3)
    original = Set(two, one, two)
    for expression, expected in (
        (Set[i64](), set()),
        (Set[i64].empty(), set()),
        (convert(set(), Set[i64]), set()),
        (convert({1, 2}, Set[i64]), {one, two}),
        (original, {one, two}),
        (Set(one + 1, two), {two}),
        (original.insert(three), {one, two, three}),
        (original.remove(one), {two}),
        (original | Set(three), {one, two, three}),
        (original - Set(one), {two}),
        (original & Set(one, three), {one}),
        (Set(Pair(one, String("a"))), {Pair(one, String("a"))}),
        (Set(Vec(one, two)), {Vec(one, two)}),
        (Set(Set(one)), {Set(one)}),
    ):
        assert expression.__egg_ref__.read().kind.field == "call"
        extracted = graph.extract(expression)
        assert extracted.__egg_ref__.read().kind.value.value.field == "set"
        if not expected or isinstance(next(iter(expected)), i64):
            assert extracted.value == expected
        else:
            # Nested children become formed values rather than constructor Calls.
            assert len(extracted) == 1
            assert next(iter(extracted)).value == graph.extract(next(iter(expression))).value
        assert EGraph().extract(extracted).value == extracted.value
        record = extracted.__egg_ref__.read()
        refs = [extracted.__egg_ref__.owner.ref("nodes", item) for item in record.kind.value.value.value.items]
        assert [item.__egg_ref__ for item in extracted] == list(dict.fromkeys(refs))
    assert graph.extract(original.length()).value == 2
    graph.check(original.contains(one), original.not_contains(three))
    graph.check_fail(original.contains(three))
    graph.check_fail(original.not_contains(one))


def test_exact_wave_public_native_bytes() -> None:
    # Imported here so source-only scalar regressions remain collectable before
    # the actual rational catalog is packaged. This gate requires the new wheel.
    from egglog import BigInt, BigRat, Rational, convert  # noqa: PLC0415

    graph = EGraph()
    huge = 10**100 + 1
    for expression, expected in (
        (Rational(1, 2), Fraction(1, 2)),
        (Rational(1, -2), Fraction(-1, 2)),
        (Rational(i64(1) + 2, 2), Fraction(3, 2)),
        (Rational(1, 2) + Fraction(1, 3), Fraction(5, 6)),
        (Fraction(1, 3) + Rational(1, 2), Fraction(5, 6)),
        (BigInt(10), 10),
        (BigInt(i64(1) + 2), 3),
        (BigInt.from_string(str(huge)), huge),
        (BigRat(1, 2), Fraction(1, 2)),
        (BigRat(BigInt.from_string(str(huge)), 2), Fraction(huge, 2)),
        (convert(Fraction(1, 2), BigRat), Fraction(1, 2)),
    ):
        assert expression.__egg_ref__.read().kind.field == "call"
        extracted = graph.extract(expression)
        assert extracted.__egg_ref__.read().kind.field == "primitive_value"
        assert extracted.value == expected
        assert EGraph().extract(extracted).value == expected
    assert int(graph.extract(BigInt(10))) == 10
    assert float(graph.extract(BigRat(1, 2))) == 0.5


def test_scalar_wave_public_native_bytes() -> None:
    graph = EGraph()
    sample = "a\0b\n☃"
    for expression, expected in (
        (String(sample), sample),
        (join(), ""),
        (join(sample), sample),
        (join(sample, "!", "?"), sample + "!?"),
        (String(sample) + "!", sample + "!"),
        ("!" + String(sample), "!" + sample),
        (String(sample).replace("b", "xy"), "a\0xy\n☃"),
        (i64(-42).to_string(), "-42"),
    ):
        assert graph.extract(expression).value == expected
    assert str(graph.extract(Unit())) == "Unit()"
    assert get_literal_value(graph.extract(Unit())) is None
    for left in (False, True):
        assert graph.extract(~Bool(left)).value is (not left)
        for right in (False, True):
            for expression, expected in (
                (Bool(left) & right, left and right),
                (Bool(left) | right, left or right),
                (Bool(left) ^ right, left != right),
                (Bool(left).implies(right), not left or right),
                (left & Bool(right), left and right),
                (left | Bool(right), left or right),
                (left ^ Bool(right), left != right),
            ):
                assert graph.extract(expression).value is expected
    pair = graph.extract(Pair[String, Bool](String("a") + "b", ~Bool(True)))
    assert tuple(part.value for part in pair.value) == ("ab", False)
    mapping = graph.extract(Map[String, Bool].empty().insert("key", Bool(False) | True))
    assert mapping.value[String("key")].value is True


def test_closed_scalar_bool_comparisons_and_conversion_native_bytes() -> None:
    from egglog import count_matches  # noqa: PLC0415

    graph = EGraph()
    retained: list[tuple[Bool, bool]] = []
    for left, right in ((0, 1), (1, 1), (2, 1), (-(2**63), 2**63 - 1)):
        for expression, expected in (
            (i64(left).bool_lt(right), left < right),
            (i64(left).bool_gt(right), left > right),
            (i64(left).bool_le(right), left <= right),
            (i64(left).bool_ge(right), left >= right),
        ):
            with pytest.raises(ExprValueError):
                _ = expression.value
            with pytest.raises(ExprValueError):
                bool(expression)
            result = graph.extract(expression)
            assert isinstance(result, Bool)
            assert result.value is expected
            retained.append((result, expected))
    graph.check(i64(2) > 1, i64(1) <= 1)
    graph.check_fail(i64(1) > 1)
    graph.check_fail(i64(2) <= 1)
    assert graph.extract(f64.from_i64(i=-3)).value == -3.0
    assert graph.extract(count_matches(s="a\0a☃a", pattern="a")).value == 3
    del graph
    gc.collect()
    for result, expected in retained:
        assert bool(result) is expected
        assert EGraph().extract(result).value is expected


@pytest.mark.parametrize("values", [(), (False,), (True,), (False, False), (False, True, False, False, True)])
def test_scalar_wave_variadic_bool_native_bytes(values) -> None:
    # Canonical bytes deliberately support arities without a public Python view.
    expression = RuntimeFunction(Bool.__or__.__egg_ref__, None)(*values)
    assert EGraph().extract(expression).value is any(values)


def test_scalar_wave_functions_and_concrete_defaults_native_bytes() -> None:
    @function
    def message(prefix: StringLike = String("a") + "b", enabled: BoolLike = False) -> String: ...

    @function
    def flag(value: BoolLike = Bool(False)) -> Bool: ...

    @function
    def done() -> Unit: ...

    graph = EGraph()
    graph.register(set_(message()).to("hello"), set_(flag()).to(True), set_(done()).to(Unit()))
    assert graph.extract(message()).value == "hello"
    assert graph.extract(flag()).value is True
    assert str(graph.extract(done())) == "Unit()"


def test_concrete_class_members_public_bytes() -> None:
    class E(Expr):
        X: ClassVar[i64]

        def __init__(self) -> None: ...
        def m(self) -> i64: ...

        @property
        def p(self) -> i64: ...

        @classmethod
        def cm(cls) -> i64: ...

    graph = EGraph()
    values = [E().m(), E.X, E().p, E.cm()]
    graph.register(*(set_(value).to(i64(n)) for n, value in enumerate(values, 1)))
    for n, value in enumerate(values, 1):
        graph.check(eq(value).to(i64(n)))
        assert graph.extract(value).value == n


def test_concrete_eqsort_and_container_members_public_bytes() -> None:
    class Token(Expr):
        ROOT: ClassVar["Token"]

        @classmethod
        def make(cls, value: i64) -> "Token": ...

        def next(self) -> "Token": ...
        def values(self) -> Vec[Pair[i64, f64]]: ...

    graph = EGraph()
    root = Token.ROOT
    assert str(graph.extract(root)) == "Token.ROOT"
    assert str(graph.extract(root)) == "Token.ROOT"
    expression = Token.make(i64(1)).next()
    assert str(graph.extract(expression)) == "Token.make(1).next()"
    pairs = Vec(Pair(i64(2), f64(3.0)))
    graph.register(set_(expression.values()).to(pairs))
    graph.check(eq(expression.values()).to(pairs))
    value = graph.extract(expression.values()).value[0]
    assert tuple(part.value for part in value.value) == (2, 3.0)


def test_symbolic_getattr_public_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class A(Expr):
        def __init__(self, tag: i64) -> None: ...
        def __getattr__(self, name: StringLike) -> "A": ...

    graph = EGraph()
    transport = graph._state.transport
    requests, responses = [], []

    class AuditedTransport:
        def run(self, encoded):
            assert isinstance(encoded, bytes)
            index = len(requests)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned, host_error = transport.run(encoded)
            assert isinstance(returned, bytes)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            responses.append(pb.RunProgramResponse.from_binary(returned))
            return returned, host_error

    monkeypatch.setattr(graph._state, "transport", AuditedTransport())
    receiver = A(i64(7))
    name = String("a") + "b"
    computed = receiver.__getattr__(name)
    assert get_callable_args(computed, A.__getattr__)[1].__egg_ref__ == name.__egg_ref__
    for expression, expected in ((receiver.foo, "foo"), (computed, "ab")):
        graph.register(expression)
        result = graph.extract(expression)
        graph.check(eq(result).to(expression))
        assert get_callable_fn(result) == A.__getattr__
        arguments = get_callable_args(result, A.__getattr__)
        assert arguments is not None
        assert expr_parts(arguments[0]) == expr_parts(receiver)
        assert get_callable_args(arguments[0], A)[0].value == 7
        assert arguments[1].value == expected
    assert requests
    assert len(requests) == len(responses)
    assert all(response.error is None for response in responses)
    assert any(
        node.kind.field == "call" and node.kind.value.func == "egglog.core.string.concat"
        for request in requests
        for node in request.program.nodes
    )


def test_preserved_getattr_survives_public_result_adoption() -> None:
    class Local(Expr):
        def __init__(self, tag: i64) -> None: ...

        @method(preserve=True)
        def __getattr__(self, name) -> object:
            return None if name == "nothing" else ("local", name)

    receiver = Local(i64(9))
    graph = EGraph()
    graph.register(receiver)
    result = graph.extract(receiver)
    assert get_callable_fn(result) == Local
    assert get_callable_args(result, Local)[0].value == 9
    assert result.foo == ("local", "foo")
    assert result.nothing is None


def test_symbolic_getattr_corrupt_bytes_reject_before_execution(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class A(Expr):
        def __init__(self) -> None: ...
        def __getattr__(self, name: StringLike) -> "A": ...

    graph = EGraph()
    expression = A().foo
    graph.register(A())
    # Install the callable without inserting any of its rows.
    assert graph.function_size(A.__getattr__) == 0
    callee = A.__getattr__.__egg_ref__.read().kind.value.name
    transport = graph._state.transport
    rejected = []

    class CorruptTransport:
        def run(self, encoded):
            (tmp_path / "request-original.pb").write_bytes(encoded)
            request = pb.RunProgramRequest.from_binary(encoded)
            calls = [
                node.kind.value
                for node in request.program.nodes
                if node.kind.field == "call" and node.kind.value.func == callee
            ]
            assert len(calls) == 1
            assert len(calls[0].args) == 2
            calls[0].args.pop()
            corrupted = request.to_binary()
            assert corrupted != encoded
            (tmp_path / "request-corrupted.pb").write_bytes(corrupted)
            returned, host_error = transport.run(corrupted)
            (tmp_path / "response.pb").write_bytes(returned)
            rejected.append(pb.RunProgramResponse.from_binary(returned))
            return returned, host_error

    with monkeypatch.context() as patch:
        patch.setattr(graph._state, "transport", CorruptTransport())
        with pytest.raises(bindings.EggSmolError):
            graph.register(expression)
    assert len(rejected) == 1
    assert rejected[0].error is not None
    assert rejected[0].error.location is None
    assert graph.function_size(A.__getattr__) == 0
    graph.register(expression)
    assert graph.function_size(A.__getattr__) == 1


def test_bodyless_mutating_members_public_bytes() -> None:
    class Mutable(Expr):
        def __init__(self, tag: i64) -> None: ...
        def __delitem__(self, key: "Mutable") -> None: ...
        def __setitem__(self, key: "Mutable", value: "Mutable") -> None: ...

        @method(mutates_self=True, egg_fn="mutable update")
        def update(self, value: i64 = i64(4)) -> None: ...

    graph = EGraph()
    receiver, key, value = Mutable(i64(1)), Mutable(i64(2)), Mutable(i64(3))
    original = copy(receiver)
    del receiver[key]
    graph.register(receiver)
    deleted = graph.extract(receiver)
    assert get_callable_fn(deleted) == Mutable.__delitem__
    arguments = get_callable_args(deleted, Mutable.__delitem__)
    assert arguments is not None
    assert [expr_parts(argument) for argument in arguments] == [
        expr_parts(original),
        expr_parts(key),
    ]
    receiver[key] = value
    graph.register(receiver)
    assigned = graph.extract(receiver)
    assert get_callable_fn(assigned) == Mutable.__setitem__
    arguments = get_callable_args(assigned, Mutable.__setitem__)
    assert arguments is not None
    assert [expr_parts(argument) for argument in arguments] == [
        expr_parts(deleted),
        expr_parts(key),
        expr_parts(value),
    ]
    assert receiver.update() is None
    for _ in range(2):
        graph.register(receiver)
        result = graph.extract(receiver)
        graph.check(eq(result).to(receiver))
        assert get_callable_fn(result) == Mutable.update
        arguments = get_callable_args(result, Mutable.update)
        assert arguments is not None
        assert expr_parts(arguments[0]) == expr_parts(assigned)
        assert arguments[1].value == 4
    assert get_callable_fn(graph.extract(original)) == Mutable
    assert get_callable_args(graph.extract(original))[0].value == 1
    assert get_callable_args(graph.extract(key))[0].value == 2
    assert get_callable_args(graph.extract(value))[0].value == 3


def test_scalar_local_value_and_presentation() -> None:
    assert i64(3).value == 3
    assert f64(1.25).value == 1.25
    assert str(i64(1) + 2) == "i64(1) + 2"
    assert repr(f64(1.0) + 2.0) == "f64(1.0) + 2.0"
    assert get_literal_value(i64(1) + 2) is None
    with pytest.raises(ExprValueError, match="Cannot get Python value"):
        (i64(1) + 2).value
    with pytest.raises(TypeError, match="Expected Expression"):
        get_literal_value(1)


def test_public_pyobject_defaults_and_results_cross_bytes(monkeypatch, tmp_path) -> None:
    from egglog import PyObject, py_eval, py_exec  # noqa: PLC0415

    graph = EGraph()
    transport = graph._state.transport
    requests, responses = [], []

    class AuditedTransport:
        def run(self, encoded):
            index = len(requests)
            assert isinstance(encoded, bytes)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned, host_error = transport.run(encoded)
            assert isinstance(returned, bytes)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            responses.append(pb.RunProgramResponse.from_binary(returned))
            return returned, host_error

    monkeypatch.setattr(graph._state, "transport", AuditedTransport())
    original = {"x": 7}
    for expression, expected in (
        (py_exec("x = 10"), {"x": 10}),
        (py_eval("1 + 2"), 3),
        (py_eval("x", {"x": 1}, {"x": 2}), 2),
        (py_exec("x = y + 1", {"y": 10}), {"x": 11}),
        (PyObject(original).dict_update("x", 8), {"x": 8}),
        (PyObject.from_string("hello"), "hello"),
        (PyObject.from_int(42), 42),
        (PyObject(None), None),
        (PyObject(int)(PyObject.from_int(10)), 10),
        (PyObject(lambda x, *, factor=1: x * factor).call_extended((10,), {"factor": 3}), 30),
    ):
        result = graph.extract(expression)
        assert result.value == expected
        assert result.__egg_ref__.read().kind.value.value.field == "custom"
    assert original == {"x": 7}
    graph.check(PyObject("hello").to_string() == String("hello"))
    assert graph.extract(PyObject(True).to_bool()).value is True
    assert requests
    assert len(requests) == len(responses)
    assert all(response.error is None for response in responses)
    assert any(
        node.kind.field == "call" and node.kind.value.func == "egglog.python.pyobject.dict"
        for request in requests
        for node in request.program.nodes
    )
    monkeypatch.setattr(graph._state, "transport", transport)
    graph._state.destroy()
    assert result.value == 30


def test_public_pyobject_callback_errors_none_and_recovery() -> None:
    from egglog import PrimitiveFailure, PyObject, py_eval_fn  # noqa: PLC0415

    def callback(value):
        if value == "expected":
            msg = "no match"
            raise PrimitiveFailure(msg)
        if value == "unexpected":
            msg = "original callback failure"
            raise ValueError(msg)

    graph = EGraph()
    fn = PyObject(callback)
    with pytest.raises(ValueError, match="original callback failure") as caught:
        graph.extract(fn("unexpected"))
    assert caught.value.args == ("original callback failure",)
    with pytest.raises(bindings.EggSmolError):
        graph.extract(fn("expected"))
    assert graph.extract(fn("success")).value is None
    with pytest.warns(DeprecationWarning, match=r"use PyObject\(fn\) directly"):
        wrapped = py_eval_fn(int)
    assert graph.extract(wrapped(PyObject("12"))).value == 12


def test_scalar_call_children_are_retained_records() -> None:
    leaf = i64(1)
    expression = leaf + 2
    assert isinstance(expression, RuntimeExpr)
    assert copy(expression).__egg_ref__ == expression.__egg_ref__
    children = get_callable_args(expression)
    assert children is not None
    assert children[0].__egg_ref__ == leaf.__egg_ref__
    assert tuple(child.value for child in children) == (1, 2)
    assert get_callable_fn(expression) == i64.__add__
    assert tuple(child.__egg_ref__ for child in get_callable_args(expression, i64.__add__)) == tuple(
        child.__egg_ref__ for child in children
    )
    assert str(i64) == "i64"


def test_authoring_never_creates_native_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Authoring accessed the native engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    assert (i64(2) + 3).__egg_ref__.read().kind.value.func == "egglog.core.i64.add"


def test_structural_inspection_and_equality_fact() -> None:
    assert expr_parts(i64(1) + 2) == expr_parts(i64(1) + 2)
    assert hash(expr_parts(i64(1) + 2)) == hash(expr_parts(i64(1) + 2))
    assert bool(eq(i64(1) + 2).to(i64(1) + 2))
    assert not bool(eq(i64(1)).to(i64(2)))
    assert str(eq(i64(1)).to(i64(2))) == "eq(i64(1)).to(i64(2))"
    assert expr_parts(f64(0.0)) == expr_parts(f64(-0.0))


def test_ne_authoring_pretty_and_structural_truth_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Inequality authoring or truth accessed the native engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)

    class Compared(Expr):
        def __init__(self, value: i64) -> None: ...

    pairs = [
        (Bool(True), Bool(False)),
        (i64(1), i64(2)),
        (Compared(i64(1)), Compared(i64(2))),
        (Vec[i64](i64(1)), Vec[i64](i64(2))),
        (Vec[Compared](Compared(i64(1))), Vec[Compared](Compared(i64(2)))),
        (Pair(i64(1), String("a")), Pair(i64(1), String("b"))),
    ]
    for left, right in pairs:
        different = ne(left).to(right)
        assert isinstance(different, Unit)
        assert bool(different)
        assert bool(left != right)
        assert not bool(ne(left).to(copy(left)))
        assert not bool(left != copy(left))
        assert bool(copy(different))
        assert str(ne(left)) == repr(ne(left)) == f"ne({left})"
        assert str(different) == repr(different) == f"ne({left}).to({right})"
    assert str(ne(i64(1)).to(cast("i64", 2))) == "ne(i64(1)).to(i64(2))"
    assert bool(ne(i64(1) + 1).to(i64(2)))  # Structural, not arithmetic evaluation.
    assert not bool(ne(f64(0.0)).to(f64(-0.0)))
    with pytest.raises(ValueError, match="Can only check equality for == or !="):
        bool(Unit())
    with pytest.raises(ValueError, match="Can only check equality for == or !="):
        bool(i64(1) < 2)


def test_ne_rejects_wrong_sort() -> None:
    with pytest.raises(ConvertError):
        ne(i64(1)).to(cast("i64", String("1")))
    with pytest.raises(ConvertError):
        ne(Vec[i64](i64(1))).to(cast("Vec[i64]", Vec[String](String("1"))))


def test_ne_byte_execution_and_structural_truth_after_union() -> None:
    class Compared(Expr):
        def __init__(self, value: i64) -> None: ...

    left, right = Compared(i64(1)), Compared(i64(2))
    different = ne(left).to(right)
    graph = EGraph(left, right)
    graph.check(ne(Bool(True)).to(Bool(False)), ne(i64(1)).to(i64(2)), different)
    graph.check_fail(ne(i64(1) + 1).to(i64(2)))
    assert bool(ne(i64(1) + 1).to(i64(2)))
    graph.check(ne(Vec[Compared](left)).to(Vec[Compared](right)))
    graph.register(union(left).with_(right))
    graph.check_fail(different)
    graph.check_fail(ne(Vec[Compared](left)).to(Vec[Compared](right)))
    assert bool(different)  # Python truth does not query the mutated e-graph.
    extracted = graph.extract(Vec[Compared](left))
    assert not bool(ne(extracted).to(copy(extracted)))
    graph.check_fail(ne(extracted).to(Vec[Compared](right)))


def test_normal_scalar_bytes_boundary() -> None:
    graph = EGraph()
    expression = i64(1) + 2
    assert graph.extract(expression).value == 3
    assert graph.extract(f64(1.0) + 2.0).value == 3.0
    graph.check(eq(expression).to(i64(3)))


def test_vec_public_generic_and_computed_table_bytes() -> None:
    @function
    def vector() -> Vec[i64]: ...

    @function
    def pair() -> Pair[i64, f64]: ...

    graph = EGraph()
    expression = Vec[i64](i64(1) + 2, i64(4))
    graph.register(set_(vector()).to(expression), set_(pair()).to(Pair(i64(5), f64(6.0))))
    extracted = graph.extract(vector())
    assert extracted.__egg_ref__.read().kind.value.value.field == "vec"
    assert tuple(child.value for child in extracted.value) == (3, 4)
    assert graph.extract(vector()[0]).value == 3
    graph.check(eq(vector()[1]).to(i64(4)))
    graph.check(eq(extracted).to(expression))
    assert graph.extract(Vec[i64].empty()).value == ()
    assert tuple(child.value for child in graph.extract(pair()).value) == (5, 6.0)
    with pytest.raises(ExprValueError):
        vector().value

    for end, expected in [(-(2**63), ()), (-1, ()), (0, ()), (1, (0,)), (5, (0, 1, 2, 3, 4))]:
        ranged = i64(end).range()
        with pytest.raises(ExprValueError):
            ranged.value
        value = graph.extract(ranged)
        assert tuple(child.value for child in value.value) == expected
        graph.check(eq(value).to(ranged))
    assert graph.extract(i64(5).range()[4]).value == 4
    with pytest.raises(bindings.EggSmolError):
        graph.extract(i64(5).range()[5])


def test_map_public_computed_vec_table_and_reingested_value_bytes() -> None:
    @function
    def entries() -> Vec[Pair[i64, f64]]: ...

    @function
    def mapping() -> Map[i64, f64]: ...

    graph = EGraph()
    pairs = Vec(Pair(i64(1), f64(2.0)), Pair(i64(1), f64(3.0)), Pair(i64(2), f64(4.0)))
    graph.register(set_(entries()).to(pairs))
    expression = Map.from_pairs(entries())
    assert len(get_callable_args(expression)) == 1
    with pytest.raises(ExprValueError):
        _ = expression.value
    graph.register(set_(mapping()).to(expression))
    result = graph.extract(mapping())
    assert result.__egg_ref__.read().kind.value.value.field == "map"
    assert {key.value: value.value for key, value in result.value.items()} == {1: 3.0, 2: 4.0}
    graph.check(eq(result).to(expression))
    assert graph.extract(mapping()[i64(1)]).value == 3.0
    assert graph.extract(result.insert(i64(3), f64(5.0))[i64(3)]).value == 5.0
    assert graph.extract(Map[i64, f64].empty()).value == {}
    assert graph.extract(Map[i64, f64].from_pairs(Vec[Pair[i64, f64]].empty())).value == {}
    with pytest.raises(ExprValueError):
        _ = mapping().value


def test_nominal_pair_coexists_with_ambient_family_through_bytes() -> None:
    class NominalPair(Expr, egg_sort="Pair"):
        @method(egg_fn="P")
        def __init__(self) -> None: ...

        @method(preserve=True)
        def tag(self):
            return "nominal Pair"

    expression = NominalPair()
    graph = EGraph()
    assert ("host_sort_family", "Pair") in graph._state.definitions
    graph.register(expression)
    result = graph.extract(expression)
    graph.check(eq(expression).to(result))
    assert result.tag() == "nominal Pair"
    assert result.__egg_ref__.read().kind.value.func == "P"
    result_sort = result.__egg_ref__.owner.ref("sorts", result.__egg_ref__.read().sort_id).read()
    assert result_sort.kind.field == "eq"
    assert result_sort.kind.value == "Pair"
    assert ("eq_sort", "Pair") in graph._state.definitions
    assert ("host_sort_family", "Pair") in graph._state.definitions


def test_bodyless_free_function_public_bytes() -> None:
    @function(egg_fn="score with spaces")
    def score(x: i64, offset: i64 = i64(2)) -> i64: ...

    expression = score(1)
    assert str(expression) == "score(1)"
    assert str(set_(expression).to(7)) == "set_(score(1)).to(i64(7))"
    graph = EGraph()
    with pytest.raises(bindings.EggSmolError):
        graph.extract(expression)
    graph.register(set_(score(offset=2, x=1)).to(i64(3) + 4))
    assert graph.extract(score(1)).value == 7
    graph.check(eq(score(1, 2)).to(i64(7)))
    with pytest.raises(ExprValueError):
        _ = expression.value
    with pytest.raises(bindings.EggSmolError, match="Illegal merge"):
        graph.register(set_(score(1)).to(i64(8)))


def test_nullary_and_f64_free_functions_public_bytes() -> None:
    @function
    def fixed() -> i64: ...

    @function
    def measure(x: f64) -> f64: ...

    assert str(fixed) == "fixed"
    assert str(fixed()) == "fixed()"
    graph = EGraph()
    graph.register(set_(fixed()).to(3), set_(measure(1.5)).to(f64(2.0) + 3.0))
    assert graph.extract(fixed()).value == 3
    assert graph.extract(measure(1.5)).value == 5.0
    graph.check(eq(measure(1.5)).to(f64(5.0)))


def test_free_constructor_function_recovers_local_output_hooks_through_bytes() -> None:
    def factory():
        class Num(Expr):
            @method(preserve=True)
            def tag(self):
                return "local Num"

        @function
        def leaf(value: i64 = i64(1)) -> Num:
            """A free constructor."""

        return leaf

    leaf = factory()
    expression = leaf(i64(1) + 2)
    graph = EGraph()
    result = graph.extract(expression)
    assert get_callable_fn(result) == leaf
    assert get_callable_args(result, leaf)[0].value == 3
    assert str(result) == "leaf(3)"
    graph.check(eq(result).to(leaf(3)))
    with pytest.raises(TypeError, match="table-backed"):
        set_(expression).to(result)
    del expression, leaf
    graph._state.destroy()
    assert result.tag() == "local Num"
    assert get_callable_fn(result).__doc__ == "A free constructor."


def test_constant_set_and_extract_public_bytes() -> None:
    current = constant("current", i64, egg_name="table with spaces")
    assert str(current) == "current"
    with pytest.raises(ExprValueError):
        _ = current.value
    graph = EGraph()
    with pytest.raises(bindings.EggSmolError):
        graph.extract(current)
    action = set_(current).to(i64(1) + 2)
    assert str(action) == "set_(current).to(i64(1) + 2)"
    graph.register(action)
    assert graph.extract(current).value == 3
    graph.check(eq(current).to(i64(3)))
    graph.register(set_(current).to(i64(3)))
    assert graph.extract(current).value == 3
    with pytest.raises(ExprValueError):
        _ = current.value
    with pytest.raises(bindings.EggSmolError, match="Illegal merge"):
        graph.register(set_(current).to(i64(4)))


@pytest.mark.parametrize("name", ["two words", "class", ""])
@pytest.mark.parametrize("egg_name", [None, "exact", ""])
def test_constant_string_names_through_bytes(name, egg_name):
    value = constant(name, i64, egg_name=egg_name)
    assert str(value) == name
    assert get_constant_name(value).name == name
    graph = EGraph()
    graph.register(set_(value).to(i64(1)))
    assert graph.extract(value).value == 1


def test_equality_constant_retains_name_and_local_hooks_through_bytes() -> None:
    def make():
        class Token(Expr):
            @method(preserve=True)
            def tag(self) -> str:
                return "local token"

        return constant("token", Token, egg_name="constant with spaces")

    token = make()
    graph = EGraph()
    graph.register(token)
    result = graph.extract(token)
    graph.check(eq(result).to(token))
    graph._state.destroy()
    assert str(result) == "token"
    assert get_constant_name(result).name == "token"
    assert result.tag() == token.tag() == "local token"


def test_user_constructor_public_bytes() -> None:
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

        @method(preserve=True)
        @property
        def value(self) -> int:
            args = get_callable_args(self, Num)
            assert args is not None
            return args[0].value

    expression = Num(i64(1) + 2)
    assert str(expression) == "Num(i64(1) + 2)"
    with pytest.raises(ExprValueError):
        _ = expression.value
    graph = EGraph()
    result = graph.extract(expression)
    assert isinstance(result, Num)
    assert get_callable_fn(result) == Num
    assert get_callable_args(result, Num)[0].value == result.value == 3
    assert str(result) == "Num(3)"
    graph.check(eq(result).to(Num(3)))
    assert graph.extract(Num(value=3)).value == 3


def test_user_constructor_exact_names_and_retained_local_hooks() -> None:
    def make():
        class Num(Expr, egg_sort="sort with spaces"):
            @method(egg_fn="constructor with spaces")
            def __init__(self, value: i64 = i64(7)) -> None: ...

            @method(preserve=True)
            def tag(self) -> str:
                return "retained local method"

        return Num()

    graph = EGraph()
    expression = make()
    result = graph.extract(expression)
    graph.check(eq(result).to(expression))
    graph._state.destroy()
    assert result.tag() == expression.tag() == "retained local method"
    assert get_callable_args(result)[0].value == 7


def test_scalar_equality_dispatch_respects_operand_sorts() -> None:
    assert (i64(1) == f64(1.0)) is False
    assert (f64(1.0) == i64(1)) is False
    equal = i64(1) == i64(1)
    different = f64(1.0) == f64(2.0)
    assert isinstance(equal, Fact)
    assert isinstance(different, Fact)
    assert bool(equal)
    assert not bool(different)


def test_automatic_threads_reports_actual_count() -> None:
    # Existing public behavior; unresolved until byte-level configuration exists.
    graph = EGraph(num_threads=0)
    assert graph.num_threads() >= 1


def test_automatic_threads_after_update_reports_actual_count() -> None:
    graph = EGraph(num_threads=1)
    graph.set_num_threads(0)
    assert graph.num_threads() >= 1


def test_thread_update_and_context_restore_actual_count() -> None:
    graph = EGraph(num_threads=2)
    assert graph.num_threads() == 2
    with graph:
        graph.set_num_threads(1)
        assert graph.num_threads() == 1
    assert graph.num_threads() == 2
    graph.set_num_threads(1)
    assert graph.num_threads() == 1


def test_ten_thousand_ordinary_calls_are_incremental(monkeypatch: pytest.MonkeyPatch) -> None:
    copied = 0
    visited = 0
    original_copy = _program._copy_record
    original_parts = _program._structural_parts

    def counted_copy(*args, **kwargs):
        nonlocal copied
        copied += 1
        return original_copy(*args, **kwargs)

    def counted_parts(*args, **kwargs):
        nonlocal visited
        visited += 1
        return original_parts(*args, **kwargs)

    monkeypatch.setattr(_program, "_copy_record", counted_copy)
    monkeypatch.setattr(_program, "_structural_parts", counted_parts)
    expression = i64(0)
    for _ in range(10_000):
        expression = expression + 1
        hash(expression)
    # Fixed per-call limits, not wall-clock timing: retained prefixes are not
    # rescanned during authoring/hash, independent of chain depth.
    assert copied < 50 * 10_000
    assert visited < 20 * 10_000
    packed = _program.pack([expression.__egg_ref__])
    assert len(packed.program.nodes) == 20_001


def test_pair_public_generic_bytes_roundtrip() -> None:
    from egglog import Pair, convert, f64  # noqa: PLC0415

    expression = Pair[i64, f64](left=i64(1) + 2, right=f64(4.5))
    assert expression.value[0].__egg_ref__.read().kind.field == "call"
    graph = EGraph(num_threads=1)
    graph.register(expression)
    extracted = graph.extract(expression)
    assert extracted.__egg_ref__.read().kind.value.value.field == "pair"
    assert [child.value for child in extracted.value] == [3, 4.5]
    assert graph.extract(extracted.left).value == 3
    assert graph.extract(extracted.right).value == 4.5
    graph.register(extracted)
    graph.check(eq(expression).to(extracted))
    nested = convert(((5, 6.5), 7), Pair[Pair[i64, f64], i64])
    result = graph.extract(nested)
    assert result.value[0].value[1].value == 6.5
    assert result.value[1].value == 7


def test_multiset_packaged_catalog_public_values_and_replay_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from egglog._catalog import builtin_catalog  # noqa: PLC0415

    definitions = builtin_catalog().definitions
    assert ("host_sort_family", "MultiSet") in definitions, (
        f"Packaged catalog is missing the MultiSet family ({len(definitions)} definitions); "
        "this is not a public-import or native-build failure"
    )
    assert {name for kind, name in definitions if kind == "callable" and name.startswith("egglog.core.multiset.")} == {
        f"egglog.core.multiset.{suffix}"
        for suffix in (
            "of", "single", "pick", "insert", "remove", "remove_swapped", "subtract", "subtract_swapped",
            "length", "contains", "not_contains", "contains_swapped", "not_contains_swapped", "intersection",
            "sum", "reset_counts", "pick_max", "count", "sum_multisets", "union_values",
        )
    }
    from egglog import MultiSet, convert  # noqa: PLC0415 -- only after the actual packaged-family discriminator

    def forbidden(*args, **kwargs):
        pytest.fail("MultiSet authoring or retained-value inspection accessed an engine")

    with monkeypatch.context() as inert:
        inert.setattr(bindings, "EGraph", forbidden)
        inert.setattr(bindings, "_ProtoEngine", forbidden)
        one = i64(1)
        symbolic = one + one
        inferred = MultiSet(symbolic, one, symbolic)
        explicit = MultiSet[i64](symbolic, one, symbolic)
        empty = MultiSet[i64]()
        assert expr_parts(inferred) == expr_parts(explicit)
        assert isinstance(explicit.value, list)
        assert explicit.value == list(explicit) == [symbolic, one, symbolic]
        assert len(explicit) == 3
        assert symbolic in explicit
        assert explicit.value[0].__egg_ref__ == explicit.value[2].__egg_ref__
        assert explicit.value[0].__egg_ref__.read().kind.field == "call"
        inspected = explicit.value
        inspected.clear()
        assert len(explicit.value) == 3
        assert empty.value == list(empty) == []
        assert convert((), MultiSet[i64]).value == []
        assert convert((1, 2, 1), MultiSet[i64]).value == [one, i64(2), one]
        assert str(inferred) == str(explicit)
        assert str(empty) == "MultiSet[i64]()"
        nested = MultiSet[MultiSet[i64]](explicit, explicit, empty)
        assert nested.value == [explicit, explicit, empty]

    requests: list[pb.RunProgramRequest] = []
    responses: list[pb.RunProgramResponse] = []
    destroyed: list[pb.DestroyEGraphRequest] = []

    class AuditedTransport:
        def __init__(self, transport: bindings._ProtoEngine) -> None:
            self.transport = transport

        def run(self, encoded: bytes) -> tuple[bytes, BaseException | None]:
            assert isinstance(encoded, bytes)
            index = len(requests)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned, host_error = self.transport.run(encoded)
            assert isinstance(returned, bytes)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            responses.append(pb.RunProgramResponse.from_binary(returned))
            assert host_error is None
            return returned, host_error

        def destroy(self, encoded: bytes) -> bytes:
            index = len(destroyed)
            (tmp_path / f"destroy-request-{index}.pb").write_bytes(encoded)
            destroyed.append(pb.DestroyEGraphRequest.from_binary(encoded))
            returned = self.transport.destroy(encoded)
            pb.DestroyEGraphResponse.from_binary(returned)
            (tmp_path / f"destroy-response-{index}.pb").write_bytes(returned)
            return returned

    graph = EGraph()
    monkeypatch.setattr(graph._state, "transport", AuditedTransport(graph._state.transport))
    result = graph.extract(explicit)
    retained_nested = graph.extract(nested)
    assert result.__egg_ref__.read().kind.field == "primitive_value"
    value = result.__egg_ref__.read().kind.value.value
    assert value.field == "multiset"
    assert len(value.value.items) == 3
    assert len(set(value.value.items)) == 2
    assert [item.__egg_ref__ for item in result.value] == [
        result.__egg_ref__.owner.ref("nodes", index) for index in value.value.items
    ]
    assert any(
        node.kind.field == "call" and node.kind.value.func == "egglog.core.multiset.of"
        and len(node.kind.value.args) == 3 and node.kind.value.args[0] == node.kind.value.args[2]
        for request in requests for node in request.program.nodes
    )
    original_handle = graph._state.handle
    graph._state.destroy()
    assert graph._state.handle is None
    assert [request.egraph_id for request in destroyed] == [original_handle]
    del graph
    gc.collect()
    run_count = len(requests)
    with monkeypatch.context() as inert:
        inert.setattr(bindings, "EGraph", forbidden)
        inert.setattr(bindings, "_ProtoEngine", forbidden)
        assert isinstance(result.value, list)
        assert sorted(child.value for child in result.value) == [1, 2, 2]
        assert list(result) == result.value
        assert sorted(len(child.value) for child in retained_nested.value) == [0, 3, 3]
        assert "MultiSet" in str(result)
        assert "MultiSet" in str(retained_nested)
        assert len(requests) == run_count
        assert len(destroyed) == 1

    fresh = EGraph()
    monkeypatch.setattr(fresh._state, "transport", AuditedTransport(fresh._state.transport))
    replayed = fresh.extract(result)
    fresh.check(eq(replayed).to(explicit))
    assert fresh.extract(replayed.count(i64(2))).value == 2
    assert fresh.extract(replayed.length()).value == 3
    assert fresh.extract(empty).value == []
    assert sorted(len(child.value) for child in fresh.extract(retained_nested).value) == [0, 3, 3]
    fresh_handle = fresh._state.handle
    fresh._state.destroy()
    assert fresh._state.handle is None
    assert len(destroyed) == 2
    assert destroyed[-1].egraph_id == fresh_handle
    assert len(requests) > run_count
    assert len(requests) == len(responses)
    assert all(response.error is None for response in responses)


def test_multiset_all_ordinary_views_and_partial_failures_public_bytes() -> None:
    from egglog import (  # noqa: PLC0415
        MultiSet,
        multiset_contains_swapped,
        multiset_not_contains_swapped,
        multiset_remove_swapped,
        multiset_subtract_swapped,
    )

    one, two, three = i64(1), i64(2), i64(3)
    xs = MultiSet(two, one, two)
    empty = MultiSet[i64]()
    graph = EGraph()
    for expression, expected in (
        (xs, [1, 2, 2]),
        (MultiSet[i64].single(two, 2), [2, 2]),
        (MultiSet[i64].single(two, 0), []),
        (xs.insert(three), [1, 2, 2, 3]),
        (xs.remove(one), [2, 2]),
        (multiset_remove_swapped(two, xs), [1, 2]),
        (xs - MultiSet(one), [2, 2]),
        (multiset_subtract_swapped(MultiSet(one), xs), [2, 2]),
        (xs & MultiSet(two, three), [2]),
        (xs + MultiSet(three), [1, 2, 2, 3]),
        (xs.reset_counts(), [1, 2]),
        (MultiSet.sum_multisets(MultiSet(xs, xs, empty)), [1, 1, 2, 2, 2, 2]),
    ):
        result = graph.extract(expression)
        assert result.__egg_ref__.read().kind.value.value.field == "multiset"
        assert sorted(child.value for child in result.value) == expected
    assert graph.extract(xs.pick()).value in (1, 2)
    assert graph.extract(xs.pick_max()).value == 2
    assert graph.extract(xs.length()).value == 3
    assert graph.extract(xs.count(two)).value == 2
    assert graph.extract(xs.count(three)).value == 0
    graph.check(xs.contains(one), xs.not_contains(three))
    graph.check(multiset_contains_swapped(one, xs), multiset_not_contains_swapped(three, xs))
    graph.check_fail(xs.contains(three))
    graph.check_fail(xs.not_contains(one))
    graph.check_fail(multiset_contains_swapped(three, xs))
    graph.check_fail(multiset_not_contains_swapped(one, xs))
    for expression in (
        MultiSet[i64].single(one, -1),
        xs.remove(three),
        multiset_remove_swapped(three, xs),
        xs - MultiSet(two, two, two),
        multiset_subtract_swapped(MultiSet(two, two, two), xs),
        empty.pick(),
        empty.pick_max(),
    ):
        with pytest.raises(bindings.EggSmolError):
            graph.extract(expression)
        assert graph.extract(xs.count(two)).value == 2


def test_multiset_union_values_requires_exact_eq_action_and_keeps_multiplicity(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from egglog import MultiSet  # noqa: PLC0415
    from egglog.type_constraint_solver import TypeConstraintError  # noqa: PLC0415

    class Atom(Expr):
        def __init__(self, value: i64) -> None: ...

    a, b = Atom(i64(1)), Atom(i64(2))
    xs = MultiSet(a, b, a)
    for invalid in (MultiSet(i64(1)), MultiSet(Vec(a)), MultiSet(MultiSet(a))):
        with pytest.raises(TypeConstraintError, match="requires an equality sort"):
            invalid.union_values()
    graph = EGraph(xs)
    graph.check_fail(eq(a).to(b))
    with pytest.raises(bindings.EggSmolError):
        graph.check(eq(xs.union_values()).to(a))
    graph.check_fail(eq(a).to(b))  # Rejected queries must not perform the write.
    graph.register(xs.union_values())
    graph.check(eq(a).to(b))
    assert graph.extract(xs.count(a)).value == 3
    assert graph.extract(xs.length()).value == 3
    assert len(graph.extract(xs).value) == 3
    with pytest.raises(bindings.EggSmolError):
        graph.register(MultiSet[Atom]().union_values())
    assert graph.extract(xs.length()).value == 3

    from protobuf import Oneof  # noqa: PLC0415

    @function
    def via_union(values: MultiSet[Atom]) -> Atom:
        return values.union_values()

    declaration = cast("RuntimeFunction", via_union).__egg_ref__
    primitive = declaration.read().kind
    assert primitive is not None
    assert primitive.field == "primitive"
    body = declaration.owner.ref("nodes", primitive.value.body).read()
    assert body.kind is not None
    assert body.kind.field == "call"
    assert body.kind.value.func == "egglog.core.multiset.union_values"

    @function
    def marker(key: i64) -> i64: ...

    c, d = Atom(i64(3)), Atom(i64(4))
    repeated = MultiSet(c, d, c)
    fresh = EGraph(repeated)
    assert fresh.function_values(marker) == {}
    before = set_(marker(i64(10))).to(i64(1))
    after = set_(marker(i64(20))).to(i64(2))
    requests: dict[str, pb.RunProgramRequest] = {}
    responses: dict[str, pb.RunProgramResponse] = {}

    class AuditedTransport:
        def __init__(self, transport: bindings._ProtoEngine, label: str) -> None:
            self.transport = transport
            self.label = label

        def run(self, encoded: bytes) -> tuple[bytes, BaseException | None]:
            assert self.label not in requests
            (tmp_path / f"{self.label}-request.pb").write_bytes(encoded)
            requests[self.label] = pb.RunProgramRequest.from_binary(encoded)
            returned, host_error = self.transport.run(encoded)
            assert host_error is None
            (tmp_path / f"{self.label}-response.pb").write_bytes(returned)
            responses[self.label] = pb.RunProgramResponse.from_binary(returned)
            return returned, host_error

    for label, expression in (("direct-query", repeated.union_values()), ("body-query", via_union(repeated))):
        builder = _program.Builder()
        fact = builder.import_ref(eq(expression).to(c).__egg_ref__)
        check = builder.add("commands", pb.Command(kind=Oneof("check", pb.Check(facts=[fact]))))
        command = builder.publish().ref("commands", check)
        with monkeypatch.context() as patch:
            patch.setattr(fresh._state, "transport", AuditedTransport(fresh._state.transport, label))
            with pytest.raises(bindings.EggSmolError):
                fresh._state.run_program([before.__egg_ref__, command, after.__egg_ref__])
        assert [item.kind.field for item in requests[label].program.commands] == ["action", "check", "action"]
        if label == "body-query":
            assert any(
                item.kind.field == "primitive" and item.kind.value.name == primitive.value.name
                for item in requests[label].program.declarations
            )
        error = responses[label].error
        assert error is not None
        assert error.code == pb.ErrorCode.INVALID_PROGRAM
        assert error.location is None
        assert fresh.function_values(marker) == {}
        fresh.check_fail(eq(c).to(d))

    with monkeypatch.context() as patch:
        patch.setattr(fresh._state, "transport", AuditedTransport(fresh._state.transport, "body-action"))
        fresh.register(via_union(repeated))
    assert responses["body-action"].error is None
    fresh.check(eq(c).to(d))
    assert fresh.extract(repeated.count(c)).value == 3
    assert fresh.extract(repeated.length()).value == 3
    assert len(fresh.extract(repeated).value) == 3

    with monkeypatch.context() as patch:
        patch.setattr(fresh._state, "transport", AuditedTransport(fresh._state.transport, "empty-prefix"))
        with pytest.raises(bindings.EggSmolError):
            fresh.register(before, MultiSet[Atom]().union_values(), after)
    assert [item.kind.field for item in requests["empty-prefix"].program.commands] == ["action"] * 3
    error = responses["empty-prefix"].error
    assert error is not None
    assert error.code == pb.ErrorCode.EVALUATION_FAILED
    assert error.location is not None
    assert error.location.path == [1]
    assert fresh.function_values(marker) == {marker(i64(10)): i64(1)}
    assert fresh.extract(repeated.count(c)).value == 3
    assert fresh.extract(repeated.length()).value == 3
    assert set(requests) == set(responses) == {"direct-query", "body-query", "body-action", "empty-prefix"}
