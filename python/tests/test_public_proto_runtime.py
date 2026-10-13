"""Normal Python syntax over the generated-record runtime, without an engine."""

import gc
from copy import copy
from fractions import Fraction
from typing import ClassVar, cast

import pytest

from egglog import (
    Bool,
    BoolLike,
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
    set_,
    union,
)
from egglog.deconstruct import get_callable_args, get_callable_fn, get_constant_name, get_literal_value
from egglog.runtime import RuntimeExpr, RuntimeFunction


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
