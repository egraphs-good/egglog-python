# mypy: allow-empty-bodies
# Ellipsis bodies are DSL declarations or annotation-introspection fixtures.
import pytest

from egglog import *
from egglog import conversion
from egglog.conversion import with_type_args
from egglog.type_constraint_solver import TypeConstraintError

# TODO: Revert global conversion state after each test w/ fixture
# Exact comparison expectations below observe RuntimeClass presentations of authored Expr types.


@pytest.mark.skip(reason="We dont support looking up by metaclass anymore")
def test_conversion_custom_metaclass():
    class MyMeta(type):
        pass

    class MyType(metaclass=MyMeta):
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    converter(MyMeta, MyTypeExpr, lambda x: MyTypeExpr())
    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion():
    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())

    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


def test_conversion_transitive_forward():
    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    class MyTypeExpr2(Expr):
        def __init__(self) -> None: ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())

    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())


def test_conversion_transitive_backward():
    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    class MyTypeExpr2(Expr):
        def __init__(self) -> None: ...

    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())
    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())


def test_conversion_transitive_cycle():
    class MyType:
        pass

    class MyTypeExpr(Expr):
        def __init__(self) -> None: ...

    class MyTypeExpr2(Expr):
        def __init__(self) -> None: ...

    converter(MyType, MyTypeExpr, lambda x: MyTypeExpr())
    converter(MyTypeExpr, MyTypeExpr2, lambda x: MyTypeExpr2())
    converter(MyTypeExpr2, MyTypeExpr, lambda x: MyTypeExpr())

    assert expr_parts(convert(MyType(), MyTypeExpr2)) == expr_parts(MyTypeExpr2())
    assert expr_parts(convert(MyType(), MyTypeExpr)) == expr_parts(MyTypeExpr())


@pytest.mark.parametrize("generic_first", [False, True])
def test_generic_family_composition_keeps_concrete_intermediate_context(generic_first):
    class Source:
        def __init__(self, values) -> None:
            self.values = values

    class Derived(Source):
        pass

    class IntBox(Expr):
        def __init__(self, values: Vec[i64]) -> None: ...

    class FloatBox(Expr):
        def __init__(self, values: Vec[f64]) -> None: ...

    calls = []

    def to_vector(source):
        calls.append(get_type_args())
        (element,) = get_type_args()
        return convert(source.values, Vec[element])  # type: ignore[valid-type]  # RuntimeClass argument, not a static alias.

    def to_int_box(value):
        assert get_type_args() == ()
        return IntBox(value)

    if generic_first:
        converter(Source, Vec, to_vector)
    converter(Vec[i64], IntBox, to_int_box)
    converter(Vec[f64], FloatBox, FloatBox)
    if not generic_first:
        converter(Source, Vec, to_vector)
    with with_type_args((String,)):  # type: ignore[arg-type]  # Authored Expr classes have RuntimeClass presentations.
        assert expr_parts(convert(Derived((1, 2)), IntBox)) == expr_parts(IntBox(Vec[i64](1, 2)))  # type: ignore[arg-type]
        assert expr_parts(convert(Derived((1.5, 2.5)), FloatBox)) == expr_parts(FloatBox(Vec[f64](1.5, 2.5)))  # type: ignore[arg-type]
        assert get_type_args() == (String,)  # type: ignore[comparison-overlap]
    assert calls == [(i64,), (f64,)]  # type: ignore[comparison-overlap]
    assert get_type_args() == ()
    converter(IntBox, Vec[i64], lambda _: Vec[i64].empty(), cost=10)
    assert expr_parts(convert(Derived((1, 2)), IntBox)) == expr_parts(IntBox(Vec[i64](1, 2)))  # type: ignore[arg-type]
    assert calls == [(i64,), (f64,), (i64,)]  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("direct_first", [False, True])
def test_generic_family_composition_preserves_cheaper_direct_conversion(direct_first):
    class Source:
        pass

    class Box(Expr):
        def __init__(self, values: Vec[i64]) -> None: ...

    calls = []

    def indirect(_):
        calls.append("indirect")
        return Vec[get_type_args()[0]].empty()  # type: ignore[misc]  # The conversion context computes the runtime type.

    direct = Box(Vec[i64](9))  # type: ignore[arg-type]  # Retain runtime integer-to-i64 coercion.
    if direct_first:
        converter(Source, Box, lambda _: direct, cost=0)
    converter(Source, Vec, indirect)
    converter(Vec[i64], Box, Box)
    if not direct_first:
        converter(Source, Box, lambda _: direct, cost=0)
    assert convert(Source(), Box) is direct
    assert not calls


def test_generic_family_composition_restores_context_after_original_error():
    class Source:
        pass

    class Box(Expr):
        def __init__(self, values: Vec[i64]) -> None: ...

    marker = ValueError("intermediate converter failed")
    calls = []

    def fail(_):
        calls.append(get_type_args())
        raise marker

    converter(Source, Vec, fail)
    converter(Vec[i64], Box, Box)
    with with_type_args((String,)):  # type: ignore[arg-type]  # Authored Expr classes have RuntimeClass presentations.
        with pytest.raises(ValueError, match="intermediate converter failed") as caught:
            convert(Source(), Box)
        assert caught.value is marker
        assert get_type_args() == (String,)  # type: ignore[comparison-overlap]
    assert get_type_args() == ()
    assert calls == [(i64,)]  # type: ignore[comparison-overlap]


def test_generic_family_composition_does_not_match_another_family():
    class Source:
        pass

    class Box(Expr):
        def __init__(self, value: Pair[i64, f64]) -> None: ...

    converter(Source, Vec, lambda _: pytest.fail("Unrelated family converter ran"))
    converter(Pair[i64, f64], Box, Box)
    with pytest.raises(ConvertError, match="Cannot convert"):
        convert(Source(), Box)


@pytest.mark.parametrize("generic_first", [False, True])
def test_generic_family_lookup_uses_cost_then_exact_tie_break(generic_first):
    class Source:
        pass

    calls = []

    def exact(_):
        calls.append("exact")
        return Vec[i64](99)  # type: ignore[arg-type]

    def generic(_):
        calls.append("generic")
        return Vec[get_type_args()[0]](1)  # type: ignore[misc, operator]  # Call the context-computed RuntimeClass.

    if generic_first:
        converter(Source, Vec, generic, cost=1)
    converter(Source, Vec[i64], exact, cost=10)
    if not generic_first:
        converter(Source, Vec, generic, cost=1)
    assert expr_parts(convert(Source(), Vec[i64])) == expr_parts(Vec[i64](1))  # type: ignore[arg-type]
    assert calls == ["generic"]
    calls.clear()
    converter(Source, Vec[i64], exact, cost=1)
    assert expr_parts(convert(Source(), Vec[i64])) == expr_parts(Vec[i64](99))  # type: ignore[arg-type]
    assert calls == ["exact"]


def test_generic_family_lookup_keeps_source_mro_preference():
    class Source:
        pass

    class Derived(Source):
        pass

    converter(Source, Vec[i64], lambda _: pytest.fail("More-expensive exact converter ran"), cost=10)
    converter(Source, Vec, lambda _: pytest.fail("Less-specific family converter ran"), cost=0)
    converter(Derived, Vec, lambda _: Vec[get_type_args()[0]](1), cost=2)  # type: ignore[misc, operator]  # RuntimeClass target.
    assert expr_parts(convert(Derived(), Vec[i64])) == expr_parts(Vec[i64](1))  # type: ignore[arg-type]


def test_numeric_winner_error_is_not_retried_and_restores_context():
    class Left(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: "Left") -> "Left": ...

    class Right(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: "Right") -> "Right": ...

    marker = ValueError("winning converter failed")
    calls = []

    def fail(_):
        calls.append(get_type_args())
        raise marker

    converter(Left, Right, fail, cost=1)
    converter(Right, Left, lambda _: pytest.fail("Fallback converter ran"), cost=100)
    left, right = Left(), Right()
    with with_type_args((String,)):  # type: ignore[arg-type]  # Authored Expr classes have RuntimeClass presentations.
        with pytest.raises(ValueError, match="winning converter failed") as caught:
            left - right  # type: ignore[operator]
        assert caught.value is marker
        assert get_type_args() == (String,)  # type: ignore[comparison-overlap]
    assert calls == [()]
    assert get_type_args() == ()


@pytest.mark.parametrize("raises", [False, True])
def test_numeric_positive_original_winner_preserves_ties_and_context(monkeypatch, raises):
    monkeypatch.setattr(conversion, "CONVERSIONS", {})

    class Source:
        pass

    class Operand(Expr):
        def __init__(self) -> None: ...

    class Left(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: Operand) -> "Left": ...

    prepared = []

    class Alternative(Expr):
        def __sub__(self, other: "Alternative") -> "Alternative":
            prepared.append("alternative")
            pytest.fail("Losing member was prepared")

    marker = ValueError("original winning converter failed")
    calls = []

    def from_source(_):
        calls.append(get_type_args())
        if raises:
            raise marker
        return Operand()

    converter(Source, Operand, from_source)
    converter(Left, Alternative, lambda _: pytest.fail("Losing receiver converter ran"), cost=0)
    converter(Source, Alternative, lambda _: pytest.fail("Losing RHS converter ran"))
    left = Left()
    with with_type_args((String,)):  # type: ignore[arg-type]  # Authored Expr classes have RuntimeClass presentations.
        if raises:
            with pytest.raises(ValueError, match="original winning converter failed") as caught:
                left - Source()  # type: ignore[operator]
            assert caught.value is marker
        else:
            expression = left - Source()  # type: ignore[operator]  # Registered conversion supplies the authored operand.
            assert expr_parts(expression) == expr_parts(Left.__sub__(left, Operand()))
        assert get_type_args() == (String,)  # type: ignore[comparison-overlap]
    assert calls == [()]
    assert get_type_args() == ()
    assert prepared == []
    assert Alternative.__egg_pending__ is not None  # type: ignore[attr-defined]  # Authored RuntimeClass metadata.


@pytest.mark.parametrize("reentrant", [False, True])
def test_numeric_cutoff_preserves_negative_left_costs(monkeypatch, reentrant):
    monkeypatch.setattr(conversion, "CONVERSIONS", {})

    class Source:
        pass

    class Operand(Expr):
        def __init__(self) -> None: ...

    class Left(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: Operand) -> "Left": ...

    class Alternative(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: "Alternative") -> "Alternative": ...

    calls = []

    def from_left(_):
        calls.append("left")
        return Alternative()

    def from_right(_):
        calls.append("right")
        return Alternative()

    converter(Left, Alternative, from_left, cost=0 if reentrant else -5)
    converter(Source, Operand, lambda _: pytest.fail("Losing original converter ran"))
    converter(Source, Alternative, from_right, cost=2)
    updates = []
    armed = False

    class MutatingMeta(type):
        __hash__ = type.__hash__

        def __eq__(cls, other) -> bool:
            nonlocal armed
            if armed and other is Source:
                armed = False
                converter(Left, Alternative, from_left, cost=-5)
                updates.append("lowered")
            return type.__eq__(cls, other)

    class Trigger(metaclass=MutatingMeta):
        pass

    converter(Trigger, Operand, lambda _: pytest.fail("Unrelated converter ran"))
    armed = reentrant
    expression = Left() - Source()  # type: ignore[operator]  # Conversion dispatch selects an Alternative receiver.
    assert expr_parts(expression) == expr_parts(Alternative.__sub__(Alternative(), Alternative()))
    assert calls == ["left", "right"]
    assert updates == (["lowered"] if reentrant else [])


@pytest.mark.parametrize("custom_rhs", [False, True])
def test_numeric_cutoff_opts_out_of_custom_integer_costs(monkeypatch, custom_rhs):
    monkeypatch.setattr(conversion, "CONVERSIONS", {})

    class Source:
        pass

    class Unrelated:
        pass

    class Operand(Expr):
        def __init__(self) -> None: ...

    class Left(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: Operand) -> "Left": ...

    class Alternative(Expr):
        def __sub__(self, other: "Alternative") -> "Alternative": ...

    class Cost(int):
        def __ge__(self, other) -> bool:
            pytest.fail("Custom conversion cost was compared")

    calls = []

    def from_source(_):
        calls.append("original")
        return Operand()

    converter(Source, Operand, from_source, cost=Cost(1) if custom_rhs else 1)
    converter(
        Unrelated, Alternative, lambda _: pytest.fail("Unrelated converter ran"), cost=1 if custom_rhs else Cost(1)
    )
    left = Left()
    expression = left - Source()  # type: ignore[operator]  # Registered conversion supplies the authored operand.
    assert expr_parts(expression) == expr_parts(Left.__sub__(left, Operand()))
    assert calls == ["original"]


def test_numeric_generic_owner_preserves_element_context():
    class Element(Expr):
        def __init__(self, value: i64) -> None: ...

    left = Vec[Element](Element(1))  # type: ignore[arg-type]
    with with_type_args((String,)):  # type: ignore[arg-type]  # Authored Expr classes have RuntimeClass presentations.
        expression = left | [Element(2)]  # type: ignore[arg-type, operator]  # Convert the list into the generic Vec operand.
        expected = Vec[Element].__or__(left, Vec[Element](Element(2)))  # type: ignore[arg-type]
        assert expr_parts(expression) == expr_parts(expected)
        assert get_type_args() == (String,)  # type: ignore[comparison-overlap]
    with pytest.raises(TypeConstraintError, match="requires an equality sort"):
        Vec[i64](1) | Vec[i64](2)  # type: ignore[arg-type]


def test_numeric_conversion_can_select_a_third_closed_receiver():
    class Left(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: "Left") -> "Left": ...

    class Right(Expr):
        def __init__(self) -> None: ...
        def __sub__(self, other: "Right") -> "Right": ...

    class Common(Expr):
        def __init__(self, label: StringLike) -> None: ...
        def __sub__(self, other: "Common") -> "Common": ...

    calls = []

    def from_left(_):
        calls.append("left")
        return Common("left")

    def from_right(_):
        calls.append("right")
        return Common("right")

    converter(Left, Right, lambda _: pytest.fail("Expensive left conversion ran"), cost=10)
    converter(Right, Left, lambda _: pytest.fail("Expensive right conversion ran"), cost=100)
    converter(Left, Common, from_left)
    converter(Right, Common, from_right)
    expression = Left() - Right()  # type: ignore[operator]  # Conversion dispatch selects the third Common receiver.
    assert expr_parts(expression) == expr_parts(Common.__sub__(Common("left"), Common("right")))
    assert calls == ["left", "right"]


def test_convert_to_generic(monkeypatch):
    """A concrete family target uses constructor inference and rejects incompatible types."""
    monkeypatch.setattr(conversion, "CONVERSIONS", conversion.CONVERSIONS.copy())

    converter(i64, Vec[i64], lambda value: Vec(value))
    assert expr_parts(convert(10, Vec[i64])) == expr_parts(Vec(i64(10)))

    with pytest.raises(ConvertError):
        convert(10, Vec[String])

    with pytest.raises(ConvertError):
        convert("hi", Vec[i64])


def test_convert_to_unbound_generic(monkeypatch):
    """An unbound family converter receives target arguments not inferable from its input."""
    monkeypatch.setattr(conversion, "CONVERSIONS", conversion.CONVERSIONS.copy())
    contexts: list[tuple[object, ...]] = []

    def to_pair(value):
        contexts.append(get_type_args())
        empty_vector, scalar = get_type_args()
        return Pair[empty_vector, scalar](empty_vector.empty(), value)  # type: ignore[valid-type, operator]

    converter(i64, Pair, to_pair)
    assert expr_parts(convert(10, Pair[Vec[String], i64])) == expr_parts(Pair(Vec[String].empty(), i64(10)))
    assert contexts == [(Vec[String], i64)]
    assert get_type_args() == ()


def test_convert_generic_transitive(monkeypatch):
    """A -> Vec and Vec[C] -> D compose while supplying the intermediate argument C."""
    monkeypatch.setattr(conversion, "CONVERSIONS", conversion.CONVERSIONS.copy())

    class A(Expr):
        def __init__(self) -> None: ...

    class C(Expr): ...

    class D(Expr):
        def __init__(self) -> None: ...

    contexts: list[tuple[object, ...]] = []

    def to_vector(_):
        contexts.append(get_type_args())
        return Vec[get_type_args()[0]].empty()  # type: ignore[misc]

    converter(A, Vec, to_vector)
    converter(Vec[C], D, lambda _: D())

    assert expr_parts(convert(A(), D)) == expr_parts(D())
    assert contexts == [(C,)]
    assert get_type_args() == ()
