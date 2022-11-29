from __future__ import annotations
from datetime import timedelta
import inspect

from typing import (
    Iterable,
    ParamSpec,
    TypeVar,
    Callable,
    NewType,
    TYPE_CHECKING,
    Generic,
    Union,
    overload,
    Any,
)
import egg_smol.bindings as py

if TYPE_CHECKING:
    from egg_smol.bindings import _Literal


T = TypeVar("T")


def literal(lit: type[_Literal]) -> Callable[[T], T]:
    ...


builtins = Module("builtins")

# Nothing should ever be this value, as a way of disallowing methods
_Nothing = NewType("_Nothing", object)


class Sort:
    @builtins.function("!=")
    def __ne__(self: T, __o: T) -> Unit:  # type: ignore
        ...

    def __eq__(self, other: _Nothing) -> _Nothing:  # type: ignore
        raise NotImplementedError()


@builtins.sort("unit")
class Unit(Sort):
    # ???
    @literal(py.Unit)
    def __init__(self) -> None:
        ...


@builtins.sort("i64")
class i64(Sort):
    @literal(py.Int)
    def __init__(self, value: int):
        ...

    @builtins.function("+")
    def __add__(self, other: i64) -> i64:
        ...

    @builtins.function("-")
    def __sub__(self, other: i64) -> i64:
        ...

    @builtins.function("*")
    def __mul__(self, other: i64) -> i64:
        ...

    @builtins.function("/")
    def __truediv__(self, other: i64) -> i64:
        ...

    @builtins.function("%")
    def __mod__(self, other: i64) -> i64:
        ...

    @builtins.function("&")
    def __and__(self, other: i64) -> i64:
        ...

    @builtins.function("|")
    def __or__(self, other: i64) -> i64:
        ...

    @builtins.function("^")
    def __xor__(self, other: i64) -> i64:
        ...

    @builtins.function("<<")
    def __lshift__(self, other: i64) -> i64:
        ...

    @builtins.function(">>")
    def __rshift__(self, other: i64) -> i64:
        ...

    @builtins.function("not-64")
    def __invert__(self) -> i64:
        ...

    @builtins.function("<")
    def __lt__(self, other: i64) -> Unit:  # type: ignore
        ...

    @builtins.function(">")
    def __gt__(self, other: i64) -> Unit:
        ...

    @builtins.function("min")
    def min(self, other: i64) -> i64:
        ...

    @builtins.function("max")
    def max(self, other: i64) -> i64:
        ...


@builtins.sort("String")
class String(Sort):
    @literal(py.String)
    def __init__(self, value: str):
        ...


K = TypeVar("K", bound=Sort)
V = TypeVar("V", bound=Sort)


@builtins.sort("Map")
class Map(Sort, Generic[K, V]):
    @builtins.function("empty")
    @classmethod
    def empty(cls) -> Map[K, V]:
        ...

    @builtins.function("insert")
    def insert(self, key: K, value: V) -> Map[K, V]:
        ...

    @builtins.function("get")
    def __getitem__(self, key: K) -> V:
        ...

    @builtins.function("not-contains")
    def not_contains(self, key: K) -> Unit:
        ...

    @builtins.function("contains")
    def contains(self, key: K) -> Unit:
        ...

    @builtins.function("set-union")
    def __or__(self, __t: Map[K, V]) -> Map[K, V]:
        ...

    @builtins.function("set-diff")
    def __sub__(self, __t: Map[K, V]) -> Map[K, V]:
        ...

    @builtins.function("set-intersect")
    def __and__(self, __t: Map[K, V]) -> Map[K, V]:
        ...

    @builtins.function("map-remove")
    def map_remove(self, key: K) -> Map[K, V]:
        ...


@builtins.sort("Rational")
class Rational(Sort):
    @builtins.function("rational")
    def __init__(self, num: i64, den: i64):
        ...

    @builtins.function("+")
    def __add__(self, other: Rational) -> Rational:
        ...

    @builtins.function("-")
    def __sub__(self, other: Rational) -> Rational:
        ...

    @builtins.function("*")
    def __mul__(self, other: Rational) -> Rational:
        ...

    @builtins.function("/")
    def __truediv__(self, other: Rational) -> Rational:
        ...

    @builtins.function("min")
    def min(self, other: Rational) -> Rational:
        ...

    @builtins.function("max")
    def max(self, other: Rational) -> Rational:
        ...

    @builtins.function("neg")
    def __neg__(self) -> Rational:
        ...

    @builtins.function("abs")
    def __abs__(self) -> Rational:
        ...

    @builtins.function("floor")
    def floor(self) -> Rational:
        ...

    @builtins.function("ceil")
    def ceil(self) -> Rational:
        ...

    @builtins.function("round")
    def round(self) -> Rational:
        ...

    @builtins.function("pow")
    def __pow__(self, other: Rational) -> Rational:
        ...

    @builtins.function("log")
    def log(self) -> Rational:
        ...

    @builtins.function("sqrt")
    def sqrt(self) -> Rational:
        ...

    @builtins.function("cbrt")
    def cbrt(self) -> Rational:
        ...


FN = TypeVar("FN", bound=Callable[..., Sort | None])
S = TypeVar("S", bound=type[Sort])

SORT_TP = TypeVar("SORT_TP", bound=type[Sort])


class Module:
    def __init__(self, name: str):
        ...

    @overload
    def sort(self, name: str, /) -> Callable[[S], S]:
        ...

    @overload
    def sort(self, tp: S, /) -> S:
        ...

    def sort(self, name_or_tp: str | S, /) -> Callable[[S], S] | S:
        ...

    @overload
    def function(
        self,
        name: str = None,
        cost: int = None,
        merge: Callable[[Any, Any], Any] | None = None,
    ) -> Callable[[FN], FN]:
        ...

    @overload
    def function(self, fn: FN, /) -> FN:
        ...

    def function(self, name=None, cost=None, merge=None):
        ...

    def __call__(self, *definitions: Definition) -> None:
        ...

    def action(self, action: Action) -> None:
        ...

    def define(self, name: str, expr: K, cost: int = 0) -> K:
        ...

    def define_sort(self, name: str, sort: SORT_TP) -> SORT_TP:
        ...


class Rewrite:
    ...


class Rule:
    def __init__(self, body: Iterable[Fact] | Fact, head: Iterable[Action] | Action):
        ...


class EGraph(Module):
    def __init__(self, *modules: Module) -> None:
        self.egraph = py.EGraph()

    def register(self, module: Module):
        ...

    def check(self, fact: Fact) -> None:
        ...

    def run(self, limit: int) -> tuple[timedelta, timedelta, timedelta]:
        ...

    def extract(self, expr: K) -> K:
        ...

    def extract_variants(self, expr: K, variants: int) -> Iterable[K]:
        ...

    def __enter__(self) -> EGraph:
        """
        Push an egraph during the contet manager.
        """
        self.egraph.push()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Pop the egraph after the context manager.
        """
        self.egraph.pop()


class Eq:
    pass


class Action:
    pass


Definition = Union[Action, Rule, Rewrite]


class E(Generic[K]):
    def __init__(self, expr: K) -> None:
        ...

    def set(self, replacement: K) -> Action:
        ...

    def __eq__(self, other: K) -> Eq:  # type: ignore
        ...

    def rewrite(self, rhs: K, conditions: list[Fact] = []) -> Rewrite:
        ...


Fact = Union[Eq, Unit]


def vars_(names: str, sort: type[K]) -> list[K]:
    ...


def var(name: str, sort: type[K]) -> K:
    ...


# fib_mod = Module("fib")


# @fib_mod.function
# def fib(x: i64) -> i64:
#     ...


# fib_mod.action(E(fib(i64(0))).set(i64(1)))
# fib_mod.action(E(fib(i64(1))).set(i64(1)))


# f0, f1, x = vars_("f0 f1 x", i64)
# fib_mod.rule(
#     (E(f0) == fib(x), E(f1) == fib(x + i64(1))),
#     E(fib(x + i64(2))).set(f0 + f1),
# )


# egraph = EGraph()
# egraph.register(fib_mod)
# egraph.run(7)
# egraph.check(E(fib(i64(7))) == i64(21))

# fib_mod = Module("fib_demand")


# @fib_mod.sort
# class Expr(Sort):
#     ...


# @fib_mod.function
# def num(x: i64) -> Expr:
#     ...


# @fib_mod.function
# def add(x: Expr, y: Expr) -> Expr:
#     ...


# @fib_mod.function
# def fib(x: i64) -> Expr:
#     ...


# a, b = vars_("a, b", i64)
# f, x = var("f", Expr), var("x", i64)
# fib_mod(
#     E(add(num(a), num(b))).rewrite(num(a + b)),
#     Rule(
#         (E(f) == fib(x), x > i64(1)),
#         E(fib(x)).set(add(fib(x - i64(1)), fib(x - i64(2)))),
#     ),
#     E(fib(i64(0))).set(num(i64(1))),
#     E(fib(i64(1))).set(num(i64(1))),
# )


# egraph = EGraph(fib_mod)
# f7 = egraph.define("f7", fib(i64(7)))
# egraph.run(14)
# egraph.extract(f7)
# egraph.check(E(fib(i64(7))) == num(i64(13)))


lambda_mod = Module("lambda")


@lambda_mod.sort
class Value(Sort):
    ...


@lambda_mod.function
def true() -> Value:
    ...


@lambda_mod.function
def false() -> Value:
    ...


@lambda_mod.function
def num(x: i64) -> Value:
    ...


@lambda_mod.sort
class Var(Sort):
    @lambda_mod.function
    def __init__(self, name: str) -> None:
        ...

    @lambda_mod.function
    @classmethod
    def term(cls, term: Term) -> Var:
        ...


@lambda_mod.sort
class Term(Sort):
    @lambda_mod.function
    @classmethod
    def var(cls, var: Var) -> Term:
        ...

    @lambda_mod.function
    @classmethod
    def val(cls, val: Value) -> Term:
        ...

    @lambda_mod.function
    def __add__(self, other: Term) -> Term:
        ...

    @lambda_mod.function
    def __eq__(self, other: Term) -> Term:  # type: ignore
        ...

    @lambda_mod.function
    def __call__(self, arg: Term) -> Term:
        ...

    @lambda_mod.function
    @classmethod
    def lam(cls, var: Var, body: Term) -> Term:
        ...

    @lambda_mod.function
    def if_(self, then: Term, else_: Term) -> Term:
        ...


def lam(fn: Callable[[Term], Term]) -> Term:
    # The variable is the name of the first argument of the function.
    var_name = inspect.signature(fn).parameters.values().__iter__().__next__().name
    var = Var(var_name)
    return Term.lam(var, fn(Term.var(var)))


@lambda_mod.function
def let(var: Var, value: Term, body: Term) -> Term:
    ...


@lambda_mod.function
def fix(var: Var, body: Term) -> Term:
    ...


StringSet = Map[Var, i64]


@lambda_mod.function(merge=lambda old, new: old & new)
def freer(term: Term) -> StringSet:
    ...


e, e1, e2 = vars_("e e1 e2", Term)
v = var("v", Value)
va = var("va", Var)
fv1, fv2 = vars_("fv1 fv2", StringSet)
lambda_mod(
    Rule(
        E(e) == Term.val(v),
        E(freer(e)).set(StringSet.empty()),
    ),
    Rule(
        E(e) == Term.var(va),
        E(freer(e)).set(StringSet.empty().insert(va, i64(1))),
    ),
)
