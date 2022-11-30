from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    NewType,
    ParamSpec,
    TypeVar,
    Union,
    overload,
    Arg
)

import egg_smol.bindings as py

if TYPE_CHECKING:
    from egg_smol.bindings import _Literal


T = TypeVar("T")
S = TypeVar("S", bound="Sort")
S1 = TypeVar("S1", bound="Sort")
S2 = TypeVar("S2", bound="Sort")
ST = TypeVar("ST", bound=type["Sort"])
FN = TypeVar("FN", bound=Callable[..., "Sort" | None])

# How do we want to organize this data at runtime in Python?
# Do we want to end up wrapping builtin Python types eventually?
# We could either have the values be actual runtime values of the type (how we do it in Metadsl)
# Or make them all just instance of one type that acts like whatever type it needs to be.
# The latter is nicer in that we can keep seperate the runtime values their python types
# (which are really just needed for Mypy to look at)
# Let's try the latter "for fun"




@dataclass
class _PrettyPrinter:
    # Mapping of egg function name to a function which mapps the string args to the result
    functions: dict[str, Callable[[Iterable[str]], str]]
    sorts: dict[str, str]

    def __call__(self, expr: py._Expr) -> str:
        """
        Pretty print an expression.

        >>> pp = _PrettyPrinter()
        >>> pp.pretty_print_expr(py.Lit(py.Int(1)))
        'i64(1)'
        >>> pp.pretty_print_expr(py.Lit(py.

        """
        if isinstance(expr, py.Lit):
            return self._pretty_print_literal(expr.value)
        elif isinstance(expr, py.Var):
            return expr.name
        elif isinstance(expr, py.Call):
            return self._pretty_print_call(expr.name, expr.args)
        else:
            raise NotImplementedError

    def _pretty_print_literal(self, literal: _Literal) -> str:
        if isinstance(literal, py.Int):
            return f"i64({literal.value})"
        elif isinstance(literal, py.String):
            return f'String("{literal.value}")'
        elif isinstance(literal, py.Unit):
            return "Unit()"
        else:
            raise NotImplementedError
    
    def _pretty_print_call(self, name: str, args: list[py._Expr]) -> str:
        if name in self.functions:
            return self.functions[name]([self(arg) for arg in args])
        else:
            raise NotImplementedError


def test_pretty_print_lit():
    pp = _PrettyPrinter()
    assert pp(py.Lit(py.Int(1))) == "i64(1)"
    assert pp(py.Lit(py.Unit())) == "Unit()"
    assert pp(py.Lit(py.String("hello"))) == 'String("hello")'

def test_pretty_print_var():
    pp = _PrettyPrinter()
    assert pp(py.Var("x")) == "x"


def test_pretty_print_function():
    pp = _PrettyPrinter()
    pp.function['add'] = lambda args: f"({args[0]} + {args[1]})"
    assert pp(py.Call("add", [py.Lit(py.Int(1)), py.Lit(py.Int(2))])) == "i64(1) + i64(2)"


@dataclass
class E

@dataclass
class _Function:
    # Name of Python function
    name: str
    decl: py.FunctionDecl

    def __call__(self, *args):


# How would I teach a course that could train people to help me with all of this?

class Module:
    def __init__(self, name: str):
        self.name = name
        self.on_register: list[Callable[[py.EGraph], None]] = []
        # Mapping of sort name to the sort Python type
        self.sort_to_str: dict[str, str] = {}

    @overload
    def sort(self, name: str) -> Callable[[ST], ST]:
        ...

    @overload
    def sort(self, tp: ST, /) -> ST:
        ...

    def sort(self, name: str | ST) -> Callable[[ST], ST] | ST:
        if isinstance(name, str):

            def decorator(tp: ST) -> ST:
                return self.define_sort(name, tp)

            return decorator
        return self.define_sort(name.name, name_or_tp)

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

    def define(self, name: str, expr: S, cost: int = 0) -> S:
        ...

    def define_sort(self, name: str, sort: ST) -> ST:
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

    def extract(self, expr: S) -> S:
        ...

    def extract_variants(self, expr: S, variants: int) -> Iterable[S]:
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


class E(Generic[S1]):
    def __init__(self, expr: S1) -> None:
        ...

    def set(self, replacement: S1) -> Action:
        ...

    def __eq__(self, other: S1) -> Eq:  # type: ignore
        ...

    def rewrite(self, rhs: S1, conditions: list[Fact] = []) -> Rewrite:
        ...


def vars_(names: str, sort: type[S1]) -> list[S1]:
    ...


def var(name: str, sort: type[S1]) -> S1:
    ...


def literal(lit: type[_Literal]) -> Callable[[T], T]:
    ...


@builtins.sort("unit")
class Unit(Sort):
    # ???
    @literal(py.Unit)
    def __init__(self) -> None:
        ...


Fact = Union[Eq, Unit]


@builtins.sort
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


@builtins.sort
class String(Sort):
    @literal(py.String)
    def __init__(self, value: str):
        ...


@builtins.sort
class Map(Sort, Generic[S1, S2]):
    @builtins.function("empty")
    @classmethod
    def empty(cls) -> Map[S1, S2]:
        ...

    @builtins.function("insert")
    def insert(self, key: S1, value: S2) -> Map[S1, S2]:
        ...

    @builtins.function("get")
    def __getitem__(self, key: S1) -> S2:
        ...

    @builtins.function("not-contains")
    def not_contains(self, key: S1) -> Unit:
        ...

    @builtins.function("contains")
    def contains(self, key: S1) -> Unit:
        ...

    @builtins.function("set-union")
    def __or__(self, __t: Map[S1, S2]) -> Map[S1, S2]:
        ...

    @builtins.function("set-diff")
    def __sub__(self, __t: Map[S1, S2]) -> Map[S1, S2]:
        ...

    @builtins.function("set-intersect")
    def __and__(self, __t: Map[S1, S2]) -> Map[S1, S2]:
        ...

    @builtins.function("map-remove")
    def map_remove(self, key: S1) -> Map[S1, S2]:
        ...


@builtins.sort
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


def test_fib():
    egraph = EGraph()

    @egraph.function
    def fib(x: i64) -> i64:
        ...

    f0, f1, x = vars_("f0 f1 x", i64)
    egraph(
        E(fib(i64(0))).set(i64(1)),
        E(fib(i64(1))).set(i64(1)),
        Rule(
            (E(f0) == fib(x), E(f1) == fib(x + i64(1))),
            E(fib(x + i64(2))).set(f0 + f1),
        ),
    )
    egraph.run(7)
    egraph.check(E(fib(i64(7))) == i64(21))


def test_fib_demand():
    mod = Module("fib_demand")

    @mod.sort
    class Expr(Sort):
        ...

    @mod.function
    def num(x: i64) -> Expr:
        ...

    @mod.function
    def add(x: Expr, y: Expr) -> Expr:
        ...

    @mod.function
    def fib(x: i64) -> Expr:
        ...

    a, b = vars_("a, b", i64)
    f, x = var("f", Expr), var("x", i64)
    mod(
        E(add(num(a), num(b))).rewrite(num(a + b)),
        Rule(
            (E(f) == fib(x), x > i64(1)),
            E(fib(x)).set(add(fib(x - i64(1)), fib(x - i64(2)))),
        ),
        E(fib(i64(0))).set(num(i64(1))),
        E(fib(i64(1))).set(num(i64(1))),
    )

    egraph = EGraph(mod)
    f7 = egraph.define("f7", fib(i64(7)))
    egraph.run(14)
    egraph.extract(f7)
    egraph.check(E(fib(i64(7))) == num(i64(13)))


# lambda_mod = Module("lambda")


# @lambda_mod.sort
# class Value(Sort):
#     ...


# @lambda_mod.function
# def true() -> Value:
#     ...


# @lambda_mod.function
# def false() -> Value:
#     ...


# @lambda_mod.function
# def num(x: i64) -> Value:
#     ...


# @lambda_mod.sort
# class Var(Sort):
#     @lambda_mod.function
#     def __init__(self, name: str) -> None:
#         ...

#     @lambda_mod.function
#     @classmethod
#     def term(cls, term: Term) -> Var:
#         ...


# @lambda_mod.sort
# class Term(Sort):
#     @lambda_mod.function
#     @classmethod
#     def var(cls, var: Var) -> Term:
#         ...

#     @lambda_mod.function
#     @classmethod
#     def val(cls, val: Value) -> Term:
#         ...

#     @lambda_mod.function
#     def __add__(self, other: Term) -> Term:
#         ...

#     @lambda_mod.function
#     def __eq__(self, other: Term) -> Term:  # type: ignore
#         ...

#     @lambda_mod.function
#     def __call__(self, arg: Term) -> Term:
#         ...

#     @lambda_mod.function
#     @classmethod
#     def lam(cls, var: Var, body: Term) -> Term:
#         ...

#     @lambda_mod.function
#     def if_(self, then: Term, else_: Term) -> Term:
#         ...


# def lam(fn: Callable[[Term], Term]) -> Term:
#     # The variable is the name of the first argument of the function.
#     var_name = inspect.signature(fn).parameters.values().__iter__().__next__().name
#     var = Var(var_name)
#     return Term.lam(var, fn(Term.var(var)))


# @lambda_mod.function
# def let(var: Var, value: Term, body: Term) -> Term:
#     ...


# @lambda_mod.function
# def fix(var: Var, body: Term) -> Term:
#     ...


# StringSet = Map[Var, i64]


# @lambda_mod.function(merge=lambda old, new: old & new)
# def freer(term: Term) -> StringSet:
#     ...


# e, e1, e2 = vars_("e e1 e2", Term)
# v = var("v", Value)
# va = var("va", Var)
# fv1, fv2 = vars_("fv1 fv2", StringSet)
# lambda_mod(
#     Rule(
#         E(e) == Term.val(v),
#         E(freer(e)).set(StringSet.empty()),
#     ),
#     Rule(
#         E(e) == Term.var(va),
#         E(freer(e)).set(StringSet.empty().insert(va, i64(1))),
#     ),
# )
