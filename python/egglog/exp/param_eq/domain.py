"""
Core expression-domain helpers for `param_eq`.

This module defines the paper-language `Num` expression type together with four
derived operations over it:

1. parsing surface syntax into `Num`
2. rendering `Num` back to surface syntax
3. lowering binary syntax into a container representation
4. computing extraction costs for both binary and container forms


Binary form
===========

`Num` expressions in binary form use these nodes:

- constants: `Num(f64)`
- variables: `Num.var(String)`
- binary operators: `+`, `-`, `*`, `/`, `**`
- unary operators/functions: unary `-`, `abs`, `exp`, `log`, `sqrt`

`Num.__neg__()` is defined as `Num(-1.0) * self`; there is no distinct unary
minus node in the egg representation.

Container form
==============

Container form represents polynomial-like structure as:

- `ContainerMonomial = Map[Num, BigRat]`
- `ContainerPolynomial = Map[ContainerMonomial, f64]`
- `polynomial(container)` wraps a `ContainerPolynomial` back into `Num`

A monomial maps each factor `Num` term to its exponent. A polynomial maps each
monomial to its floating-point coefficient.

Not every `Num` subtree becomes a container map. Non-polynomial operators such
as `exp`, `log`, and `abs` remain explicit `Num` calls whose children may
themselves be lowered to containers and then wrapped back with `polynomial(...)`
as needed.

Parsing
=======

`parse_expression(source)` parses a restricted Python-like syntax into binary
`Num` form.

Accepted syntax:

- variable names, parsed as `Num.var(name)`
- integer and float literals, parsed as `Num(float(value))`
- binary `+`, `-`, `*`, `/`, `**`
- unary negation
- `exp(x)`, `log(x)`, `sqrt(x)`, `abs(x)`
- `Exp(x)`, `Log(x)`, `Sqrt(x)`, `Abs(x)` as aliases
- `plog(x)`, parsed as `log(abs(x))`
- `square(x)`, parsed as `x ** 2`
- `cube(x)`, parsed as `x ** 3`

Normalization performed by the parser:

- every `^` in the input string is replaced with `**` before parsing
- negative numeric literals such as `-2` are parsed as `Num(-2.0)`
- non-literal unary minus is parsed as `Num(-1.0) * expr`
- numeric literals are converted eagerly to `Num(float(...))`

Parsing does not accept arbitrary Python. Unsupported call names, operators, or
AST nodes raise an exception.

Parsing also fixes some structure:

- the exponent of `**` must parse as a literal `Num(...)`; this is asserted
- `square` and `cube` always become powers with literal exponents

Rendering
=========

`render_num(num)` produces a Python-like string for reports and snapshots.

Rendering rules:

- if the top-level node is `polynomial(Map(...))`, it is first decoded back to
  binary form with `containers_to_binary`
- binary operators render with explicit parentheses
- `exp`, `log`, `sqrt`, and `abs` render as ordinary function calls
- integer-valued floats render with a `.0` suffix
- zero always renders as `0.0`
- non-integer floats render with `repr(float_value)`

After local rendering, the string is round-tripped through `ast.parse` and
`ast.unparse` to normalize redundant whitespace and parentheses.

For binary expressions, parsing a rendered string is intended to reproduce the
same extracted binary `Num` tree. Rendering is therefore a canonicalization step
on strings, but a fixed point on extracted binary expressions:

- `render_num(parse_expression(s))` need not equal `s` textually
- `parse_expression(render_num(binary_expr))` is intended to equal the extracted
  binary expression that was rendered

Rendering a container expression first decodes it to binary, so reparsing a
rendered container expression yields the decoded binary form rather than the
original `polynomial(...)` container node.

Container lowering
==================

`binary_to_containers(expr)` lowers a binary `Num` tree into the container
representation and then converts the result back to `Num`.

Lowering is structural and eager. It performs some canonicalization while
lowering:

- constants and variables stay as `Num`
- `x + y` lowers to polynomial merge; coefficients of identical monomials are
  added immediately
- `x - y` lowers like addition, but the coefficients of the lowered right-hand
  polynomial are negated
- `x * y` usually lowers to monomial merge; exponents of identical factors are
  added immediately
- `x / y` lowers by negating exponents in the lowered denominator monomial
- `1 / y` is represented as a monomial with only negative exponents
- `x ** c` with literal exponent `c` lowers to a monomial containing `x` with
  exponent `BigRat.from_f64(c)`; exponent `1` is elided
- `sqrt(x)` lowers to exponent `1/2`
- `exp(x)`, `log(x)`, and `abs(x)` remain explicit function nodes over the
  lowered child, wrapped back to `Num` with `polynomial(...)` if needed

This means binary-to-container lowering is not a pure re-encoding step. It can
merge repeated terms and repeated factors as soon as they appear.

Container decoding
==================

`containers_to_binary(num)` reverses container lowering by decoding every
`polynomial(...)` node back into binary form.

Monomial decoding:

- positive exponents go to the numerator
- negative exponents go to the denominator
- exponent `1/2` decodes as `sqrt(term)`
- exponent `1` decodes as the bare term
- any other exponent decodes as `term ** abs(exp)` using a float exponent
- numerator factors are multiplied left-associatively, starting from the
  monomial coefficient
- if there are denominator factors, the decoded numerator is divided by their
  left-associated product
- an empty monomial decodes to its coefficient as `Num(coef)`

Polynomial decoding:

- the first monomial becomes the initial term directly
- later positive coefficients are added
- later negative coefficients are turned into subtraction of the corresponding
  positive-magnitude monomial term
- an empty polynomial decodes to `Num(0.0)`

`containers_to_binary(binary_to_containers(expr))` is intended to preserve
meaning, but not necessarily exact tree shape. The container form merges like
terms and has its own canonicalization rules.

No exact fixed-point guarantee is made for raw container syntax after repeated
decode/re-lower cycles. Different but semantically related container forms can
arise when sign placement and multiplication/division associativity are
reshuffled by decoding. The tested invariant is weaker:

- decoding an extracted container gives a binary expression that is still in the
  supported binary language
- re-lowering that decoded binary expression gives a valid container expression
- the cost invariants below hold for the decoded binary and re-lowered
  container forms

Validation helpers
==================

- `validate_is_binary(expr)` rejects any use of `polynomial(...)`
- `validate_is_containers(expr)` rejects explicit binary `+`, `-`, `*`, `/`,
  and `sqrt`, and also rejects `**` when the exponent is a literal `Num`

These functions are structural checks for tests and debugging; they do not
prove semantic equivalence.

Cost model
==========

`ParamCost` is lexicographic:

- `floats`: count of non-integer floating-point constants, used as the fitted
  parameter count
- `ops_and_ints`: count of operators plus integer-valued literals and variable
  names

Therefore it always will choose terms with less floats
and break ties by choosing terms with fewer total operators and integer literals.

`node_count` is `floats + ops_and_ints`.

`param_cost_model` for binary form:

- `f64` leaves:
  - integer-valued floats cost `ParamCost(0, 1)`
  - non-integer floats cost `ParamCost(1, 0)`
- `String` leaves cost `ParamCost(0, 1)`
- `Num(...)` and `Num.var(...)` nodes add no overhead themselves
- `+`, `-`, `*`, `/`, `**`, `exp`, `log`, `sqrt`, and `abs` each add one
  `ops_and_ints`
- total cost is the node overhead plus child costs

As a consequence, a variable contributes one `ops_and_ints` through its string
child, and `Num(1.0)` contributes one `ops_and_ints` through its `f64` child.

`container_cost_model` is defined relative to the decoded binary form of the
container representation, not necessarily the original pre-lowering binary
spelling.

Monomial cost:

- factor cost is the decoded cost of each exponent case:
  - exponent `1`: cost of the term
  - exponent `1/2`: cost of the term plus one `sqrt`
  - other exponents: cost of the term plus one `**` plus the cost of the
    absolute exponent literal
- numerator factors add multiplication costs between them
- if there are no numerator factors, the numerator contributes the cost of
  `Num(1.0)`
- denominator factors add multiplication costs between themselves plus one
  division at the monomial level

Polynomial cost:

- each monomial term is costed relative to its coefficient
- an empty monomial costs the coefficient alone
- if the coefficient is `1.0`, the coefficient is elided (unless the monomial is empty)
- if every exponent in the monomial is negative, the current implementation also
  elides the coefficient in the cost model, because the coefficient will appear in the numerator instead of Num(1.0).
- otherwise the term cost is `coef * monomial_body`, so the coefficient cost and
  one multiplication are added
- later polynomial terms add one `+` or `-` cost at the polynomial layer
- later negative coefficients are costed using their absolute value in the term
  and the subtraction cost at the polynomial layer
- an empty polynomial costs like `Num(0.0)`

The tested cost invariants are:

- for an extracted container `c`, `container_cost_model(c).floats` should equal
  the float count of the extracted binary expression `containers_to_binary(c)`
- after decoding an extracted container to binary, re-lowering it to an
  extracted container `c2`, and decoding `c2` again, the full `ParamCost`
  should match between `c2` and that second decoded binary form

So the container representation is expected to preserve parameter count for its
decoded binary meaning immediately, and to match the full decoded binary cost of
its own re-lowered extracted form, even though the first decoded binary tree and
the raw container syntax itself are not required to be fixed points.
"""

# mypy: disable-error-code="empty-body"

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias, TypeVar, cast

from egglog import *


class Num(Expr):
    """
    Paper EqSat language subset.

    This is deliberately closer to `FixTree`'s `SRTreeF` than to the broader
    experimental translations that were removed during cleanup. The paper
    corpus only needs constants,
    variables, arithmetic, and a small unary-function set.
    """

    def __init__(self, value: f64Like) -> None: ...

    __match_args__ = ("value",)

    @method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def value(self) -> f64:
        match get_callable_args(self, Num):
            case (value,):
                return cast("f64", value)
        raise ExprValueError(self, "Num")

    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    def __add__(self, other: NumLike) -> Num: ...

    def __sub__(self, other: NumLike) -> Num: ...

    def __mul__(self, other: NumLike) -> Num: ...

    def __truediv__(self, other: NumLike) -> Num: ...

    def __pow__(self, other: NumLike) -> Num: ...

    def __abs__(self) -> Num: ...

    def __radd__(self, other: NumLike) -> Num: ...

    def __rsub__(self, other: NumLike) -> Num: ...

    def __rmul__(self, other: NumLike) -> Num: ...

    def __rtruediv__(self, other: NumLike) -> Num: ...

    def __rpow__(self, other: NumLike) -> Num: ...

    @method(preserve=True)
    def __neg__(self) -> Num:
        return Num(-1.0) * self


@function
def exp(num: NumLike) -> Num: ...


@function
def log(num: NumLike) -> Num: ...


@function
def sqrt(num: NumLike) -> Num: ...


@function
def polynomial(p: ContainerPolynomialLike) -> Num: ...


ContainerMonomial: TypeAlias = Map[Num, BigRat]
ContainerPolynomial: TypeAlias = Map[ContainerMonomial, f64]

ContainerMonomialLike: TypeAlias = MapLike[Num, BigRat, "NumLike", BigRatLike]
ContainerPolynomialLike: TypeAlias = MapLike[ContainerMonomial, f64, ContainerMonomialLike, f64Like]
NumLike: TypeAlias = Num | StringLike | f64Like | i64Like | ContainerMonomialLike | ContainerPolynomialLike

converter(f64, Num, Num)
converter(i64, Num, lambda value: Num(f64.from_i64(value)))
converter(String, Num, Num.var)
converter(ContainerPolynomial, Num, polynomial)
converter(ContainerMonomial, Num, lambda mono: polynomial(ContainerPolynomial.empty().insert(mono, f64(1.0))))


def parse_expression(source: str) -> Num:
    """
    Parse a string of the expression syntax into a `Num` expression.
    """
    return convert(_from_ast(ast.parse(_normalize_expression_source(source), mode="eval")), Num)


def render_num(num: Num) -> str:
    """Render a `Num` back into a Python-like surface syntax for reports."""
    # parse and unparse to remove redundant parentheses and spacing.
    return ast.unparse(ast.parse(_render_num(num), mode="eval"))


def count_params(num: Num) -> int:
    """Count non-integer floating constants in a `Num` using the domain cost model."""
    _, cost = EGraph().extract(num, include_cost=True, cost_model=param_cost_model)
    return cost.floats


def binary_to_containers(expr: Num) -> Num:
    """
    Convert a binary expression to its container form.
    """
    return convert(_binary_to_containers(expr), Num)


def containers_to_binary(num: Num) -> Num:
    """
    Convert a container expression back to its binary form.

    Should be inverse of `binary_to_containers` (modulo ordering)
    """
    match get_callable_args(num, polynomial):
        case (poly,):
            return _decode_container_polynomial(cast("ContainerPolynomial", poly))
    fn = get_callable_fn(num)
    if fn in (Num, Num.var):
        return num
    args = get_callable_args(num)
    return fn(*map(containers_to_binary, args))  # type: ignore


BASE_EXPR = TypeVar("BASE_EXPR", bound=BaseExpr)


def validate_is_containers(num: BASE_EXPR) -> BASE_EXPR:
    """
    Validate that a `Num` is in container form, i.e. that it only uses the `polynomial` function and does not use any of the binary operators or other functions.
    """
    if (fn := get_callable_fn(num)) in [Num.__add__, Num.__sub__, Num.__mul__, Num.__truediv__, sqrt]:
        raise ValueError(f"Expected container form, but found use of {fn} in {num}")
    match get_callable_args(num, Num.__pow__):  # type: ignore[arg-type]
        case (_base, e) if get_callable_fn(cast("Num", e)) == Num:
            raise ValueError(f"Expected container form, but found use of ** with constant exponent in {num}")
    for arg in get_callable_args(num) or []:
        validate_is_containers(arg)
    return num


def validate_is_binary(expr: BASE_EXPR) -> BASE_EXPR:
    """
    Validate that a `Num` is in binary form, i.e. that it only uses the binary operators and does not use the `polynomial` function.
    """
    if get_callable_fn(expr) == polynomial:
        raise ValueError(f"Expected binary form, but found use of polynomial in {expr}")
    for arg in get_callable_args(expr) or []:
        validate_is_binary(arg)
    return expr


def _from_ast(node: ast.AST) -> Num:  # noqa: C901, PLR0911, PLR0912
    """
    Parse a subset of Python expressions into the `Num` DSL.

    Keep things as floats for as long as possible, so that when we convert to containers we know which terms are constants without running them through the e-graph.
    """
    if isinstance(node, ast.Expression):
        return _from_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, float | int):
            return Num(float(node.value))
        msg = f"Unsupported constant: {node.value!r}"
        raise ValueError(msg)
    if isinstance(node, ast.Name):
        return Num.var(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return Num(float(-node.operand.value))
        return -_from_ast(node.operand)
    if isinstance(node, ast.BinOp):
        lhs = _from_ast(node.left)
        rhs = _from_ast(node.right)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.Div):
            return lhs / rhs
        if isinstance(node.op, ast.Pow):
            # Only support powers of constants
            assert get_callable_fn(rhs) == Num
            return lhs**rhs
        msg = f"Unsupported binary operator: {ast.dump(node.op)}"
        raise TypeError(msg)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            msg = f"Unsupported call target: {ast.dump(node.func)}"
            raise TypeError(msg)
        name = node.func.id
        (arg,) = [_from_ast(arg) for arg in node.args]
        if name == "exp":
            return exp(arg)
        if name == "log":
            return log(arg)
        if name == "sqrt":
            return sqrt(arg)
        if name == "abs":
            return arg.__abs__()
        if name == "plog":
            return log(arg.__abs__())
        if name == "square":
            return arg**2
        if name == "cube":
            return arg**3
        msg = f"Unsupported function call: {name}"
        raise ValueError(msg)
    msg = f"Unsupported AST node: {ast.dump(node)}"
    raise TypeError(msg)


def _normalize_expression_source(source: str) -> str:
    normalized = source.strip()
    replacements = {
        "Log(": "log(",
        "Exp(": "exp(",
        "Sqrt(": "sqrt(",
        "Abs(": "abs(",
        "^": "**",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _binary_to_containers(expr: Num) -> Num | ContainerPolynomial | ContainerMonomial:
    """
    Turn all instances of *, +, etc. into container expressions.
    """
    if get_callable_fn(expr) in {Num.var, Num}:
        return expr
    match get_callable_args(expr, Num.__add__):
        case (lhs, rhs):
            return map_merge_with(
                lambda a, b: a + b,
                _to_container_poly(_binary_to_containers(cast("Num", lhs))),
                _to_container_poly(_binary_to_containers(cast("Num", rhs))),
            )
    match get_callable_args(expr, Num.__sub__):
        case (lhs, rhs):
            return map_merge_with(
                lambda a, b: a + b,
                _to_container_poly(_binary_to_containers(cast("Num", lhs))),
                map_map_values(lambda _, v: -v, _to_container_poly(_binary_to_containers(cast("Num", rhs)))),
            )
    match get_callable_args(expr, Num.__mul__):
        case (lhs, rhs):
            lhs_mapped = _binary_to_containers(cast("Num", lhs))
            rhs_mapped = _binary_to_containers(cast("Num", rhs))
            return map_merge_with(lambda a, b: a + b, _to_container_mono(lhs_mapped), _to_container_mono(rhs_mapped))
    match get_callable_args(expr, Num.__truediv__):
        case (lhs, rhs):
            denom = map_map_values(lambda _, v: -v, _to_container_mono(_binary_to_containers(cast("Num", rhs))))
            # If the numerator is just one, then dont add this as a term to the polynomial
            if cast("Num", lhs) == Num(1.0):
                return denom
            num = _to_container_mono(_binary_to_containers(cast("Num", lhs)))
            return map_merge_with(lambda a, b: a + b, num, denom)
    match get_callable_args(expr, Num.__pow__):
        case (n, Num(f64(f))):
            n_mapped = _to_num(_binary_to_containers(cast("Num", n)))
            if f == 1:
                return n_mapped
            return ContainerMonomial.empty().insert(n_mapped, BigRat.from_f64(f))
    match get_callable_args(expr, exp):
        case (inner,):
            return exp(_to_num(_binary_to_containers(cast("Num", inner))))
    match get_callable_args(expr, log):
        case (inner,):
            return log(_to_num(_binary_to_containers(cast("Num", inner))))
    match get_callable_args(expr, sqrt):
        case (n,):
            return ContainerMonomial.empty().insert(_to_num(_binary_to_containers(cast("Num", n))), BigRat(1, 2))
    match get_callable_args(expr, Num.__abs__):
        case (inner,):
            return abs(_to_num(_binary_to_containers(inner)))
    raise ValueError(f"Cannot decode to container: {expr}")


def _to_container_poly(v: Num | ContainerPolynomial | ContainerMonomial | f64) -> ContainerPolynomial:
    if is_expr_instance(v, ContainerPolynomial):
        return v
    return ContainerPolynomial.empty().insert(_to_container_mono(v), f64(1.0))


def _to_container_mono(v: Num | ContainerPolynomial | ContainerMonomial | f64) -> ContainerMonomial:
    if is_expr_instance(v, ContainerMonomial):
        return v
    return ContainerMonomial.empty().insert(_to_num(v), BigRat(1, 1))


def _to_num(v: ContainerMonomial | ContainerPolynomial | Num | f64) -> Num:
    if isinstance(v, Num):
        return v
    if is_expr_instance(v, ContainerPolynomial):
        return polynomial(v)
    if is_expr_instance(v, ContainerMonomial):
        return polynomial(ContainerPolynomial.empty().insert(v, f64(1.0)))
    return Num(v)


def _render_float(value: float) -> str:
    if value == 0.0:
        return "0.0"
    if value.is_integer():
        return f"{value:.1f}"
    return repr(value)


def _render_num(num: Num) -> str:  # noqa: C901, PLR0911
    match get_callable_args(num, polynomial):
        case (poly,) if isinstance(poly, Map):
            return _render_num(containers_to_binary(num))
    match get_callable_args(num, Num):
        case (f64(f),):
            res = _render_float(f)
            if f < 0.0:
                res = f"({res})"
            return res
    match get_callable_args(num, Num.var):
        case (String(s),):
            return s
    match get_callable_args(num, Num.__add__):
        case (lhs, rhs):
            return f"({_render_num(cast('Num', lhs))} + {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            return f"({_render_num(cast('Num', lhs))} - {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            return f"({_render_num(cast('Num', lhs))} * {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            return f"({_render_num(cast('Num', lhs))} / {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            return f"({_render_num(cast('Num', lhs))} ** {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, exp):
        case (inner,):
            return f"exp({_render_num(cast('Num', inner))})"
    match get_callable_args(num, log):
        case (inner,):
            return f"log({_render_num(cast('Num', inner))})"
    match get_callable_args(num, sqrt):
        case (inner,):
            return f"sqrt({_render_num(cast('Num', inner))})"
    match get_callable_args(num, Num.__abs__):
        case (inner,):
            return f"abs({_render_num(inner)})"
    msg = f"Unsupported Num node for rendering: {num!r}"
    raise TypeError(msg)


def _product(factors: list[Num], initial: float = 1.0) -> Num:
    if not factors:
        return Num(initial)
    total = factors.pop(0) if initial == 1.0 else Num(initial)
    for next_factor in factors:
        total *= next_factor
    return total


def _decode_container_mono_term(mono: dict[Num, BigRat], coef: float) -> Num:
    if not mono:
        return Num(coef)
    numerator_factors: list[Num] = []
    denominator_factors: list[Num] = []
    for term, exp in mono.items():
        exp_value = exp.value
        term_decoded = containers_to_binary(term)
        abs_exp_value = abs(exp_value)
        factor = (
            sqrt(term_decoded)
            if abs_exp_value == Fraction(1, 2)
            else term_decoded
            if abs_exp_value == 1
            else term_decoded ** float(abs_exp_value)
        )
        (denominator_factors if exp_value < 0 else numerator_factors).append(factor)
    numerator = _product(numerator_factors, coef)
    if not denominator_factors:
        return numerator
    return numerator / _product(denominator_factors)


def _decode_container_polynomial(poly: ContainerPolynomial) -> Num:
    """
    Decode a polynomial into binary ops. Negative coefficients are turned into subtraction and 1.0 coefficients are elided.
    """
    poly_items = [(mono.value, float(coef)) for (mono, coef) in poly.value.items()]
    if not poly_items:
        return Num(0.0)
    mono, coef = poly_items.pop(0)
    total = _decode_container_mono_term(mono, coef)
    for mono, coef in poly_items:
        if coef < 0.0:
            total -= _decode_container_mono_term(mono, abs(coef))
        else:
            total += _decode_container_mono_term(mono, coef)
    return total


@dataclass(frozen=True, order=True)
class ParamCost:
    """
    Custom cost type that prioritizes minimizing number of floats (which correspond to fitted parameters), and then on ties
    minimizes the sum of ops and ints (which correspond to the complexity or cost of the operation).
    """

    # count of floats
    floats: int = 0
    # count of +, *, /, **, exp, log, sqrt, abs operations plus any floats that can be parsed as ints (like 1, 2, etc)
    ops_and_ints: int = 0

    @property
    def node_count(self) -> int:
        return self.floats + self.ops_and_ints

    def __add__(self, other: ParamCost) -> ParamCost:
        return ParamCost(
            floats=self.floats + other.floats,
            ops_and_ints=self.ops_and_ints + other.ops_and_ints,
        )

    def __str__(self) -> str:
        return f"ParamCost({self.floats}, {self.ops_and_ints})"

    def __repr__(self) -> str:
        return str(self)


def _float_cost(f: float) -> ParamCost:
    """
    Dont count integer floats as floats, since they aren't parameters
    """
    if f.is_integer():
        return ParamCost(ops_and_ints=1)
    return ParamCost(floats=1)


def param_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[ParamCost]) -> ParamCost:
    if isinstance(expr, f64):
        return _float_cost(float(expr))
    if isinstance(expr, String):
        return ParamCost(ops_and_ints=1)
    fn = get_callable_fn(expr)
    if fn in (Num, Num.var):
        cost = 0
    elif fn in (Num.__add__, Num.__sub__, Num.__mul__, Num.__truediv__, Num.__pow__, exp, log, sqrt, Num.__abs__):
        cost = 1
    else:
        raise ValueError(f"Unsupported expression in cost model: {expr}")
    return sum(children_costs, start=ParamCost(ops_and_ints=cost))


def _decoded_monomial_cost(
    egraph: EGraph,
    mono: ContainerMonomial,
    children_costs: list[ParamCost],
) -> ParamCost:
    """
    Like _decode_container_mono_term. Assumes that if we have an empty numerator we include the 1.0
    """
    items = list(mono.value.items())
    if len(children_costs) != len(items) * 2:
        msg = f"Expected {len(items) * 2} monomial child costs, got {len(children_costs)}"
        raise ValueError(msg)

    numerator_factor_costs: list[ParamCost] = []
    denominator_factor_costs: list[ParamCost] = []
    for i, (_, exp) in enumerate(items):
        exp_value = exp.value
        abs_exp_value = abs(exp_value)
        term_cost = children_costs[i * 2]
        # factor cost
        factor_cost = (
            # sqrt is one op plus the inside
            term_cost + ParamCost(ops_and_ints=1)
            if abs_exp_value == Fraction(1, 2)
            else term_cost
            if abs_exp_value == 1
            else term_cost + ParamCost(ops_and_ints=1) + _float_cost(float(abs_exp_value))
        )
        (denominator_factor_costs if exp_value < 0 else numerator_factor_costs).append(factor_cost)

    numerator_cost = (
        sum(numerator_factor_costs, ParamCost(ops_and_ints=len(numerator_factor_costs) - 1))
        if numerator_factor_costs
        else ParamCost(ops_and_ints=1)
    )
    if not denominator_factor_costs:
        return numerator_cost
    denominator_cost = sum(denominator_factor_costs, ParamCost(ops_and_ints=len(denominator_factor_costs)))
    return numerator_cost + denominator_cost


def _decoded_polynomial_term_cost(
    egraph: EGraph, mono: dict[Num, BigRat], coef: float, mono_cost: ParamCost
) -> ParamCost:
    """
    Gives the cost of one monomial and its coefficient based on the cost of the monomial.

    mirrors _decode_container_mono_term
    """
    coef_cost = _float_cost(coef)
    if not mono:
        return coef_cost
    has_empty_numerator = all(exp.value < 0 for exp in mono.values())
    # If we have an empty numerator, then the coefficient will end up there
    if coef == 1.0 or has_empty_numerator:
        return mono_cost
    # if we are multiplying them, add their costs and the cost of the mul
    return coef_cost + mono_cost + ParamCost(ops_and_ints=1)


def _decoded_polynomial_cost(
    egraph: EGraph,
    poly: ContainerPolynomial,
    children_costs: list[ParamCost],
) -> ParamCost:
    """
    Should correspond to getting the cost from the return value of _decode_container_polynomial
    """
    items = list(poly.value.items())
    if len(children_costs) != len(items) * 2:
        msg = f"Expected {len(items) * 2} polynomial child costs, got {len(children_costs)}"
        raise ValueError(msg)
    if not items:
        # empty is zero
        return ParamCost(ops_and_ints=1)

    mono, coef = items[0]
    total = _decoded_polynomial_term_cost(egraph, mono.value, float(coef), children_costs[0])
    for i, (mono, coef) in enumerate(items[1:], start=1):
        term_cost = _decoded_polynomial_term_cost(egraph, mono.value, abs(float(coef)), children_costs[i * 2])
        # the cost is the cost of the monomial plus the cost of an add/sub
        total += term_cost + ParamCost(ops_and_ints=1)
    return total


# Container specific cost model that should give the same cost as the default cost model on the decoded expression.
def container_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[ParamCost]) -> ParamCost:
    if is_expr_instance(expr, ContainerPolynomial):
        return _decoded_polynomial_cost(egraph, expr, children_costs)
    if is_expr_instance(expr, ContainerMonomial):
        return _decoded_monomial_cost(egraph, expr, children_costs)
    if get_callable_fn(expr) == polynomial:
        return children_costs[0]
    if isinstance(expr, BigRat):
        return ParamCost(ops_and_ints=1)
    return param_cost_model(egraph, expr, children_costs)
