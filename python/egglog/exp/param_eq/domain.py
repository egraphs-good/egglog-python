"""
Expression domain for the retained Param-Eq stress cases.

This module defines the restricted symbolic language, parser and renderer,
binary/container conversions, and lexicographic extraction costs. Container
polynomials use nested `Map` values to canonicalize repeated terms and factors;
lowering and decoding preserve supported expression meaning, not exact tree
shape. The detailed research method and restart notes live in
`experiments/param_eq/NOTES.md`.
"""

# mypy: disable-error-code="empty-body"

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias, TypeVar, cast

from typing_extensions import TypeIs

from egglog import *

from ...runtime import RuntimeClass, RuntimeExpr  # noqa: TID252

_T_EXPR = TypeVar("_T_EXPR", bound=BaseExpr)


def _is_expr_instance(x: BaseExpr, cls: type[_T_EXPR]) -> TypeIs[_T_EXPR]:
    """Check an expression's concrete Egglog type, including generic arguments."""
    if not isinstance(cast("object", x), RuntimeExpr):
        raise TypeError(f"Expected Expression, got {type(x).__name__}")
    if not isinstance(cast("object", cls), RuntimeClass):
        raise TypeError(f"Expected expression class, got {type(cls).__name__}")
    return cast("RuntimeExpr", x).__egg_typed_expr__.tp == cast("RuntimeClass", cls).__egg_tp__.to_just()


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
    if fn is None or args is None:
        raise ValueError(f"Cannot decode container expression: {num}")
    constructor = cast("Callable[..., Num]", fn)
    return constructor(*(containers_to_binary(cast("Num", arg)) for arg in args))


def _finite_num_literal(value: float) -> Num:
    """Construct a numeric literal while enforcing the surface language's finite-value invariant."""
    finite_value = float(value)
    if not math.isfinite(finite_value):
        msg = "Numeric literals must be finite"
        raise ValueError(msg)
    return Num(finite_value)


def _from_ast(node: ast.AST) -> Num:  # noqa: C901, PLR0911, PLR0912
    """
    Parse a subset of Python expressions into the `Num` DSL.

    Keep things as floats for as long as possible, so that when we convert to containers we know which terms are constants without running them through the e-graph.
    """
    if isinstance(node, ast.Expression):
        return _from_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, float | int) and not isinstance(node.value, bool):
            return _finite_num_literal(node.value)
        msg = f"Unsupported constant: {node.value!r}"
        raise ValueError(msg)
    if isinstance(node, ast.Name):
        return Num.var(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if (
            isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int | float)
            and not isinstance(node.operand.value, bool)
        ):
            return _finite_num_literal(-node.operand.value)
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
            if get_callable_fn(rhs) != Num:
                msg = "Power exponent must be a numeric literal"
                raise ValueError(msg)
            return lhs**rhs
        msg = f"Unsupported binary operator: {ast.dump(node.op)}"
        raise TypeError(msg)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            msg = f"Unsupported call target: {ast.dump(node.func)}"
            raise TypeError(msg)
        name = node.func.id
        if node.keywords:
            msg = f"Function call {name!r} does not accept keyword arguments"
            raise ValueError(msg)
        if len(node.args) != 1:
            msg = f"Function call {name!r} expects exactly one argument"
            raise ValueError(msg)
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


def _float_to_bigrat(value: float) -> BigRat:
    """Preserve a Python float's exact binary value as a rational exponent."""
    numerator, denominator = value.as_integer_ratio()
    return BigRat(numerator, denominator)


def _binary_to_containers(  # noqa: C901, PLR0911, PLR0912
    expr: Num,
) -> Num | ContainerPolynomial | ContainerMonomial:
    """
    Turn all instances of *, +, etc. into container expressions.
    """
    if get_callable_fn(expr) in {Num.var, Num}:
        return expr
    match get_callable_args(expr, Num.__add__):
        case (lhs, rhs):
            return map_fold_kv(
                lambda result, mono, coef: catch(lambda: result[mono]).match(
                    lambda old_coef: result.insert(mono, old_coef + coef), result.insert(mono, coef)
                ),
                _to_container_poly(_binary_to_containers(lhs)),
                _to_container_poly(_binary_to_containers(cast("Num", rhs))),
            )
    match get_callable_args(expr, Num.__sub__):
        case (lhs, rhs):
            return map_fold_kv(
                lambda result, mono, coef: catch(lambda: result[mono]).match(
                    lambda old_coef: result.insert(mono, old_coef + coef), result.insert(mono, coef)
                ),
                _to_container_poly(_binary_to_containers(lhs)),
                map_fold_kv(
                    lambda result, mono, coef: result.insert(mono, -coef),
                    ContainerPolynomial.empty(),
                    _to_container_poly(_binary_to_containers(cast("Num", rhs))),
                ),
            )
    match get_callable_args(expr, Num.__mul__):
        case (lhs, rhs):
            lhs_mapped = _binary_to_containers(lhs)
            rhs_mapped = _binary_to_containers(cast("Num", rhs))
            lhs_is_polynomial = _is_expr_instance(lhs_mapped, ContainerPolynomial)
            rhs_is_polynomial = _is_expr_instance(rhs_mapped, ContainerPolynomial)
            if rhs_is_polynomial and not lhs_is_polynomial:
                lhs_mapped, rhs_mapped = rhs_mapped, lhs_mapped
                lhs_is_polynomial, rhs_is_polynomial = rhs_is_polynomial, lhs_is_polynomial
            if lhs_is_polynomial and not rhs_is_polynomial:
                lhs_poly = cast("ContainerPolynomial", lhs_mapped)
                match get_callable_args(rhs_mapped, Num):
                    case (f64(scalar),):
                        return map_fold_kv(
                            lambda result, mono, coef: result.insert(mono, coef * scalar),
                            ContainerPolynomial.empty(),
                            lhs_poly,
                        )
                return _multiply_container_polynomial_by_monomial(lhs_poly, _to_container_mono(rhs_mapped))
            return map_fold_kv(
                lambda result, term, exponent: catch(lambda: result[term]).match(
                    lambda old_exponent: result.insert(term, old_exponent + exponent),
                    result.insert(term, exponent),
                ),
                _to_container_mono(lhs_mapped),
                _to_container_mono(rhs_mapped),
            )
    match get_callable_args(expr, Num.__truediv__):
        case (lhs, rhs):
            lhs_mapped = _binary_to_containers(lhs)
            rhs_mapped = _binary_to_containers(cast("Num", rhs))
            lhs_is_polynomial = _is_expr_instance(lhs_mapped, ContainerPolynomial)
            rhs_is_polynomial = _is_expr_instance(rhs_mapped, ContainerPolynomial)
            if lhs_is_polynomial:
                lhs_poly = cast("ContainerPolynomial", lhs_mapped)
                match get_callable_args(rhs_mapped, Num):
                    case (f64(scalar),):
                        return map_fold_kv(
                            lambda result, mono, coef: result.insert(mono, coef / scalar),
                            ContainerPolynomial.empty(),
                            lhs_poly,
                        )
            denom = map_fold_kv(
                lambda result, term, exponent: result.insert(term, -exponent),
                ContainerMonomial.empty(),
                _to_container_mono(rhs_mapped),
            )
            if lhs_is_polynomial and not rhs_is_polynomial:
                return _multiply_container_polynomial_by_monomial(cast("ContainerPolynomial", lhs_mapped), denom)
            # If the numerator is just one, then dont add this as a term to the polynomial
            if _is_expr_instance(lhs_mapped, Num) and lhs_mapped == Num(1.0):
                return denom
            num = _to_container_mono(lhs_mapped)
            return map_fold_kv(
                lambda result, term, exponent: catch(lambda: result[term]).match(
                    lambda old_exponent: result.insert(term, old_exponent + exponent),
                    result.insert(term, exponent),
                ),
                num,
                denom,
            )
    match get_callable_args(expr, Num.__pow__):
        case (n, Num(f64(f))):
            n_mapped = _to_num(_binary_to_containers(n))
            if f == 1:
                return n_mapped
            return ContainerMonomial.empty().insert(n_mapped, _float_to_bigrat(f))
    match get_callable_args(expr, exp):
        case (inner,):
            return exp(_to_num(_binary_to_containers(cast("Num", inner))))
    match get_callable_args(expr, log):
        case (inner,):
            return log(_to_num(_binary_to_containers(cast("Num", inner))))
    match get_callable_args(expr, sqrt):
        case (inner,):
            return ContainerMonomial.empty().insert(_to_num(_binary_to_containers(cast("Num", inner))), BigRat(1, 2))
    match get_callable_args(expr, Num.__abs__):
        case (inner,):
            return abs(_to_num(_binary_to_containers(inner)))
    raise ValueError(f"Cannot decode to container: {expr}")


def _multiply_container_polynomial_by_monomial(
    poly: ContainerPolynomial, factor: ContainerMonomial
) -> ContainerPolynomial:
    """Distribute one monomial into a polynomial and combine coefficient collisions."""
    return map_fold_kv(
        lambda result, mono, coef: map_fold_kv(
            lambda merged_poly, merged_mono, new_coef: catch(lambda: merged_poly[merged_mono]).match(
                lambda old_coef: merged_poly.insert(merged_mono, old_coef + new_coef),
                merged_poly.insert(merged_mono, new_coef),
            ),
            result,
            ContainerPolynomial.empty().insert(
                map_fold_kv(
                    lambda merged_mono, term, exponent: catch(lambda: merged_mono[term]).match(
                        lambda old_exponent: merged_mono.insert(term, old_exponent + exponent),
                        merged_mono.insert(term, exponent),
                    ),
                    mono,
                    factor,
                ),
                coef,
            ),
        ),
        ContainerPolynomial.empty(),
        poly,
    )


def _to_container_poly(v: Num | ContainerPolynomial | ContainerMonomial | f64) -> ContainerPolynomial:
    if _is_expr_instance(v, ContainerPolynomial):
        return v
    return ContainerPolynomial.empty().insert(_to_container_mono(v), f64(1.0))


def _to_container_mono(v: Num | ContainerPolynomial | ContainerMonomial | f64) -> ContainerMonomial:
    if _is_expr_instance(v, ContainerMonomial):
        return v
    return ContainerMonomial.empty().insert(_to_num(v), BigRat(1, 1))


def _to_num(v: ContainerMonomial | ContainerPolynomial | Num | f64) -> Num:
    if isinstance(v, Num):
        return v
    if _is_expr_instance(v, ContainerPolynomial):
        return polynomial(v)
    if _is_expr_instance(v, ContainerMonomial):
        return polynomial(ContainerPolynomial.empty().insert(v, f64(1.0)))
    return Num(v)


def _render_float(value: float) -> str:
    if not math.isfinite(value):
        msg = "Cannot render a non-finite Param-Eq result"
        raise ValueError(msg)
    if value == 0.0:
        return "0.0"
    if value.is_integer():
        return f"{value:.1f}"
    return repr(value)


def _render_num(num: Num) -> str:  # noqa: C901, PLR0911, PLR0912
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
            return f"({_render_num(lhs)} + {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__sub__):
        case (lhs, rhs):
            return f"({_render_num(lhs)} - {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__mul__):
        case (lhs, rhs):
            return f"({_render_num(lhs)} * {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__truediv__):
        case (lhs, rhs):
            return f"({_render_num(lhs)} / {_render_num(cast('Num', rhs))})"
    match get_callable_args(num, Num.__pow__):
        case (lhs, rhs):
            return f"({_render_num(lhs)} ** {_render_num(cast('Num', rhs))})"
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


def _decoded_monomial_cost(mono: ContainerMonomial, children_costs: list[ParamCost]) -> ParamCost:
    """
    Like _decode_container_mono_term. Assumes that if we have an empty numerator we include the 1.0
    """
    items = list(mono.value.items())
    # Cost-model callbacks receive Map children in Map.value.items() order: key, value, key, value, ...
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


def _decoded_polynomial_term_cost(mono: dict[Num, BigRat], coef: float, mono_cost: ParamCost) -> ParamCost:
    """
    Gives the cost of one monomial and its coefficient based on the cost of the monomial.

    mirrors _decode_container_mono_term
    """
    coef_cost = _float_cost(coef)
    if not mono:
        return coef_cost
    has_empty_numerator = all(exp.value < 0 for exp in mono.values())
    if coef == 1.0:
        return mono_cost
    if has_empty_numerator:
        # mono_cost includes the synthetic numerator `1`; decoding replaces
        # that with the coefficient, so charge the coefficient instead.
        return ParamCost(
            floats=mono_cost.floats + coef_cost.floats,
            ops_and_ints=mono_cost.ops_and_ints + coef_cost.ops_and_ints - 1,
        )
    # if we are multiplying them, add their costs and the cost of the mul
    return coef_cost + mono_cost + ParamCost(ops_and_ints=1)


def _decoded_polynomial_cost(poly: ContainerPolynomial, children_costs: list[ParamCost]) -> ParamCost:
    """
    Should correspond to getting the cost from the return value of _decode_container_polynomial
    """
    items = list(poly.value.items())
    # Cost-model callbacks receive Map children in Map.value.items() order: key, value, key, value, ...
    if len(children_costs) != len(items) * 2:
        msg = f"Expected {len(items) * 2} polynomial child costs, got {len(children_costs)}"
        raise ValueError(msg)
    if not items:
        # empty is zero
        return ParamCost(ops_and_ints=1)

    mono, coef = items[0]
    total = _decoded_polynomial_term_cost(mono.value, float(coef), children_costs[0])
    for i, (mono, coef) in enumerate(items[1:], start=1):
        term_cost = _decoded_polynomial_term_cost(mono.value, abs(float(coef)), children_costs[i * 2])
        # the cost is the cost of the monomial plus the cost of an add/sub
        total += term_cost + ParamCost(ops_and_ints=1)
    return total


# Container specific cost model that should give the same cost as the default cost model on the decoded expression.
def container_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[ParamCost]) -> ParamCost:
    if _is_expr_instance(expr, ContainerPolynomial):
        return _decoded_polynomial_cost(expr, children_costs)
    if _is_expr_instance(expr, ContainerMonomial):
        return _decoded_monomial_cost(expr, children_costs)
    if get_callable_fn(expr) == polynomial:
        return children_costs[0]
    if isinstance(expr, BigRat):
        return ParamCost(ops_and_ints=1)
    return param_cost_model(egraph, expr, children_costs)
