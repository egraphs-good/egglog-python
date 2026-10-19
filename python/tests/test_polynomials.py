import operator
from fractions import Fraction

from egglog import *
from egglog.exp.array_api import Value, factor_ruleset, from_polynomial_ruleset, to_polynomial_ruleset
from egglog.exp.polynomials import distribute, remove_subtraction, symbolic_bending_examples


def _evaluate_scalar(value, variables):
    operators = {
        Value.__add__: operator.add,
        Value.__sub__: operator.sub,
        Value.__mul__: operator.mul,
        Value.__truediv__: operator.truediv,
        Value.__pow__: operator.pow,
    }
    memo = {}

    def visit(expression):
        reference = expression.__egg_ref__
        key = (id(reference.owner), reference.index)
        if key in memo:
            return memo[key]
        fn = get_callable_fn(expression)
        args = get_callable_args(expression)
        assert args is not None
        if fn == Value.var:
            (name,) = args
            result = Fraction(variables[name.value]), 0
        elif fn in (Value.from_int, Value.from_float):
            result = Fraction(expression.value), 0
        else:
            assert fn in operators, str(expression)
            left, right = args
            lhs, lhs_ops = visit(left)
            rhs, rhs_ops = visit(right)
            if fn == Value.__pow__:
                assert rhs.denominator == 1, rhs
                rhs = int(rhs)
            result = operators[fn](lhs, rhs), lhs_ops + rhs_ops + 1
        memo[key] = result
        return result

    return visit(value)


def _factor_example(expr):
    egraph = EGraph()
    x = egraph.let("x", expr)
    egraph.run(to_polynomial_ruleset.saturate() + factor_ruleset.saturate() + from_polynomial_ruleset.saturate())
    factored = egraph.extract(x)
    egraph.check(eq(x).to(factored))
    return factored


def test_factor_multisets():
    function_bending, _gradient_bending = symbolic_bending_examples()
    # remove subtraction and distribute first:
    egraph = EGraph()
    egraph.register(function_bending)
    egraph.run(remove_subtraction.saturate())
    egraph.run(distribute.saturate())
    distributed = egraph.extract(function_bending)
    factored = _factor_example(distributed)
    for sample in range(3):
        variables = {f"q{i}": i * i + sample * (i % 3 - 1) for i in range(1, 13)}
        variables.update({f"bp{i}": value + sample for i, value in enumerate((1, 2, -1, 3), 1)})
        variables.update({f"bpp{i}": value - sample for i, value in enumerate((2, -1, 4, 1), 1)})
        original_value, _ = _evaluate_scalar(function_bending.value, variables)
        distributed_value, distributed_ops = _evaluate_scalar(distributed.value, variables)
        factored_value, factored_ops = _evaluate_scalar(factored.value, variables)
        assert original_value == distributed_value == factored_value
        assert factored_ops < distributed_ops
