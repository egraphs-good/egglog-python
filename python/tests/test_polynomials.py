from egglog import *
from egglog.exp.array_api import factor_ruleset, from_polynomial_ruleset, to_polynomial_ruleset
from egglog.exp.polynomials import distribute, remove_subtraction, symbolic_bending_examples


def _factor_example(expr):
    egraph = EGraph()
    x = egraph.let("x", expr)
    egraph.run(to_polynomial_ruleset.saturate() + factor_ruleset.saturate() + from_polynomial_ruleset.saturate())
    factored = egraph.extract(x)
    egraph.check(eq(x).to(factored))
    return factored


def test_factor_multisets(snapshot_py):
    function_bending, _gradient_bending = symbolic_bending_examples()
    # remove subtraction and distribute first:
    egraph = EGraph()
    egraph.register(function_bending)
    egraph.run(remove_subtraction.saturate())
    egraph.run(distribute.saturate())
    distributed = egraph.extract(function_bending)
    factored = _factor_example(distributed)
    assert str(factored) == snapshot_py(name="code")
