from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension

from egglog import *
from egglog.exp.polynomials import distribute, remove_subtraction, symbolic_bending_examples

from .array_api import factor_ruleset, from_polynomial_ruleset, to_polynomial_ruleset


def _factor_example(expr):
    egraph = EGraph(save_egglog_string=True)
    x = egraph.let("x", expr)
    egraph.run(to_polynomial_ruleset.saturate() + factor_ruleset.saturate() + from_polynomial_ruleset.saturate())
    factored = egraph.extract(x)
    egraph.check(eq(x).to(factored))
    return factored, egraph.as_egglog_string


class EgglogSnapshotExtension(SingleFileSnapshotExtension):
    file_extension = "egg"

    def serialize(self, data, **kwargs) -> bytes:
        return str(data).encode()


def test_factor_multisets(benchmark, snapshot_py, snapshot: SnapshotAssertion):
    function_bending, _gradient_bending = symbolic_bending_examples()
    # remove subtraction and distribute first:
    egraph = EGraph()
    egraph.register(function_bending)
    egraph.run(remove_subtraction.saturate())
    egraph.run(distribute.saturate())
    distributed = egraph.extract(function_bending)
    factored, egglog_str = benchmark(_factor_example, distributed)
    assert str(factored) == snapshot_py(name="code")
    assert egglog_str == snapshot.with_defaults(extension_class=EgglogSnapshotExtension)(name="egglog")
